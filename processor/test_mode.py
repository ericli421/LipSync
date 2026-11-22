"""Live separation test mode: dashboard settings, model loading, per-session separator."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.io import wavfile

from processor.separation import MODEL_RATE, MONITORS, LipPicker, StreamingSeparator

ROOT = Path(__file__).resolve().parents[1]
WINDOWS, HOPS = (1.0, 1.5, 2.0), (0.25, 0.5)
LOOKAHEAD_S, RECORD_SECONDS = 0.25, 300


def _number(message, key, low, high):
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"{key} must be a number in [{low}, {high}]")
    return float(value)


class TestMode:
    def __init__(self, loader=None, on_change=None, recordings_dir=ROOT / "recordings", log=None):
        if loader is None:
            from processor.separation_models import load as loader
        self.loader, self.on_change, self.recordings_dir = loader, on_change, Path(recordings_dir)
        self.log = log or (lambda *args, **kwargs: None)
        self.reported_skips = 0
        self.lock = threading.RLock()
        self.enabled, self.model = False, "auto"
        self.window_s, self.hop_s = 1.5, 0.25
        self.monitor = "original"
        self.focus_db, self.duck_db, self.lip_offset_ms = 3.0, -15.0, 0.0
        self.record = False
        self.state, self.error, self.info = "off", None, None
        self.separate, self.loaded_model, self.loading = None, None, None
        self.session = None          # dict(id, rate, anchor_us)
        self.separator, self.version, self.built_version = None, 0, -1
        self.picker = LipPicker()
        self.last_recording = None

    # ---- controls from viewers ----
    def update(self, message):
        kind = message.get("type")
        with self.lock:
            if kind == "set_test_mode":
                enabled = message.get("enabled")
                if not isinstance(enabled, bool):
                    raise ValueError("enabled must be boolean")
                model = message.get("model", self.model)
                from processor.separation_models import MODELS
                if model != "auto" and model not in MODELS:
                    raise ValueError(f"model must be auto or one of {sorted(MODELS)}")
                window = message.get("window_s", self.window_s)
                hop = message.get("hop_s", self.hop_s)
                if window not in WINDOWS or hop not in HOPS:
                    raise ValueError(f"window_s must be in {WINDOWS} and hop_s in {HOPS}")
                if model != self.model:
                    self.separate, self.loaded_model = None, None
                self.enabled, self.model, self.window_s, self.hop_s = enabled, model, float(window), float(hop)
                self.version += 1
                if not enabled:
                    self._close_separator()
                    self.state = "off"
                elif self.separate is None:
                    self._start_loading()
                elif self.state in ("off", "error"):
                    self.state, self.error = "ready", None
            elif kind == "set_monitor":
                if message.get("source") not in MONITORS:
                    raise ValueError(f"source must be one of {MONITORS}")
                self.monitor = message["source"]
            elif kind == "set_focus":
                self.focus_db = _number(message, "focus_db", 0, 12) if "focus_db" in message else self.focus_db
                self.duck_db = _number(message, "duck_db", -40, 0) if "duck_db" in message else self.duck_db
                if "lip_offset_ms" in message:
                    self.lip_offset_ms = _number(message, "lip_offset_ms", -500, 500)
                self.picker.offset_ms = self.lip_offset_ms
                if self.separator:
                    self.separator.focus_gain = 10 ** (self.focus_db / 20)
                    self.separator.duck_gain = 10 ** (self.duck_db / 20)
            elif kind == "set_recording":
                record = message.get("enabled")
                if not isinstance(record, bool):
                    raise ValueError("enabled must be boolean")
                if record and not self.record:
                    self.version += 1  # rebuild with a recording buffer
                if not record and self.record:
                    self._save_recording()
                self.record = record
            else:
                raise ValueError("unsupported control message")

    def _start_loading(self):
        if self.loading and self.loading.is_alive():
            return  # the running loader re-checks self.model when it finishes
        self.state, self.error = "loading", None

        def run():
            while True:
                with self.lock:
                    model = self.model
                self.log("info", "test", f"loading model {model}")
                began = time.perf_counter()
                try:
                    separate, info = self.loader(model)
                    error = None
                    where = info.get("gpu") or info.get("device")
                    self.log("info", "test", f"loaded {info.get('model')} on {where} in {time.perf_counter() - began:.1f} s")
                except Exception as exc:
                    separate, info, error = None, None, f"Model failed to load: {exc}"
                    self.log("error", "test", error)
                with self.lock:
                    if model != self.model:
                        continue  # changed while loading: load the new choice
                    if error:
                        self.state, self.error = "error", error
                    else:
                        self.separate, self.info, self.loaded_model = separate, info, model
                        self.state = "ready" if self.enabled else "off"
                        self.version += 1
                    break
            if self.on_change:
                self.on_change()

        self.loading = threading.Thread(target=run, name="separator-loader", daemon=True)
        self.loading.start()

    # ---- session / media ----
    def attach(self, session_id, rate):
        with self.lock:
            self._close_separator()
            self.session = {"id": session_id, "rate": rate, "anchor_us": None}
            self.picker = LipPicker(offset_ms=self.lip_offset_ms)
            self.version += 1

    def detach(self, session_id):
        with self.lock:
            if self.session and self.session["id"] == session_id:
                if self.record:
                    self._save_recording()
                    self.record = False
                self._close_separator()
                self.session = None
                if self.state == "running":
                    self.state = "ready"

    def add_mouth(self, capture_us, info):
        self.picker.add(capture_us, float(info["openness"]) if info.get("face_found") else None)

    def feed(self, pcm, first_sample, capture_us):
        """Called for every phone block (already filtered). Never raises."""
        with self.lock:
            if not self.session:
                return
            if self.session["anchor_us"] is None:
                self.session["anchor_us"] = capture_us - first_sample * 1_000_000 // self.session["rate"]
            if not self.enabled or self.separate is None:
                return
            try:
                if self.built_version != self.version or self.separator is None:
                    self._build()
                if self.separator.error:
                    raise RuntimeError(self.separator.error)
                self.separator.feed(np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0)
            except Exception as exc:
                self.state, self.error = "error", str(exc)
                self.enabled = False
                self._close_separator()
                self.log("error", "test", f"stopped: {exc}")

    def output(self, n):
        """Processed PCM16 bytes for the selected monitor, or None to relay the live chain."""
        with self.lock:
            separator, monitor = self.separator, self.monitor
        if separator is None or monitor == "original" or separator.error:
            return None
        y = separator.read(n, monitor)
        return (np.clip(y, -1.0, 32767 / 32768) * 32768.0).round().astype("<i2").tobytes()

    def _build(self):
        self._close_separator()
        session = self.session
        self.picker.choice, self.picker.reason = -1, "waiting for face"
        self.separator = StreamingSeparator(
            session["rate"], self.separate, window_s=self.window_s, hop_s=self.hop_s,
            lookahead_s=LOOKAHEAD_S, picker=self.picker, anchor_us=session["anchor_us"],
            focus_db=self.focus_db, duck_db=self.duck_db,
            record_seconds=RECORD_SECONDS if self.record else 0)
        self.built_version, self.state, self.error = self.version, "running", None
        self.reported_skips = 0
        delay_ms = self.separator.delay * 1000 // session["rate"]
        self.log("info", "test", f"separating: window {self.window_s} s, hop {self.hop_s} s, "
                 f"delay {delay_ms} ms, input {session['rate']} Hz")

    def _close_separator(self):
        if self.separator is not None:
            self.separator.close()
            self.separator = None

    def _save_recording(self):
        if self.separator is None:
            return
        data = self.separator.take_recording()
        if not data or not len(data["mix"]):
            return
        folder = self.recordings_dir / f"live-test-{datetime.now():%Y%m%d-%H%M%S}"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("mix", "voice1", "voice2"):
            if name in data and len(data[name]):
                wavfile.write(folder / f"{name}.wav", MODEL_RATE,
                              (np.clip(data[name], -1, 1) * 32767).astype(np.int16))
        picks = data.get("focus_pick", np.zeros(0))
        meta = {"sample_rate": MODEL_RATE, "seconds": round(len(data["mix"]) / MODEL_RATE, 2),
                "model": self.info, "window_s": self.window_s, "hop_s": self.hop_s,
                "lookahead_s": LOOKAHEAD_S, "lip_offset_ms": self.lip_offset_ms,
                "lip_pick_fraction": {"none": float(np.mean(picks == -1)) if len(picks) else None,
                                      "voice1": float(np.mean(picks == 0)) if len(picks) else None,
                                      "voice2": float(np.mean(picks == 1)) if len(picks) else None},
                "note": "voice1/voice2 are aligned to mix sample 0 (silence before first output)."}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        self.last_recording = str(folder)
        self.log("info", "test", f"saved {meta['seconds']} s recording to {folder}")

    def tick(self):
        """Called every 100 ms while a phone streams: report a model that cannot keep up."""
        with self.lock:
            separator = self.separator
        if separator is None:
            return
        live = separator.status()
        if live["skipped_hops"] > self.reported_skips:
            self.log("warn", "test", f"model too slow: {live['skipped_hops']} hops skipped "
                     f"(p95 {live['infer_p95_ms']} ms > hop {live['hop_s'] * 1000:.0f} ms)", every_s=2)
            self.reported_skips = live["skipped_hops"]

    def status(self):
        with self.lock:
            return {"enabled": self.enabled, "state": self.state, "error": self.error,
                    "model": self.model, "info": self.info,
                    "window_s": self.window_s, "hop_s": self.hop_s, "lookahead_s": LOOKAHEAD_S,
                    "monitor": self.monitor, "focus_db": self.focus_db, "duck_db": self.duck_db,
                    "lip_offset_ms": self.lip_offset_ms, "recording": self.record,
                    "last_recording": self.last_recording,
                    "live": self.separator.status() if self.separator else None}
