"""Live separation test mode: dashboard settings, model loading, per-session separator."""
from __future__ import annotations

import collections
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from scipy.io import wavfile

from processor.separation import MODEL_RATE, MONITORS, LipPicker, StreamingSeparator
from processor.speaker_id import IdentityTracker

ROOT = Path(__file__).resolve().parents[1]
WINDOWS, HOPS = (1.0, 1.5, 2.0), (0.25, 0.5)
LOOKAHEAD_S, RECORD_SECONDS = 0.25, 300


def _number(message, key, low, high):
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"{key} must be a number in [{low}, {high}]")
    return float(value)


class TestMode:
    def __init__(self, loader=None, on_change=None, recordings_dir=ROOT / "recordings", log=None,
                 embedder_loader=None, face_loader="auto"):
        if face_loader == "auto":  # the real detector goes with the real separator (tests pass fakes)
            face_loader = _load_face_model if loader is None else None
        self.face_loader, self.face_model = face_loader, None
        self.picker_kind = "face"  # "face": pretrained speaking detector; "lips": mouth-openness rule
        if loader is None:
            from processor.separation_models import load as loader
        self.loader, self.on_change, self.recordings_dir = loader, on_change, Path(recordings_dir)
        self.log = log or (lambda *args, **kwargs: None)
        self.embedder_loader = embedder_loader
        self.identity, self.embed = False, None  # "Voice lock": speaker voiceprints (opt-in)
        self.reported_skips = 0
        self.last_counts = collections.Counter()
        self.lock = threading.RLock()
        self.enabled, self.model = False, "auto"
        self.window_s, self.hop_s = 2.0, 0.25  # 0.75 s delay; more context separates better
        self.monitor = "focus"  # the person on camera, with everyone else lowered
        self.focus_style = "mask"
        # Boost the person on camera relative to everyone else; do not remove the rest.
        self.focus_db, self.duck_db, self.lip_offset_ms = 6.0, -8.0, 0.0
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
                picker = message.get("picker", self.picker_kind)
                if picker not in ("face", "lips"):
                    raise ValueError("picker must be face (speaking detector) or lips")
                if picker != self.picker_kind:
                    self.picker_kind = picker
                    self.version += 1  # rebuild with the other picker
                identity = message.get("identity", self.identity)
                if not isinstance(identity, bool):
                    raise ValueError("identity must be boolean")
                window = message.get("window_s", self.window_s)
                hop = message.get("hop_s", self.hop_s)
                if window not in WINDOWS or hop not in HOPS:
                    raise ValueError(f"window_s must be in {WINDOWS} and hop_s in {HOPS}")
                if model != self.model:
                    self.separate, self.loaded_model = None, None
                if identity != self.identity and self.separate is not None and identity and self.embed is None:
                    self.separate, self.loaded_model = None, None  # reload to add the voiceprint model
                self.enabled, self.model, self.window_s, self.hop_s = enabled, model, float(window), float(hop)
                self.identity = identity
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
                if "style" in message:
                    if message["style"] not in ("mask", "mix"):
                        raise ValueError("style must be mask (natural) or mix")
                    self.focus_style = message["style"]
                self.picker.offset_ms = self.lip_offset_ms
                if self.separator:
                    self.separator.focus_gain = 10 ** (self.focus_db / 20)
                    self.separator.duck_gain = 10 ** (self.duck_db / 20)
                    self.separator.focus_style = self.focus_style
            elif kind == "set_recording":
                record = message.get("enabled")
                if not isinstance(record, bool):
                    raise ValueError("enabled must be boolean")
                if record and not self.record and self.separator is not None:
                    self.separator.start_recording(RECORD_SECONDS)  # keep stitching and voiceprints
                elif record and not self.record:
                    self.version += 1
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
                    embed = self._load_embedder(info)
                    face_model = self._load_face(info)
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
                        self.embed = embed
                        self.face_model = face_model
                        self.state = "ready" if self.enabled else "off"
                        self.version += 1
                    break
            if self.on_change:
                self.on_change()

        self.loading = threading.Thread(target=run, name="separator-loader", daemon=True)
        self.loading.start()

    def _load_face(self, info):
        """Light-ASD speaking detector, on the separator's device. None: use the lip rule."""
        if self.face_loader is None:
            return self.face_model
        if self.face_model is not None and getattr(self.face_model, "device", None) == info.get("device", "cpu"):
            return self.face_model
        real = self.face_loader is _load_face_model
        if real and info.get("device") != "cuda" and os.environ.get("LIPSYNC_FACE_CPU") != "1":
            # About 0.7 s per 2 s window on a laptop CPU: too slow for every 0.25 s hop.
            self.log("info", "test", "face picker needs the GPU (set LIPSYNC_FACE_CPU=1 to force); using the lip rule")
            return None
        try:
            model = self.face_loader(info.get("device", "cpu"))
            self.log("info", "test", "face picker ready (Light-ASD speaking detector, TalkSet weights)")
            return model
        except Exception as exc:
            self.log("warn", "test", f"face picker unavailable, using the lip rule: {exc}")
            return None

    def _new_picker(self):
        if self.picker_kind == "face" and self.face_model is not None:
            from processor.active_speaker import TalkingFacePicker
            return TalkingFacePicker(self.face_model, offset_ms=self.lip_offset_ms)
        return LipPicker(offset_ms=self.lip_offset_ms)

    def _load_embedder(self, info):
        """Voice lock model; GPU only by default (about 190 ms per voice on a laptop CPU)."""
        if not self.identity:
            return None
        if info.get("device") != "cuda" and os.environ.get("LIPSYNC_IDENTITY_CPU") != "1":
            self.log("info", "test", "voice lock off: needs the GPU (set LIPSYNC_IDENTITY_CPU=1 to force)")
            return None
        try:
            if self.embedder_loader is None:
                from processor.speaker_id import load_embedder
                embed = load_embedder(info.get("device", "cpu"))
            else:
                embed = self.embedder_loader(info.get("device", "cpu"))
            self.log("info", "test", "voice lock ready (ECAPA speaker voiceprints)")
            return embed
        except Exception as exc:  # separation keeps working without it
            self.log("warn", "test", f"voice lock unavailable: {exc}")
            return None

    # ---- session / media ----
    def attach(self, session_id, rate):
        with self.lock:
            self._close_separator()
            self.session = {"id": session_id, "rate": rate, "anchor_us": None}
            self.picker = self._new_picker()
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

    def add_mouth(self, capture_us, info, face=None):
        """Every processed video frame: mouth openness, and the target face crop (or None)."""
        picker = self.picker
        picker.add(capture_us, float(info["openness"]) if info.get("face_found") else None)
        if hasattr(picker, "add_face"):
            picker.add_face(capture_us, face if info.get("face_found") else None)

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
                    self._build(capture_us)
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

    def _build(self, anchor_us):
        """anchor_us: phone capture time of the block about to be fed (separator sample 0).

        The separator counts samples from its own start, so lip frames must be
        located relative to this moment, not the session start (that mistake made
        every pick "face not visible" when test mode started after connecting)."""
        self._close_separator()
        session = self.session
        wants_face = self.picker_kind == "face" and self.face_model is not None
        if wants_face != hasattr(self.picker, "add_face"):
            self.picker = self._new_picker()
        self.picker.choice, self.picker.reason = -1, "waiting for face"
        self.separator = StreamingSeparator(
            session["rate"], self.separate, window_s=self.window_s, hop_s=self.hop_s,
            lookahead_s=LOOKAHEAD_S, picker=self.picker, anchor_us=anchor_us,
            focus_db=self.focus_db, duck_db=self.duck_db,
            record_seconds=RECORD_SECONDS if self.record else 0,
            model_rate=int((self.info or {}).get("sample_rate", MODEL_RATE)),
            identity=IdentityTracker(self.embed) if (self.identity and self.embed) else None,
            focus_style=self.focus_style)
        self.built_version, self.state, self.error = self.version, "running", None
        self.reported_skips = 0
        self.last_counts = collections.Counter()
        delay_ms = self.separator.delay * 1000 // session["rate"]
        self.log("info", "test", f"separating: window {self.window_s} s, hop {self.hop_s} s, "
                 f"delay {delay_ms} ms, input {session['rate']} Hz, model {self.separator.mr} Hz")

    def _close_separator(self):
        if self.separator is not None:
            self.separator.close()
            self.separator = None

    def _save_recording(self):
        if self.separator is None:
            return
        rate = self.separator.mr
        data = self.separator.take_recording()
        if not data or not len(data["mix"]):
            return
        folder = self.recordings_dir / f"live-test-{datetime.now():%Y%m%d-%H%M%S}"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("mix", "voice1", "voice2"):
            if name in data and len(data[name]):
                wavfile.write(folder / f"{name}.wav", rate,
                              (np.clip(data[name], -1, 1) * 32767).astype(np.int16))
        picks = data.get("focus_pick", np.zeros(0))
        meta = {"sample_rate": rate, "seconds": round(len(data["mix"]) / rate, 2),
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
        """Every 100 ms while a phone streams: explain dropped hops honestly."""
        with self.lock:
            separator = self.separator
        if separator is None:
            return
        live = separator.status()
        if live["skipped_hops"] <= self.reported_skips:
            return
        new = live["skipped_hops"] - self.reported_skips
        hop_ms, p95 = live["hop_s"] * 1000, live["infer_p95_ms"]
        if p95 is not None and p95 >= hop_ms:
            why = (f"model too slow (p95 {p95:.0f} ms >= hop {hop_ms:.0f} ms); "
                   "try hop 0.5 s, a 1 s window, a lighter model, or the GPU")
        else:
            shown = "n/a" if p95 is None else f"{p95:.0f} ms"
            why = (f"audio reached the laptop too late; the model is fast enough "
                   f"(p95 {shown}, hop {hop_ms:.0f} ms). Check Wi-Fi and CPU load")
        self.log("warn", "test", f"dropped {new} late hop(s) ({live['skipped_hops']} total): {why}", every_s=2)
        self.reported_skips = live["skipped_hops"]

    def summary(self):
        """One line every few seconds for the debug log, or None when not separating."""
        with self.lock:
            separator, picker, info = self.separator, self.picker, self.info or {}
        if separator is None:
            return None
        live = separator.status()
        delta = picker.counts - self.last_counts
        self.last_counts = picker.counts.copy()
        total = sum(delta.values())
        picks = ", ".join(f"{reason} {100 * n / total:.0f}%" for reason, n in delta.most_common()) or "no windows"
        target = f"voice {live['pick'] + 1}" if live["pick"] >= 0 else "none"
        fmt = lambda v: "-" if v is None else f"{v:.0f}"
        return (f"{info.get('model')} on {info.get('gpu') or info.get('device')} | window {live['window_s']} s, "
                f"hop {live['hop_s']} s, delay {live['delay_ms']} ms | model {fmt(live['infer_ms'])}/"
                f"{fmt(live['infer_p95_ms'])} ms (median/p95) | hops {live['hops']}, dropped late {live['recent_drops']} "
                f"(10 s), playback gaps {live['underruns']} | voices {live['voice_db'][0]}/{live['voice_db'][1]} dB, "
                f"avg gap {live['voice_gap_db']} dB | {self._picker_text(picker)}: {picks} -> {target} (evidence "
                f"{live['lip_evidence']:+.2f}, {live['pick_switches']} switches) | monitor {self.monitor}"
                f" ({self.focus_style}, target share {live['target_share']})"
                f" | voice lock: {self._lock_text(live)}")

    @staticmethod
    def _picker_text(picker):
        times = sorted(getattr(picker, "ms", []))
        if not hasattr(picker, "add_face"):
            return "lip pick"
        return f"face pick ({times[len(times) // 2]:.0f} ms)" if times else "face pick"

    @staticmethod
    def _lock_text(live):
        identity = live.get("identity")
        if not identity:
            return "off"
        return (f"{identity['identity_swaps']} corrections, last {identity.get('decision')}, "
                f"voiceprints {identity['voiceprints']}, {identity.get('ms')} ms")

    def status(self):
        with self.lock:
            return {"enabled": self.enabled, "state": self.state, "error": self.error,
                    "model": self.model, "info": self.info,
                    "window_s": self.window_s, "hop_s": self.hop_s, "lookahead_s": LOOKAHEAD_S,
                    "monitor": self.monitor, "focus_db": self.focus_db, "duck_db": self.duck_db,
                    "focus_style": self.focus_style,
                    "lip_offset_ms": self.lip_offset_ms, "recording": self.record,
                    "identity": self.identity, "identity_ready": self.embed is not None,
                    "picker": self.picker_kind, "picker_ready": self.face_model is not None,
                    "last_recording": self.last_recording,
                    "live": self.separator.status() if self.separator else None}


def _load_face_model(device):
    from processor.active_speaker import ActiveSpeaker
    return ActiveSpeaker("talkset", device=device)
