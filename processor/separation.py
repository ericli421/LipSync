"""Streaming two-voice separation with lip-driven focus (live test mode).

Every hop (e.g. 0.25 s) the last `window` seconds, resampled to the model's rate
(8 or 16 kHz), are separated
into two candidates. Consecutive chunks overlap, so each chunk's candidates are
reordered (and sign-corrected) to match the previous chunk before stitching;
this keeps "Voice 1" the same person across chunks. Only the region ending
`lookahead` seconds before the newest sample is emitted, because the model is
least reliable at the window edge. Emitted regions are crossfaded with the
previous chunk's estimate of the same samples.

Output delay (input-rate samples) is fixed: lookahead + 2 hops by default,
which leaves one full hop for inference. If inference is slower than the hop,
hops whose output would already be late are dropped and reported, rather
than letting delay grow. Bursty arrival that is still in time is not dropped.

The focus mix is: boost * target + duck * (delayed original - target), where
target is the candidate whose loudness envelope best correlates with the
on-camera mouth openness (scripts/lip_matching.py logic, applied per chunk).
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from pathlib import Path
import sys

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lip_matching import MatchConfig, correlation, filtered_signal  # noqa: E402

MODEL_RATE = 8000
FPS = 25
MONITORS = ("original", "voice1", "voice2", "focus")


class Decimator:
    """Stateful anti-aliased integer down-sampler."""
    def __init__(self, factor):
        self.factor = factor
        self.h = firwin(16 * factor + 1, 0.9 / factor) if factor > 1 else np.array([1.0])
        self.zi = lfilter_zi(self.h, 1.0) * 0.0
        self.phase = 0

    def process(self, x):
        if self.factor == 1:
            return x.copy()
        y, self.zi = lfilter(self.h, 1.0, x, zi=self.zi)
        out = y[self.phase::self.factor]
        self.phase = (self.phase - len(x)) % self.factor
        return out


class Upsampler:
    """Stateful zero-stuffing interpolator matching Decimator."""
    def __init__(self, factor):
        self.factor = factor
        self.h = firwin(16 * factor + 1, 0.9 / factor) * factor if factor > 1 else np.array([1.0])
        self.zi = lfilter_zi(self.h, 1.0) * 0.0

    def process(self, x):
        if self.factor == 1:
            return x.copy()
        stuffed = np.zeros(len(x) * self.factor)
        stuffed[::self.factor] = x
        y, self.zi = lfilter(self.h, 1.0, stuffed, zi=self.zi)
        return y


def db(x):
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return round(20 * np.log10(max(rms, 1e-6)), 1)


class LipPicker:
    """Choose the candidate whose envelope follows the on-camera mouth.

    Mouth samples are (capture_time_us, openness or None). Audio sample index i
    maps to time anchor_us + i / rate. `offset_ms` shifts video relative to audio
    because phone audio/video timestamps are not verified to be synchronized.
    """
    def __init__(self, config=MatchConfig(), offset_ms=0.0):
        self.config = config
        self.offset_ms = offset_ms
        self.mouths = collections.deque(maxlen=FPS * 10)
        self.lock = threading.Lock()
        self.choice = -1
        self.pending, self.pending_count = -1, 0
        self.scores = [None, None]
        self.reason = "waiting for face"
        self.counts = collections.Counter()  # pick outcomes, for the periodic log summary

    def add(self, capture_us, openness):
        with self.lock:
            self.mouths.append((capture_us, openness))

    def trace(self, start_us, frames):
        """Mouth openness on a 25 fps grid; NaN where no face or no frame within 80 ms."""
        with self.lock:
            samples = [(t + self.offset_ms * 1000, v) for t, v in self.mouths]
        grid = start_us + np.arange(frames) * (1e6 / FPS)
        out = np.full(frames, np.nan)
        if not samples:
            return out
        times = np.array([t for t, _ in samples], dtype=float)
        values = np.array([np.nan if v is None else v for _, v in samples], dtype=float)
        nearest = np.clip(np.searchsorted(times, grid), 1, len(times) - 1) if len(times) > 1 else np.zeros(frames, int)
        if len(times) > 1:
            left = nearest - 1
            nearest = np.where(np.abs(times[left] - grid) < np.abs(times[nearest] - grid), left, nearest)
        close = np.abs(times[nearest] - grid) <= 80_000
        out[close] = values[nearest][close]
        return out

    def pick(self, candidates, window_start_us, rate=MODEL_RATE):
        """candidates: (samples, 2) at `rate`. Returns chosen index or -1."""
        choice = self._pick(candidates, window_start_us, rate)
        self.counts[self.reason] += 1
        return choice

    def _pick(self, candidates, window_start_us, rate):
        hop = rate // FPS
        frames = len(candidates) // hop
        cfg = self.config
        mouth = self.trace(window_start_us, frames)
        valid = np.isfinite(mouth)
        if valid.mean() < cfg.min_valid_fraction:
            self.reason = "face not visible"
            return self.choice
        if np.std(mouth[valid]) < cfg.min_mouth_std:
            self.reason = "lips not moving"
            return self.choice
        lips = filtered_signal(mouth, FPS)
        envelopes = np.sqrt(np.mean(candidates[: frames * hop].reshape(frames, hop, 2) ** 2, axis=1))
        max_lag = round(cfg.max_lag_seconds * FPS)
        scores = []
        for source in range(2):
            raw = envelopes[:, source]
            floor = max(float(np.percentile(raw, 10)), float(np.max(raw)) * .02, 1e-8)
            sound = filtered_signal(np.log1p(raw / floor), FPS)
            best = -1.0
            for lag in range(-max_lag, max_lag + 1):
                a = lips[max(0, lag): frames + min(0, lag)]
                b = sound[max(0, -lag): frames - max(0, lag)]
                if len(a) >= frames // 2:
                    value = correlation(a, b)
                    if np.isfinite(value):
                        best = max(best, value)
            scores.append(best)
        self.scores = [round(s, 2) for s in scores]
        winner = int(np.argmax(scores))
        if scores[winner] < cfg.min_correlation or scores[winner] - scores[1 - winner] < cfg.min_margin:
            self.reason = "ambiguous"
            return self.choice
        if winner != self.choice:  # require consecutive agreement before switching
            self.pending_count = self.pending_count + 1 if winner == self.pending else 1
            self.pending = winner
            if self.choice != -1 and self.pending_count < cfg.switch_confirmation_windows:
                self.reason = "confirming switch"
                return self.choice
        self.choice, self.reason = winner, "matched"
        return self.choice


class StreamingSeparator:
    """Feed input-rate float blocks; read delayed output blocks for a monitor."""
    def __init__(self, rate, separate, window_s=1.5, hop_s=0.25, lookahead_s=0.25,
                 crossfade_s=0.03, delay_s=None, picker=None, anchor_us=0,
                 focus_db=3.0, duck_db=-15.0, record_seconds=0, model_rate=MODEL_RATE):
        self.mr = model_rate
        if rate % self.mr:
            raise ValueError(f"test mode needs a capture rate divisible by {self.mr} Hz, not {rate}")
        self.rate, self.factor, self.separate = rate, rate // self.mr, separate
        self.W, self.H = round(window_s * self.mr), round(hop_s * self.mr)
        self.D = round(lookahead_s * self.mr)
        self.F = min(round(crossfade_s * self.mr), self.D)
        if self.H <= 0 or self.D + self.H > self.W:
            raise ValueError("need 0 < hop and lookahead + hop <= window")
        delay_s = lookahead_s + 2 * hop_s if delay_s is None else delay_s
        self.delay = round(delay_s * rate)
        self.picker = picker or LipPicker()
        self.anchor_us = anchor_us
        self.focus_gain, self.duck_gain = 10 ** (focus_db / 20), 10 ** (duck_db / 20)
        self.down = Decimator(self.factor)
        # Both FIRs are linear phase: voices lag the input by their group delays.
        self.group = 16 * self.factor if self.factor > 1 else 0
        self.ups = [Upsampler(self.factor) for _ in range(2)]
        self.inbox = queue.Queue(maxsize=500)  # 10 s of 20 ms blocks
        self.ring = np.zeros(0)
        self.total8 = 0
        self.next_end = self.W
        self.prev = None          # (abs start, aligned/scaled estimate)
        self.gains = np.ones(2)
        self.emitted8 = None      # abs 8 kHz index where the next emission starts
        self.lock = threading.Lock()
        self.voices = np.zeros((0, 2))  # input-rate output, starts at out_start
        self.picks = np.zeros(0, dtype=np.int8)
        self.out_start = None
        self.original = collections.deque()  # (abs start, block) input-rate history
        self.fed = 0              # input-rate samples received
        self.processed = 0        # input-rate samples fully handled by the worker
        self.read_pos = 0
        self.stats = {"hops": 0, "skipped_hops": 0, "swaps": 0, "underruns": 0,
                      "infer_ms": collections.deque(maxlen=40), "behind_ms": 0.0,
                      "late_drops": collections.deque(maxlen=2000),
                      "gap_db": collections.deque(maxlen=40)}
        self.voice_db = [-120.0, -120.0]
        self.error = None
        self.record_limit = record_seconds * self.mr
        self.recording = {"mix": [], "voice1": [], "voice2": [], "focus_pick": []} if record_seconds else None
        self.running = True
        self.worker = threading.Thread(target=self._run, name="separator", daemon=True)
        self.worker.start()

    # ---- producer side (event loop) ----
    def feed(self, block):
        with self.lock:
            self.original.append((self.fed, block))
            while self.original and self.original[0][0] + len(self.original[0][1]) < self.fed - self.delay - self.rate:
                self.original.popleft()
            self.fed += len(block)
        try:
            self.inbox.put_nowait(block)
        except queue.Full:
            self.error = "separation fell more than 10 s behind; stopped"
            self.running = False

    def read(self, n, monitor):
        """Return n output samples ending `delay` samples behind the newest input."""
        if monitor not in MONITORS:
            raise ValueError("unknown monitor")
        with self.lock:
            end = self.fed - self.delay
            start = end - n
            original = self._original(start - self.group, n)
            if monitor == "original":
                return original
            out = np.zeros((n, 2))
            picks = np.full(n, -1, dtype=np.int8)
            if self.out_start is not None:
                lo, hi = max(start, self.out_start), min(end, self.out_start + len(self.voices))
                if hi > lo:
                    out[lo - start: hi - start] = self.voices[lo - self.out_start: hi - self.out_start]
                    picks[lo - start: hi - start] = self.picks[lo - self.out_start: hi - self.out_start]
                if start >= self.out_start and hi < end:
                    self.stats["underruns"] += 1
                drop = max(0, start - self.out_start)  # consumed
                if drop:
                    self.voices, self.picks = self.voices[drop:], self.picks[drop:]
                    self.out_start += drop
        if monitor == "voice1":
            return out[:, 0]
        if monitor == "voice2":
            return out[:, 1]
        target = np.where(picks == 1, out[:, 1], np.where(picks == 0, out[:, 0], 0.0))
        unpicked = picks < 0  # no confident lip match yet: pass the original through
        focus = self.focus_gain * target + self.duck_gain * (original - target)
        focus[unpicked] = original[unpicked]
        return focus

    def _original(self, start, n):
        out = np.zeros(n)
        for at, block in self.original:
            lo, hi = max(start, at), min(start + n, at + len(block))
            if hi > lo:
                out[lo - start: hi - start] = block[lo - at: hi - at]
        return out

    # ---- worker side ----
    def _run(self):
        try:
            while self.running:
                blocks = []
                try:
                    blocks.append(self.inbox.get(timeout=0.1))
                    while True:  # take the whole backlog so lateness is visible below
                        blocks.append(self.inbox.get_nowait())
                except queue.Empty:
                    pass
                if not blocks:
                    continue
                x8 = self.down.process(np.concatenate(blocks))
                self.ring = np.concatenate([self.ring, x8])[-(self.W + 2 * self.mr):]
                self.total8 += len(x8)
                if self.recording is not None and sum(map(len, self.recording["mix"])) < self.record_limit:
                    self.recording["mix"].append(x8)
                while self.running and self.total8 >= self.next_end:
                    behind = self.total8 - self.next_end
                    self.stats["behind_ms"] = behind * 1000 / self.mr
                    # Drop a hop only when its output would already have been played.
                    # Audio that arrives in a burst but is still in time is processed
                    # back to back, so Wi-Fi jitter alone no longer creates gaps.
                    if self._too_late(self.next_end):
                        self.stats["skipped_hops"] += 1
                        self.stats["late_drops"].append(time.monotonic())
                    else:
                        self._hop(self.next_end)
                    self.next_end += self.H
                self.processed += sum(map(len, blocks))
        except Exception as exc:  # surface in status; relay keeps running
            self.error = f"separation failed: {type(exc).__name__}: {exc}"
            self.running = False

    def _too_late(self, end):
        """True if the region this hop emits ends before the current playout point."""
        emitted_end = (end - self.D) * self.factor + self.group
        return self.fed - self.delay > emitted_end

    def _hop(self, end):
        start = end - self.W
        window = self.ring[len(self.ring) - (self.total8 - start): len(self.ring) - (self.total8 - end)]
        began = time.perf_counter()
        est = np.asarray(self.separate(window.astype(np.float32)), dtype=np.float64)
        self.stats["infer_ms"].append((time.perf_counter() - began) * 1000)
        if est.shape != (self.W, 2) or not np.isfinite(est).all():
            raise ValueError(f"model returned {est.shape} or non-finite output")
        est = self._scale(self._align(est, start), window)
        choice = self.picker.pick(est, self.anchor_us + start * 1_000_000 // self.mr, self.mr)
        e0, e1 = end - self.D - self.H, end - self.D
        region = est[e0 - start: e1 - start].copy()
        if self.prev is not None and self.F:
            pstart, pest = self.prev
            if pstart + self.W >= e0 + self.F:
                ramp = np.linspace(0, 1, self.F, endpoint=False)[:, None]
                old = pest[e0 - pstart: e0 - pstart + self.F]
                region[: self.F] = ramp * region[: self.F] + (1 - ramp) * old
        self.prev = (start, est)
        self.stats["hops"] += 1
        self.voice_db = [db(region[:, 0]), db(region[:, 1])]
        self.stats["gap_db"].append(abs(self.voice_db[0] - self.voice_db[1]))
        gap = 0 if self.emitted8 is None else e0 - self.emitted8
        if gap > 0:  # skipped hops: emit silence so the timeline stays aligned
            region = np.vstack([np.zeros((gap, 2)), region])
        up = np.column_stack([u.process(region[:, i]) for i, u in enumerate(self.ups)])
        with self.lock:
            if self.out_start is None:
                self.out_start = e0 * self.factor
            self.voices = np.vstack([self.voices, up])
            self.picks = np.concatenate([self.picks, np.full(len(up), choice, dtype=np.int8)])
        self.emitted8 = e1
        if self.recording is not None and sum(map(len, self.recording["voice1"])) < self.record_limit:
            if not self.recording["voice1"]:
                lead = np.zeros(max(0, e0 - gap))
                self.recording["voice1"].append(lead)
                self.recording["voice2"].append(lead)
                self.recording["focus_pick"].append(np.full(len(lead), -1))
            self.recording["voice1"].append(region[:, 0])
            self.recording["voice2"].append(region[:, 1])
            self.recording["focus_pick"].append(np.full(len(region), choice))

    def _align(self, est, start):
        """Reorder/sign-correct candidates to match the previous chunk on the overlap."""
        if self.prev is None:
            return est
        pstart, pest = self.prev
        lo, hi = start, pstart + self.W
        if hi - lo < self.mr // 20:
            return est
        a, b = est[lo - start: hi - start], pest[lo - pstart: hi - pstart]
        c = np.array([[correlation(a[:, i], b[:, j]) for j in range(2)] for i in range(2)])
        c = np.nan_to_num(c)
        if abs(c[0, 1]) + abs(c[1, 0]) > abs(c[0, 0]) + abs(c[1, 1]):
            est, c = est[:, ::-1], c[::-1]
            self.stats["swaps"] += 1
        signs = np.where(np.diag(c) < 0, -1.0, 1.0)
        return est * signs

    def _scale(self, est, mixture):
        """Project each candidate onto the mixture so levels match the input."""
        energy = np.sum(est ** 2, axis=0)
        target = np.clip(est.T @ mixture / np.maximum(energy, 1e-9), 0.05, 4.0)
        self.gains = 0.7 * self.gains + 0.3 * target
        return est * self.gains

    # ---- reporting ----
    def status(self):
        infer = list(self.stats["infer_ms"])
        hop_ms = self.H * 1000 / self.mr
        p95 = float(np.percentile(infer, 95)) if infer else None
        return {
            "window_s": self.W / self.mr, "hop_s": self.H / self.mr, "lookahead_s": self.D / self.mr,
            "delay_ms": round(self.delay * 1000 / self.rate),
            "infer_ms": round(float(np.median(infer)), 1) if infer else None,
            "infer_p95_ms": round(p95, 1) if p95 is not None else None,
            "rtf": round(float(np.median(infer)) / (self.W * 1000 / self.mr), 3) if infer else None,
            "keeping_up": None if p95 is None else bool(p95 < hop_ms and not self.recent_drops()),
            "recent_drops": self.recent_drops(),
            "voice_gap_db": round(float(np.mean(self.stats["gap_db"])), 1) if self.stats["gap_db"] else None,
            "model_rate": self.mr,
            "hops": self.stats["hops"], "skipped_hops": self.stats["skipped_hops"],
            "swaps_fixed": self.stats["swaps"], "underruns": self.stats["underruns"],
            "behind_ms": round(self.stats["behind_ms"]),
            "voice_db": self.voice_db,
            "pick": self.picker.choice, "pick_reason": self.picker.reason, "lip_scores": self.picker.scores,
            "recorded_s": round(sum(map(len, self.recording["mix"])) / self.mr, 1) if self.recording else 0,
            "error": self.error,
        }

    def recent_drops(self, seconds=10.0):
        """Hops dropped as late in the last `seconds` (not since the start)."""
        cutoff = time.monotonic() - seconds
        return sum(1 for t in list(self.stats["late_drops"]) if t >= cutoff)

    def take_recording(self):
        """Return {name: 8 kHz float array} aligned to the same start, then stop recording."""
        if self.recording is None:
            return None
        rec, self.recording = self.recording, None
        out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rec.items()}
        length = min(len(out["mix"]), len(out["voice1"])) if len(out["voice1"]) else len(out["mix"])
        return {k: v[:length] for k, v in out.items()}

    def close(self):
        """Non-blocking: an in-flight inference finishes in its daemon thread."""
        self.running = False
