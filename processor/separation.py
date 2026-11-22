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
import dataclasses
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
# Lock in: once a face is matched to a voice, the other voice must win 4 windows in
# a row (about 1 s at a 0.25 s hop) before Focus switches person.
# The live face tracker misses frames (face found in 15-85% of frames in the first
# phone tests), so a window is judged when half its frames have a face; gaps are
# interpolated by filtered_signal.
LOCK_CONFIG = dataclasses.replace(MatchConfig(), switch_confirmation_windows=4, min_valid_fraction=0.5)


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

    Evidence accumulates over about `EVIDENCE_TAU_S` seconds instead of deciding each
    2 s window on its own (single windows flipped the pick every few seconds in the
    first phone tests). Each window adds: the correlation difference between the two
    voices at one common lip/audio lag (a per-voice lag inflated the wrong voice), or,
    while the tracked lips are still, evidence against whichever voice is talking.
    The pick changes only when the evidence clearly crosses to the other voice.

    Mouth samples are (capture_time_us, openness or None). Audio sample index i
    maps to time anchor_us + i / rate. `offset_ms` shifts video relative to audio
    because phone audio/video timestamps are not verified to be synchronized.
    """
    def __init__(self, config=LOCK_CONFIG, offset_ms=0.0):
        self.config = config
        self.offset_ms = offset_ms
        self.mouths = collections.deque(maxlen=FPS * 10)
        self.lock = threading.Lock()
        self.choice = -1
        self.pending, self.pending_count = -1, 0
        self.scores = [None, None]
        self.reason = "waiting for face"
        self.counts = collections.Counter()  # pick outcomes, for the periodic log summary
        self.evidence = 0.0                   # > 0 favours voice 2, < 0 favours voice 1
        self.switches = 0
        self.last_start_us = None

    EVIDENCE_TAU_S = 3.0   # memory of the evidence
    ENTER, SWITCH = 0.08, 0.15   # first pick, and changing an existing pick
    STILL_WEIGHT = 0.5
    MAX_LAG_S = 0.2

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
        close = np.abs(times[nearest] - grid) <= 120_000  # phone video arrives at ~10 fps
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
        dt = 0.25 if self.last_start_us is None else min(1.0, max(0.05, (window_start_us - self.last_start_us) / 1e6))
        self.last_start_us = window_start_us
        mouth = self.trace(window_start_us, frames)
        valid = np.isfinite(mouth)
        if valid.mean() < cfg.min_valid_fraction:
            self.reason = "face not visible"
            return self.choice  # no information: keep the evidence and the pick
        raw = np.sqrt(np.mean(candidates[: frames * hop].reshape(frames, hop, 2) ** 2, axis=1))
        threshold = max(float(raw.max()) * 0.1, 1e-6)
        activity = (raw > threshold).mean(axis=0)  # share of the window each voice is talking
        if np.std(mouth[valid]) < cfg.min_mouth_std:
            difference = activity[1] - activity[0]
            if abs(difference) < 0.3:
                self.reason = "lips not moving"
                return self.choice
            # The tracked person is silent while one voice talks: that voice is someone else.
            delta, self.reason = -self.STILL_WEIGHT * difference, "lips still, other voice talking"
            self.scores = [None, None]
        else:
            lips = filtered_signal(mouth, FPS)
            sounds = []
            for source in range(2):
                floor = max(float(np.percentile(raw[:, source], 10)), float(np.max(raw[:, source])) * .02, 1e-8)
                sounds.append(filtered_signal(np.log1p(raw[:, source] / floor), FPS))
            best = None
            max_lag = round(self.MAX_LAG_S * FPS)
            for lag in range(-max_lag, max_lag + 1):  # one lag shared by both voices
                a = lips[max(0, lag): frames + min(0, lag)]
                if len(a) < frames // 2:
                    continue
                scores = []
                for sound in sounds:
                    value = correlation(a, sound[max(0, -lag): frames - max(0, lag)])
                    scores.append(value if np.isfinite(value) else -1.0)
                if best is None or max(scores) > max(best):
                    best = scores
            if best is None:
                self.reason = "ambiguous"
                return self.choice
            self.scores = [round(v, 2) for v in best]
            confidence = min(1.0, max(0.0, (max(best) - 0.1) / 0.3))  # weak matches count little
            delta = (best[1] - best[0]) * confidence
            self.reason = "matched" if abs(delta) > 0.05 else "ambiguous"
        alpha = 1 - np.exp(-dt / self.EVIDENCE_TAU_S)
        self.evidence += alpha * (delta - self.evidence)
        previous = self.choice
        if self.choice == -1 and abs(self.evidence) > self.ENTER:
            self.choice = int(self.evidence > 0)
        elif self.choice == 0 and self.evidence > self.SWITCH:
            self.choice = 1
        elif self.choice == 1 and self.evidence < -self.SWITCH:
            self.choice = 0
        if previous != -1 and self.choice != previous:
            self.switches += 1
            self.reason = "switched"
        elif self.choice != -1 and self.reason == "matched" and (delta > 0) != (self.choice == 1):
            self.reason = "holding (window disagrees)"
        return self.choice

SMOOTH = np.hanning(9)[1:-1] / np.hanning(9)[1:-1].sum()  # ~7 bins (~330 Hz at 48 kHz, n=1024)
SOFTEN = 0.85  # < 1 lifts weak target bins: shared bins are barely lowered


class MaskFocus:
    """Organic Focus: lower the other voices inside the original, full-band audio.

    The separated voices are only used to estimate, per short-time frequency bin,
    how much of the original belongs to the person on camera (a soft mask). That bin
    is then turned toward `boost` (target) or `duck` (everyone else) and smoothed
    over time, so the result keeps the room, the 48 kHz bandwidth and the natural
    timbre instead of the separator's artifacts. Above the model's band (8 kHz for a
    16 kHz model) bins follow the target's overall share in 300 Hz-4 kHz. Without a
    confident target the original passes through unchanged.
    """
    def __init__(self, rate, model_rate, boost=2.0, duck=0.4, attack_s=0.05, release_s=0.35):
        self.rate = rate
        self.n = 1024 if rate >= 32000 else 512
        self.h = self.n // 4
        self.win = np.sqrt(np.hanning(self.n + 1)[:-1])
        self.norm = float(np.sum(self.win ** 2)) / self.h
        freqs = np.fft.rfftfreq(self.n, 1 / rate)
        top = 0.45 * model_rate
        self.ref = (freqs >= 300) & (freqs <= min(4000.0, top))
        self.high = freqs > top
        self.boost, self.duck = boost, duck
        self.up = 1 - np.exp(-self.h / (attack_s * rate))
        self.down = 1 - np.exp(-self.h / (release_s * rate))
        self.gain = np.ones(len(freqs))
        self.frame_o, self.frame_t = np.zeros(self.n), np.zeros(self.n)
        self.ola = np.zeros(self.n)
        self.rest_o, self.rest_t, self.rest_a = np.zeros(0), np.zeros(0), np.zeros(0, bool)
        self.share = 0.0  # last target share, for status
        self.latency = self.n - self.h

    def process(self, original, target, active):
        """Feed aligned input-rate samples; returns the available output (latency samples late)."""
        o = np.concatenate([self.rest_o, original])
        t = np.concatenate([self.rest_t, target if active else np.zeros(len(original))])
        a = np.concatenate([self.rest_a, np.full(len(original), bool(active))])
        out, h = [], self.h
        k = 0
        while k + h <= len(o):
            self.frame_o = np.concatenate([self.frame_o[h:], o[k:k + h]])
            self.frame_t = np.concatenate([self.frame_t[h:], t[k:k + h]])
            X = np.fft.rfft(self.frame_o * self.win)
            if a[k:k + h].mean() > 0.5:
                T = np.fft.rfft(self.frame_t * self.win)
                pt, pn = np.abs(T) ** 2, np.abs(X - T) ** 2
                mask = pt / (pt + pn + 1e-12)
                # Organic: smooth over ~300 Hz and soften (sqrt) so bins shared by the
                # target and the room are barely lowered: no holes, no "underwater" sound.
                mask = np.convolve(mask, SMOOTH, mode="same") ** SOFTEN
                weight = np.abs(X[self.ref]) ** 2
                self.share = float(np.sum(mask[self.ref] * weight) / max(np.sum(weight), 1e-12))
                mask[self.high] = self.share
                wanted = self.duck + (self.boost - self.duck) * mask
            else:
                wanted = np.ones_like(self.gain)
            step = np.where(wanted > self.gain, self.up, self.down)
            self.gain += step * (wanted - self.gain)
            self.ola += np.fft.irfft(X * self.gain, self.n) * self.win / self.norm
            out.append(self.ola[:h].copy())
            self.ola = np.concatenate([self.ola[h:], np.zeros(h)])
            k += h
        self.rest_o, self.rest_t, self.rest_a = o[k:], t[k:], a[k:]
        return np.concatenate(out) if out else np.zeros(0)


class StreamingSeparator:
    """Feed input-rate float blocks; read delayed output blocks for a monitor."""
    def __init__(self, rate, separate, window_s=1.5, hop_s=0.25, lookahead_s=0.25,
                 crossfade_s=0.03, delay_s=None, picker=None, anchor_us=0,
                 focus_db=6.0, duck_db=-8.0, record_seconds=0, model_rate=MODEL_RATE,
                 identity=None, focus_style="mask"):
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
        self.anchor_us = anchor_us   # phone capture time of this separator's first sample
        self.identity = identity     # optional IdentityTracker (speaker voiceprints)
        self.stitch_confidence = 0.0
        self.focus_gain, self.duck_gain = 10 ** (focus_db / 20), 10 ** (duck_db / 20)
        self.focus_style = focus_style            # "mask" (organic) or "mix" (separated voice)
        self.mask = MaskFocus(rate, model_rate, self.focus_gain, self.duck_gain)
        self.focus_out, self.focus_start = np.zeros(0), None  # organic Focus, input rate
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
                      "gap_db": collections.deque(maxlen=40),
                      "identity_ms": collections.deque(maxlen=40)}
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
                # Retire only samples actually produced. Advancing past an empty
                # buffer moves the worker's append position into the future;
                # padding skipped hops then shifts voices and makes Focus read
                # original audio that has not arrived (zeros).
                drop = min(len(self.voices), max(0, start - self.out_start))
                if drop:
                    self.voices, self.picks = self.voices[drop:], self.picks[drop:]
                    self.out_start += drop
        if monitor == "focus" and self.focus_style == "mask":
            return self._read_mask(start, end, original)
        if monitor == "voice1":
            return out[:, 0]
        if monitor == "voice2":
            return out[:, 1]
        target = np.where(picks == 1, out[:, 1], np.where(picks == 0, out[:, 0], 0.0))
        unpicked = picks < 0  # no confident lip match yet: pass the original through
        focus = self.focus_gain * target + self.duck_gain * (original - target)
        focus[unpicked] = original[unpicked]
        return focus

    def _read_mask(self, start, end, original):
        """Organic Focus output for [start, end); missing samples fall back to the original."""
        n = end - start
        with self.lock:
            out = original.copy()
            if self.focus_start is not None:
                lo, hi = max(start, self.focus_start), min(end, self.focus_start + len(self.focus_out))
                if hi > lo:
                    out[lo - start: hi - start] = self.focus_out[lo - self.focus_start: hi - self.focus_start]
                drop = min(len(self.focus_out), max(0, start - self.focus_start))
                if drop:
                    self.focus_out = self.focus_out[drop:]
                    self.focus_start += drop
        return out[:n]

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
        est = self._align(est, start)
        if self.identity is not None:  # keep each output on the same person after pauses
            began = time.perf_counter()
            est, swapped = self.identity.assign(est, self.mr, self.stitch_confidence)
            self.stats["identity_ms"].append((time.perf_counter() - began) * 1000)
            if swapped:
                self.gains = self.gains[::-1].copy()
        est = self._scale(est, window)
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
            # Organic Focus for the same samples, computed on the aligned original.
            begin = self.out_start + len(self.voices)
            original = self._original(begin - self.group, len(up))
        self.mask.boost, self.mask.duck = self.focus_gain, self.duck_gain
        lead = gap * self.factor
        parts = [self.mask.process(original[:lead], np.zeros(lead), False)] if lead else []
        target = up[lead:, choice] if choice >= 0 else np.zeros(len(up) - lead)
        parts.append(self.mask.process(original[lead:], target, choice >= 0))
        focused = np.concatenate(parts)
        with self.lock:
            if self.focus_start is None:
                self.focus_start = begin - self.mask.latency
            self.focus_out = np.concatenate([self.focus_out, focused])
            self.voices = np.vstack([self.voices, up])
            self.picks = np.concatenate([self.picks, np.full(len(up), choice, dtype=np.int8)])
        self.emitted8 = e1
        if self.recording is not None and sum(map(len, self.recording["voice1"])) < self.record_limit:
            # Voice tracks are written at a running model-rate position starting at the
            # mix track's start. Emitted audio may trail the recording start by several
            # regions (skip them) or follow a gap (pad with silence).
            region_start = e0 - gap
            position = getattr(self, "record_pos", None)
            if position is None:
                position = getattr(self, "record_offset", 0)
            if region_start > position:
                pad = np.zeros(region_start - position)
                for name in ("voice1", "voice2"):
                    self.recording[name].append(pad)
                self.recording["focus_pick"].append(np.full(len(pad), -1))
                position = region_start
            skip = position - region_start
            if skip < len(region):
                self.recording["voice1"].append(region[skip:, 0])
                self.recording["voice2"].append(region[skip:, 1])
                self.recording["focus_pick"].append(np.full(len(region) - skip, choice))
                position = region_start + len(region)
            self.record_pos = position

    def _align(self, est, start):
        """Reorder/sign-correct candidates to match the previous chunk on the overlap."""
        self.stitch_confidence = 0.0
        if self.prev is None:
            return est
        pstart, pest = self.prev
        lo, hi = start, pstart + self.W
        if hi - lo < self.mr // 20:
            return est
        a, b = est[lo - start: hi - start], pest[lo - pstart: hi - pstart]
        c = np.array([[correlation(a[:, i], b[:, j]) for j in range(2)] for i in range(2)])
        c = np.nan_to_num(c)
        keep, swap = abs(c[0, 0]) + abs(c[1, 1]), abs(c[0, 1]) + abs(c[1, 0])
        self.stitch_confidence = abs(keep - swap)  # ~2 when both voices overlap clearly, ~0 after a pause
        if swap > keep:
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
            "focus_style": self.focus_style,
            "target_share": round(self.mask.share, 2),
            "lip_evidence": round(self.picker.evidence, 2), "pick_switches": self.picker.switches,
            "stitch_confidence": round(self.stitch_confidence, 2),
            "identity": None if self.identity is None else {
                **self.identity.status(),
                "ms": round(float(np.median(self.stats["identity_ms"])), 1) if self.stats["identity_ms"] else None},
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

    def start_recording(self, seconds):
        """Begin recording from now on, without restarting separation."""
        with self.lock:
            self.record_limit = seconds * self.mr
            self.record_offset = self.total8
            self.record_pos = None
            self.recording = {"mix": [], "voice1": [], "voice2": [], "focus_pick": []}

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
