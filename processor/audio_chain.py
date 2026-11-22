"""Live audio chain applied to the phone's PCM stream before it is relayed.

Layer [2] of the pyramid: a human-voice band gate. Energy below ~80 Hz (HVAC,
handling rumble, traffic) and above ~8 kHz (hiss, clatter, electronics) is
treated as background and removed; speech fundamentals (~85-255 Hz) and the
formants/consonants that carry intelligibility (~300 Hz-8 kHz) pass.

Every layer is independently toggleable and always runs internally, so its
filter state stays warm; toggling crossfades over one 20 ms block (no click).
Later layers (direction, DeepFilterNet, lip gate) plug in after this one.
"""
from dataclasses import dataclass
import threading
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

LAYERS = ("voice_band", "auto_volume")
MIN_LOW_HZ, MAX_HIGH_HZ = 20, 20_000


@dataclass
class ChainSettings:
    """Shared by all sessions; changed from the dashboard."""
    voice_band: bool = True
    auto_volume: bool = True  # last stage: bring speech to a comfortable headphone level
    low_hz: float = 80.0
    high_hz: float = 8000.0

    def __post_init__(self):
        self.lock = threading.Lock()
        self.version = 0

    def update(self, message):
        """Apply a viewer control message; raises ValueError if invalid."""
        kind = message.get("type")
        with self.lock:
            if kind == "set_layer":
                layer, enabled = message.get("layer"), message.get("enabled")
                if layer not in LAYERS or not isinstance(enabled, bool):
                    raise ValueError(f"layer must be one of {LAYERS} with boolean enabled")
                setattr(self, layer, enabled)
            elif kind == "set_band":
                low, high = message.get("low_hz"), message.get("high_hz")
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high)):
                    raise ValueError("low_hz and high_hz must be numbers")
                if not MIN_LOW_HZ <= low < high <= MAX_HIGH_HZ or high / low < 2:
                    raise ValueError("band must satisfy 20 <= low < high <= 20000 and span an octave")
                self.low_hz, self.high_hz = float(low), float(high)
            else:
                raise ValueError("unsupported control message")
            self.version += 1

    def snapshot(self):
        with self.lock:
            return self.version, {"voice_band": self.voice_band, "auto_volume": self.auto_volume}, (self.low_hz, self.high_hz)


class VoiceBand:
    """Streaming Butterworth band-pass (4th order per edge, SOS with state)."""
    def __init__(self, rate, low_hz=80.0, high_hz=8000.0, order=4):
        self.rate = rate
        self.configure(low_hz, high_hz, order)

    def configure(self, low_hz, high_hz, order=4):
        nyquist = self.rate / 2
        self.requested = (low_hz, high_hz)
        self.low_hz = low_hz
        # At 16 kHz capture, 8 kHz is the Nyquist limit: keep only the high-pass.
        self.high_hz = high_hz if high_hz < 0.9 * nyquist else None
        if self.high_hz:
            self.sos = butter(order, [low_hz, self.high_hz], btype="bandpass", fs=self.rate, output="sos")
        else:
            self.sos = butter(order, low_hz, btype="highpass", fs=self.rate, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0

    def process(self, samples):
        out, self.zi = sosfilt(self.sos, samples, zi=self.zi)
        return out


class AutoVolume:
    """Speech-gated automatic gain with a peak limiter, applied per 20 ms block.

    The phone's UNPROCESSED microphone has no gain: speech arrived around -40 dBFS
    RMS, and filtering, separation and Focus ducking only lower it further. This
    raises speech toward `target_db` (at most `max_gain_db`), holds the gain during
    pauses so room noise is not pumped up, and never lets a block clip.
    """
    def __init__(self, rate, target_db=-20.0, max_gain_db=30.0, min_gain_db=-12.0,
                 up_db_per_s=6.0, down_db_per_s=30.0, ceiling=0.89):
        self.rate, self.target_db = rate, target_db
        self.max_gain_db, self.min_gain_db = max_gain_db, min_gain_db
        self.up, self.down, self.ceiling = up_db_per_s, down_db_per_s, ceiling
        self.gain_db = 0.0
        self.noise_db = -60.0   # slowly tracked background level
        self.speech_db = None   # smoothed speech level

    def process(self, x):
        seconds = len(x) / self.rate
        level = level_db(x)
        # Background follows quiet blocks quickly and loud ones slowly.
        self.noise_db += (level - self.noise_db) * (0.3 if level < self.noise_db else 0.01)
        speech = level > max(self.noise_db + 6.0, -65.0)
        if speech:
            self.speech_db = level if self.speech_db is None else self.speech_db + 0.1 * (level - self.speech_db)
            wanted = min(self.max_gain_db, max(self.min_gain_db, self.target_db - self.speech_db))
            step = (self.up if wanted > self.gain_db else self.down) * seconds
            self.gain_db += max(-step, min(step, wanted - self.gain_db))
        start = getattr(self, "_last_linear", 10 ** (self.gain_db / 20))
        end = 10 ** (self.gain_db / 20)
        y = x * np.linspace(start, end, len(x), endpoint=False)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > self.ceiling:  # limiter: pull this block (and the running gain) under the ceiling
            y *= self.ceiling / peak
            end *= self.ceiling / peak
            self.gain_db = 20 * np.log10(max(end, 1e-6))
        self._last_linear = end
        return y


def level_db(samples):
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    return round(20 * np.log10(max(rms, 1e-6)), 1)


class AudioChain:
    """Per-session processor: PCM16 bytes in, PCM16 bytes out (same length)."""
    def __init__(self, rate, settings=None):
        self.rate = rate
        self.settings = settings or ChainSettings()
        self.version = -1
        self.gains = {name: 1.0 for name in LAYERS}  # current wet mix per layer
        self.band = None
        self.volume = AutoVolume(rate)
        self.in_db = self.out_db = -120.0
        self.final_db = -120.0
        self.process_ms = 0.0
        self.error = None
        self._sync()

    def _sync(self):
        version, enabled, (low, high) = self.settings.snapshot()
        if version == self.version:
            return
        if self.band is None or (low, high) != self.band.requested:
            self.band = VoiceBand(self.rate, low, high)
        self.enabled = enabled
        if self.version < 0:  # first block: start at the requested state, no fade
            self.gains = {name: float(on) for name, on in enabled.items()}
        self.version = version

    def process(self, pcm):
        started = time.perf_counter()
        self._sync()
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
        self.in_db = level_db(x)
        y = self._mix("voice_band", x, self.band.process(x))
        self.out_db = level_db(y)
        out = (np.clip(y, -1.0, 32767 / 32768) * 32768.0).round().astype("<i2").tobytes()
        self.process_ms = (time.perf_counter() - started) * 1000
        return out

    def finish(self, pcm):
        """Last stage before relaying (after test mode): auto volume, if enabled."""
        self._sync()
        if not self.enabled.get("auto_volume"):
            self.final_db = self.out_db
            return pcm
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
        y = self.volume.process(x)
        self.final_db = level_db(y)
        return (np.clip(y, -1.0, 32767 / 32768) * 32768.0).round().astype("<i2").tobytes()

    def _mix(self, name, dry, wet):
        """Crossfade between bypass and processed over one block when toggled."""
        start, end = self.gains[name], float(self.enabled[name])
        self.gains[name] = end
        if start == end:
            return wet if end else dry
        ramp = np.linspace(start, end, len(dry), endpoint=False)
        return ramp * wet + (1 - ramp) * dry

    def status(self):
        _, enabled, (low, high) = self.settings.snapshot()
        effective_high = self.band.high_hz if self.band else high
        return {"type": "pipeline", "layers": enabled,
                "band": {"low_hz": low, "high_hz": high, "effective_high_hz": effective_high},
                "in_db": self.in_db, "out_db": self.out_db,
                "final_db": self.final_db, "volume_gain_db": round(self.volume.gain_db, 1),
                "process_ms": round(self.process_ms, 3), "error": self.error}
