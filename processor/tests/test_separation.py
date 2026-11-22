"""Streaming separation with a fake model: stitching, lip pick, focus, timing."""
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from processor.separation import Decimator, LipPicker, StreamingSeparator, Upsampler
from processor.separation_models import choose_device
from processor.test_mode import TestMode

RATE = 48000


def sources(seconds, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    env1 = 0.5 + 0.5 * np.sin(2 * np.pi * 1.3 * t)
    env2 = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t + 1)
    return 0.3 * env1 * np.sin(2 * np.pi * 300 * t), 0.3 * env2 * np.sin(2 * np.pi * 700 * t), env2


def fake_model(swap=True, delay=0.0, rate=8000, stall_once=0.0):
    """Split below/above 500 Hz; swap output order randomly like a real separator."""
    rng = np.random.default_rng(1)
    stall = [stall_once]

    def separate(window):
        if delay:
            time.sleep(delay)
        if stall[0]:
            time.sleep(stall[0])  # one GPU/driver hiccup
            stall[0] = 0.0
        spectrum = np.fft.rfft(window)
        freqs = np.fft.rfftfreq(len(window), 1 / rate)
        low = np.fft.irfft(np.where(freqs < 500, spectrum, 0), len(window))
        high = np.fft.irfft(np.where(freqs >= 500, spectrum, 0), len(window))
        pair = [low, high]
        if swap and rng.random() < 0.5:
            pair.reverse()
        return np.column_stack(pair)
    return separate


def run(separator, mix, monitor):
    outputs = []
    for start in range(0, len(mix), 960):
        separator.feed(mix[start:start + 960])
        while separator.processed < separator.fed and not separator.error:
            time.sleep(0.001)
        outputs.append(separator.read(960, monitor))
    return np.concatenate(outputs)


def tone_db(x, freq, rate=RATE):
    t = np.arange(len(x)) / rate
    return 20 * np.log10(abs(np.dot(x, np.exp(-2j * np.pi * freq * t))) / len(x) + 1e-12)


class ResamplerTests(unittest.TestCase):
    def test_round_trip_is_transparent_in_voice_band(self):
        x = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(RATE) / RATE)
        down, up = Decimator(6), Upsampler(6)
        y = np.concatenate([up.process(down.process(b)) for b in np.split(x, 50)])
        delayed = np.concatenate([np.zeros(96), x])[:len(x)]
        error = y[RATE // 4:] - delayed[RATE // 4:]
        self.assertLess(20 * np.log10(np.std(error) / np.std(x)), -30)


class StreamingTests(unittest.TestCase):
    def test_stitching_keeps_each_voice_in_one_stream(self):
        s1, s2, _ = sources(8)
        separator = StreamingSeparator(RATE, fake_model(), window_s=1.5, hop_s=0.25)
        try:
            voice1 = run(separator, s1 + s2, "voice1")
            status = separator.status()
        finally:
            separator.close()
        tail = voice1[3 * RATE:]
        self.assertGreater(status["swaps_fixed"], 3)  # the fake model really swapped
        # One source dominates voice1 for the whole stream (no swaps leak through).
        self.assertGreater(abs(tone_db(tail, 300) - tone_db(tail, 700)), 25)
        self.assertEqual(status["skipped_hops"], 0)
        self.assertEqual(status["delay_ms"], 750)
        self.assertTrue(status["keeping_up"])

    def test_lips_pick_the_talker_and_focus_boosts_them(self):
        s1, s2, env2 = sources(10)
        picker = LipPicker()
        for frame in range(250):  # 25 fps mouth trace that follows talker 2
            picker.add(frame * 40_000, float(env2[frame * RATE // 25]))
        separator = StreamingSeparator(RATE, fake_model(), window_s=1.5, hop_s=0.25,
                                       picker=picker, focus_db=3, duck_db=-15, focus_style="mix")
        try:
            focus = run(separator, s1 + s2, "focus")
        finally:
            separator.close()
        self.assertEqual(picker.reason, "matched")
        tail, mix = focus[4 * RATE:], (s1 + s2)[4 * RATE:]
        before = tone_db(mix, 700) - tone_db(mix, 300)
        after = tone_db(tail, 700) - tone_db(tail, 300)
        self.assertGreater(after - before, 15)  # +3 dB target, -15 dB rest: ~18 dB contrast

    def test_organic_focus_lowers_the_other_voice_in_the_original(self):
        s1, s2, env2 = sources(10)
        t = np.arange(len(s2)) / RATE
        air = 0.02 * np.sin(2 * np.pi * 12000 * t) * env2  # target's sound above the model's band
        picker = LipPicker()
        for frame in range(250):
            picker.add(frame * 40_000, float(env2[frame * RATE // 25]))
        separator = StreamingSeparator(RATE, fake_model(), window_s=1.5, hop_s=0.25, picker=picker)
        try:
            mix = s1 + s2 + air
            focus = run(separator, mix, "focus")
        finally:
            separator.close()
        self.assertEqual(separator.status()["focus_style"], "mask")
        delay = separator.delay
        tail, before = focus[4 * RATE + delay:], mix[4 * RATE:len(mix) - delay]
        gain_target = tone_db(tail, 700) - tone_db(before, 700)
        gain_other = tone_db(tail, 300) - tone_db(before, 300)
        self.assertGreater(gain_target - gain_other, 10)             # the other voice is lowered ...
        self.assertGreater(gain_other, -12)                           # ... not erased
        self.assertGreater(tone_db(tail, 12000) - tone_db(before, 12000), 0)  # full band kept for the target

    def test_organic_focus_passes_the_original_without_a_face(self):
        s1, s2, _ = sources(5)
        separator = StreamingSeparator(RATE, fake_model(), window_s=1.5, hop_s=0.25)
        try:
            mix = s1 + s2
            focus = run(separator, mix, "focus")
        finally:
            separator.close()
        delay = separator.delay + separator.group
        out, ref = focus[3 * RATE:], mix[3 * RATE - delay:len(mix) - delay]
        self.assertLess(20 * np.log10(np.std(out - ref) / np.std(ref)), -25)

    def test_slow_model_skips_instead_of_growing_delay(self):
        s1, s2, _ = sources(3)
        separator = StreamingSeparator(RATE, fake_model(delay=0.4), window_s=1.0, hop_s=0.25)
        try:
            mix = s1 + s2
            for start in range(0, len(mix), 960):
                separator.feed(mix[start:start + 960])
                separator.read(960, "voice1")
                time.sleep(0.02)
            time.sleep(0.5)
            status = separator.status()
        finally:
            separator.close()
        self.assertGreater(status["skipped_hops"], 0)
        self.assertFalse(status["keeping_up"])
        self.assertIsNone(status["error"])

    def test_one_hiccup_is_caught_up_not_dropped(self):
        """A 0.6 s stall leaves a 2-hop backlog that is still in time: process it."""
        s1, s2, _ = sources(4)
        separator = StreamingSeparator(RATE, fake_model(stall_once=0.6), window_s=1.0, hop_s=0.25)
        try:
            mix = s1 + s2
            began = time.perf_counter()
            for k, start in enumerate(range(0, len(mix), 960)):
                time.sleep(max(0, began + k * 0.02 - time.perf_counter()))
                separator.feed(mix[start:start + 960])
                separator.read(960, "voice1")
            time.sleep(0.3)
            status = separator.status()
        finally:
            separator.close()
        self.assertEqual(status["skipped_hops"], 0)
        self.assertGreater(status["hops"], 8)

    def test_16k_model_rate(self):
        s1, s2, _ = sources(6)
        separator = StreamingSeparator(RATE, fake_model(rate=16000), window_s=1.5, hop_s=0.25, model_rate=16000)
        try:
            voice1 = run(separator, s1 + s2, "voice1")
            status = separator.status()
        finally:
            separator.close()
        tail = voice1[3 * RATE:]
        self.assertEqual(status["model_rate"], 16000)
        self.assertGreater(abs(tone_db(tail, 300) - tone_db(tail, 700)), 25)

    def test_recording_started_mid_stream_while_the_worker_lags(self):
        """Pressing Record crashed separation in the 00:05 phone session."""
        s1, s2, _ = sources(8)
        mix = s1 + s2
        separator = StreamingSeparator(RATE, fake_model(swap=False, rate=16000), window_s=2.0, hop_s=0.25,
                                       model_rate=16000)
        try:
            for k, start in enumerate(range(0, len(mix), 960)):
                if k == 200:
                    separator.start_recording(60)
                separator.feed(mix[start:start + 960])
                if k % 3 == 0:  # the worker runs a little behind, as on the live server
                    while separator.processed < separator.fed and not separator.error:
                        time.sleep(0.001)
            time.sleep(0.5)
            self.assertIsNone(separator.error)
            record = separator.take_recording()
        finally:
            separator.close()
        n = min(len(record["mix"]), len(record["voice1"]))
        rebuilt = record["voice1"][:n] + record["voice2"][:n]
        tail = slice(n // 2, n)
        self.assertLess(20 * np.log10(np.std(record["mix"][tail] - rebuilt[tail]) / np.std(record["mix"][tail])), -30)

    def test_rejects_unsupported_rate_and_geometry(self):
        with self.assertRaises(ValueError):
            StreamingSeparator(44100, fake_model())
        with self.assertRaises(ValueError):
            StreamingSeparator(RATE, fake_model(), window_s=0.4, hop_s=0.25, lookahead_s=0.25)


class TestModeTests(unittest.TestCase):
    def wait_state(self, mode, state):
        for _ in range(200):
            if mode.status()["state"] == state:
                return
            time.sleep(0.01)
        self.fail(f"state stayed {mode.status()['state']}")

    def test_load_feed_output_and_record(self):
        with tempfile.TemporaryDirectory() as folder:
            mode = TestMode(loader=lambda name: (fake_model(), {"model": name, "device": "cpu"}),
                            recordings_dir=folder)
            mode.update({"type": "set_test_mode", "enabled": True, "window_s": 1.0, "hop_s": 0.25})
            self.wait_state(mode, "ready")
            mode.attach("s1", RATE)
            mode.update({"type": "set_recording", "enabled": True})
            mode.update({"type": "set_monitor", "source": "voice2"})
            s1, s2, _ = sources(3)
            pcm = ((s1 + s2) * 32767).astype("<i2")
            for seq, start in enumerate(range(0, len(pcm), 960)):
                mode.feed(pcm[start:start + 960].tobytes(), start, 1_000_000 + seq * 20_000)
                self.assertEqual(len(mode.output(960)), 1920)
            self.assertEqual(mode.status()["state"], "running")
            mode.update({"type": "set_monitor", "source": "original"})
            self.assertIsNone(mode.output(960))
            time.sleep(0.3)
            mode.update({"type": "set_recording", "enabled": False})
            mode.detach("s1")
            self.assertTrue(mode.status()["last_recording"])
            for bad in ({"type": "set_monitor", "source": "voice3"},
                        {"type": "set_test_mode", "enabled": True, "window_s": 3.0},
                        {"type": "set_focus", "duck_db": 5}):
                with self.assertRaises(ValueError):
                    mode.update(bad)

    def test_summary_line_reports_timing_and_lip_pick(self):
        mode = TestMode(loader=lambda name: (fake_model(), {"model": "fake", "device": "cpu"}))
        mode.update({"type": "set_test_mode", "enabled": True, "window_s": 1.0, "hop_s": 0.25})
        self.wait_state(mode, "ready")
        mode.attach("s", RATE)
        s1, s2, _ = sources(2)
        pcm = ((s1 + s2) * 32767).astype("<i2")
        for seq, start in enumerate(range(0, len(pcm), 960)):
            mode.feed(pcm[start:start + 960].tobytes(), start, seq * 20_000)
            mode.output(960)
            time.sleep(0.005)
        time.sleep(0.3)
        line = mode.summary()
        mode.detach("s")
        for part in ("fake on cpu", "window 1.0 s", "median/p95", "dropped late", "lip pick:", "face not visible 100%"):
            self.assertIn(part, line)

    def test_load_failure_is_reported_not_raised(self):
        def broken(name):
            raise RuntimeError("no CUDA")
        mode = TestMode(loader=broken)
        mode.update({"type": "set_test_mode", "enabled": True})
        self.wait_state(mode, "error")
        self.assertIn("no CUDA", mode.status()["error"])
        mode.attach("s", RATE)
        mode.feed(bytes(1920), 0, 0)
        self.assertIsNone(mode.output(960))



class DeviceTests(unittest.TestCase):
    def torch(self, cuda):
        return SimpleNamespace(__version__="2.9.1+cpu", version=SimpleNamespace(cuda="12.6" if cuda else None),
                               cuda=SimpleNamespace(is_available=lambda: cuda))

    def test_never_silently_uses_cpu_when_a_gpu_exists(self):
        self.assertEqual(choose_device(self.torch(True), "auto", "NVIDIA RTX A3000"), "cuda")
        with patch.dict(os.environ, {"LIPSYNC_ALLOW_CPU": ""}):
            with self.assertRaisesRegex(RuntimeError, "Setup-Models.cmd"):
                choose_device(self.torch(False), "auto", "NVIDIA RTX A3000")
        with patch.dict(os.environ, {"LIPSYNC_ALLOW_CPU": "1"}):
            self.assertEqual(choose_device(self.torch(False), "auto", "NVIDIA RTX A3000"), "cpu")
        self.assertEqual(choose_device(self.torch(False), "auto", None), "cpu")
        with self.assertRaises(RuntimeError):
            choose_device(self.torch(False), "cuda", None)


if __name__ == "__main__":
    unittest.main()
