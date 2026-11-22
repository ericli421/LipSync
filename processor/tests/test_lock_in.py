"""Lip timing after a late start, and keeping each voice on one person across pauses."""
import time
import unittest

import numpy as np

from processor.separation import StreamingSeparator
from processor.speaker_id import IdentityTracker
from processor.test_mode import TestMode
from processor.tests.test_separation import RATE, fake_model, sources, tone_db

MODEL = 8000


def turns(seconds=12.5):
    """A speaks, full pause, B speaks, full pause, A again (no overlap across pauses)."""
    t = np.arange(int(seconds * MODEL)) / MODEL
    a = 0.3 * np.sin(2 * np.pi * 300 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    b = 0.3 * np.sin(2 * np.pi * 700 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    a_on = (t < 2.5) | ((t >= 10) & (t < 12.5))
    b_on = (t >= 5) & (t < 7.5)
    return a * a_on, b * b_on


def loudest_first_model(src_a, src_b, window, hop):
    """Oracle separator that, like real ones, puts whoever is talking in output 0."""
    calls = [0]

    def separate(_mixture):
        end = window + calls[0] * hop
        calls[0] += 1
        pair = [src_a[end - window:end], src_b[end - window:end]]
        if np.sum(pair[1] ** 2) > np.sum(pair[0] ** 2):
            pair.reverse()
        return np.column_stack(pair)
    return separate


def tone_embed(audio):
    """Fake voiceprint at 16 kHz: energy near 300 Hz vs 700 Hz."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / 16000)
    vector = np.array([spectrum[(freqs > 250) & (freqs < 350)].sum(),
                       spectrum[(freqs > 650) & (freqs < 750)].sum(), 1e-6])
    return vector / np.linalg.norm(vector)


def stream(identity):
    src_a, src_b = turns()
    up = lambda x: np.repeat(x, RATE // MODEL)  # input-rate mixture (model sees the oracle)
    mix = up(src_a + src_b)
    separator = StreamingSeparator(RATE, loudest_first_model(src_a, src_b, int(2.0 * MODEL), int(0.25 * MODEL)),
                                   window_s=2.0, hop_s=0.25, identity=identity)
    outputs = []
    try:
        for start in range(0, len(mix), 960):
            separator.feed(mix[start:start + 960])
            while separator.processed < separator.fed and not separator.error:
                time.sleep(0.001)
            outputs.append((separator.read(960, "voice1"), separator.read(960, "voice2")))
    finally:
        separator.close()
    voice1 = np.concatenate([o[0] for o in outputs])
    voice2 = np.concatenate([o[1] for o in outputs])
    delay = separator.delay
    segment = lambda x, lo, hi: x[int(lo * RATE) + delay: int(hi * RATE) + delay]
    return voice1, voice2, segment, separator


class LockInTests(unittest.TestCase):
    def test_without_voice_lock_voice1_switches_person_after_a_pause(self):
        voice1, voice2, segment, _ = stream(identity=None)
        a_turn, b_turn = segment(voice1, 0.5, 2.0), segment(voice1, 5.5, 7.0)
        self.assertGreater(tone_db(a_turn, 300), tone_db(a_turn, 700) + 20)  # A in voice 1 ...
        self.assertGreater(tone_db(b_turn, 700), tone_db(b_turn, 300) + 20)  # ... then B in voice 1

    def test_voice_lock_never_breaks_confident_stitching(self):
        """Overruling stitching mid-speech mixed people into each voice in the live test."""
        tracker = IdentityTracker(tone_embed)
        voice1, voice2, segment, separator = stream(identity=tracker)
        for lo, hi in ((0.5, 2.0), (10.5, 12.0)):  # A stays in Voice 1
            part1, part2 = segment(voice1, lo, hi), segment(voice2, lo, hi)
            self.assertGreater(tone_db(part1, 300), tone_db(part2, 300) + 20, (lo, hi))
        self.assertEqual(tracker.swaps, 0)
        self.assertIsNotNone(separator.status()["identity"])

    def test_confident_stitching_is_never_overridden_but_pauses_are_fixed(self):
        tracker = IdentityTracker(tone_embed)
        t = np.arange(2 * MODEL) / MODEL
        both = np.column_stack([np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 700 * 1.0 * t)]) * 0.3
        both[:, 1] *= 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)  # independent loudness: not leakage
        tracker.assign(both, MODEL, 0.0)  # learns both voiceprints
        for _ in range(3):  # confident stitching: never swapped, however strong the evidence
            self.assertFalse(tracker.assign(both[:, ::-1].copy(), MODEL, 1.5)[1])
        self.assertIn("stitching confident", tracker.status()["decision"])
        self.assertTrue(tracker.assign(both[:, ::-1].copy(), MODEL, 0.1)[1])  # after a pause: fixed

    def test_leakage_of_the_same_talker_is_not_a_second_voice(self):
        tracker = IdentityTracker(tone_embed)
        t = np.arange(2 * MODEL) / MODEL
        voice = 0.3 * np.sin(2 * np.pi * 300 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        est = np.column_stack([voice, 0.5 * voice])  # -6 dB copy that follows the talker
        speech = tracker._active_audio(est, MODEL)
        self.assertIsNotNone(speech[0])
        self.assertIsNone(speech[1])


class LipTimingTests(unittest.TestCase):
    def run_session(self, enable_after_s):
        mode = TestMode(loader=lambda name: (fake_model(), {"model": "fake", "device": "cpu"}))
        mode.attach("s", RATE)
        s1, s2, env2 = sources(enable_after_s + 8)
        pcm = ((s1 + s2) * 32767).astype("<i2")
        t0 = 5_000_000_000  # phone clock in microseconds, as on a real device
        for seq, start in enumerate(range(0, len(pcm), 960)):
            now_us = t0 + seq * 20_000
            if seq % 2 == 0:  # 25 fps mouth that follows talker 2, on the same clock
                mode.add_mouth(now_us, {"face_found": True, "openness": float(env2[start])})
            if seq == int(enable_after_s * 50):
                mode.update({"type": "set_test_mode", "enabled": True, "window_s": 1.5, "hop_s": 0.25})
                while mode.status()["state"] != "ready":
                    time.sleep(0.01)
            mode.feed(pcm[start:start + 960].tobytes(), start, now_us)
            mode.output(960)
            if mode.separator:
                while mode.separator.processed < mode.separator.fed:
                    time.sleep(0.001)
        counts = dict(mode.picker.counts)
        mode.detach("s")
        return counts

    def test_lips_match_when_test_mode_starts_late(self):
        for delay in (0, 60):
            counts = self.run_session(delay)
            self.assertNotIn("face not visible", counts, (delay, counts))
            self.assertGreater(counts.get("matched", 0), 10, (delay, counts))


if __name__ == "__main__":
    unittest.main()
