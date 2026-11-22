import os
import time
import unittest

import numpy as np

os.environ.setdefault("LIPSYNC_LOG_FILE", "off")
from processor.active_speaker import (CROP, FPS, WEIGHTS, WEIGHTS_DIR, FacePicker, TalkingFacePicker,
                                      voice_activity)

RATE = 16000
SPF = RATE // FPS


class FakeDetector:
    """Bright crops mean "talking" (the real detector is tested separately)."""
    device = "cpu"

    def speaking(self, crops):
        return np.where(np.asarray(crops).mean(axis=(1, 2)) > 128, 4.0, -4.0)


def conversation(seconds=16, turn_s=2.0, seed=0):
    """Two talkers taking turns; returns per-frame `on_camera_talking` and the two voices."""
    rng = np.random.default_rng(seed)
    frames = seconds * FPS
    turn = (np.arange(frames) // int(turn_s * FPS)) % 2  # 0: on-camera person, 1: other
    voices = np.zeros((frames * SPF, 2))
    for k in range(2):
        active = np.repeat(turn == k, SPF)
        syllables = 0.7 + 0.3 * np.sin(2 * np.pi * 4 * np.arange(len(active)) / RATE)  # dips to -8 dB
        voices[:, k] = rng.standard_normal(len(active)) * 0.1 * syllables * active
    leak = 10 ** (-15 / 20)
    voices = voices + leak * voices[:, ::-1]
    return turn == 0, voices


class FacePickerRuleTests(unittest.TestCase):
    def test_voice_activity_reads_leakage_as_inactive(self):
        _, voices = conversation(4)
        activity = voice_activity(voices[: 2 * RATE])
        self.assertGreater(activity[:50, 0].mean(), 0.9)   # first turn: voice 1 talks
        self.assertLess(activity[:50, 1].mean(), 0.1)      # voice 2 only holds leakage

    def test_picks_the_voice_that_talks_with_the_mouth_and_holds_it(self):
        talking, voices = conversation(16)
        for target in (0, 1):
            chosen = voices if target == 0 else voices[:, ::-1]
            rule = FacePicker(sync_weight=0.0, tau_s=3.0, enter=0.25, switch=0.5)
            picks = []
            for end in range(2 * FPS, len(talking) + 1, FPS // 4):
                start = end - 2 * FPS
                speaking = np.where(talking[start:end], 4.0, -4.0)
                activity = voice_activity(chosen[start * SPF: end * SPF])
                picks.append(rule.update(np.zeros((2, 2 * FPS)), speaking, activity, np.ones(2 * FPS, bool), 0.25))
            self.assertEqual(picks[-1], target)
            self.assertEqual(rule.switches, 0)
            self.assertLess(picks.index(target), 12)  # decided within about 3 s


class TalkingFacePickerTests(unittest.TestCase):
    def feed(self, picker, talking, voices, fps=12, hide=None):
        """Replays frames at the phone's rate and picks every 0.25 s like StreamingSeparator."""
        picks = []
        sent = np.unique(np.floor(np.arange(0, len(talking) / FPS, 1 / fps) * FPS).astype(int))
        added = 0
        for end in range(2 * FPS, len(talking) + 1, FPS // 4):
            while added < len(sent) and sent[added] < end:
                i = sent[added]
                hidden = hide is not None and hide[0] <= i < hide[1]
                crop = None if hidden else np.full((CROP, CROP), 200 if talking[i] else 60, np.uint8)
                picker.add(int(i * 1e6 / FPS), None if hidden else 0.1)
                picker.add_face(int(i * 1e6 / FPS), crop)
                added += 1
            start = end - 2 * FPS
            picks.append(picker.pick(voices[start * SPF: end * SPF], int(start * 1e6 / FPS), RATE))
        return picks

    def test_follows_the_face_at_phone_frame_rate(self):
        talking, voices = conversation(16)
        picker = TalkingFacePicker(FakeDetector())
        picks = self.feed(picker, talking, voices[:, ::-1])
        self.assertEqual(picks[-1], 1)
        self.assertEqual(picker.switches, 0)
        self.assertTrue(picker.counts["matched"] > 0)

    def test_holds_the_pick_while_the_face_is_hidden(self):
        talking, voices = conversation(16)
        picker = TalkingFacePicker(FakeDetector())
        picks = self.feed(picker, talking, voices, hide=(8 * FPS, 13 * FPS))
        settled = picks[(6 * FPS - 2 * FPS) // (FPS // 4)]
        self.assertEqual(settled, 0)
        self.assertTrue(all(p == 0 for p in picks[picks.index(0):]))
        self.assertGreater(picker.counts["face not visible"], 0)

    def test_choice_can_be_reset_like_the_lip_picker(self):
        picker = TalkingFacePicker(FakeDetector())
        picker.choice, picker.reason = -1, "waiting for face"
        self.assertEqual((picker.choice, picker.reason), (-1, "waiting for face"))


@unittest.skipUnless((WEIGHTS_DIR / WEIGHTS["talkset"][0]).exists(), "Light-ASD weights not installed")
class PretrainedDetectorTests(unittest.TestCase):
    def test_pinned_weights_load_and_score_each_frame(self):
        from processor.active_speaker import ActiveSpeaker
        model = ActiveSpeaker("talkset")
        faces = np.random.default_rng(0).integers(0, 255, (30, CROP, CROP), dtype=np.uint8)
        scores = model.speaking(faces)
        self.assertEqual(scores.shape, (30,))
        self.assertTrue(np.all(np.isfinite(scores)))


class TestModeUsesTheFacePicker(unittest.TestCase):
    def test_loads_the_detector_and_feeds_it_faces(self):
        from processor.tests.test_separation import fake_model
        from processor.test_mode import TestMode
        mode = TestMode(loader=lambda name: (fake_model(), {"model": "fake", "device": "cpu"}),
                        face_loader=lambda device: FakeDetector())
        try:
            mode.attach("s", 48000)
            mode.update({"type": "set_test_mode", "enabled": True, "model": "auto"})
            deadline = time.time() + 10
            while mode.state == "loading" and time.time() < deadline:
                time.sleep(0.02)
            block = np.zeros(960, "<i2").tobytes()
            mode.feed(block, 0, 1_000_000)
            self.assertTrue(hasattr(mode.picker, "add_face"))
            mode.add_mouth(1_000_000, {"face_found": True, "openness": 0.2}, np.zeros((CROP, CROP), np.uint8))
            self.assertEqual(len(mode.picker.faces), 1)
            self.assertEqual(mode.status()["picker"], "face")
            self.assertTrue(mode.status()["picker_ready"])
            mode.update({"type": "set_test_mode", "enabled": True, "picker": "lips"})
            mode.feed(block, 960, 1_020_000)
            self.assertFalse(hasattr(mode.picker, "add_face"))
            with self.assertRaises(ValueError):
                mode.update({"type": "set_test_mode", "enabled": True, "picker": "ears"})
        finally:
            mode.detach("s")


if __name__ == "__main__":
    unittest.main()
