"""Follow one person's face across frames; viewer selection; server control validation."""
import os
os.environ.setdefault("LIPSYNC_LOG_FILE", "off")
from types import SimpleNamespace
import unittest

from processor.face_target import FaceTarget


def face(cx, cy, size):
    """Fake MediaPipe landmark list: the four corners of a square face."""
    h = size / 2
    return [SimpleNamespace(x=cx + dx, y=cy + dy) for dx in (-h, h) for dy in (-h, h)]


def centre_x(landmarks):
    return round((min(p.x for p in landmarks) + max(p.x for p in landmarks)) / 2, 2)


def largest_in_centre(faces):
    """The old per-frame rule, for comparison."""
    return max(faces, key=lambda f: (max(p.x for p in f) - min(p.x for p in f)))


class FaceTargetTests(unittest.TestCase):
    def test_two_similar_faces_do_not_make_the_target_jump(self):
        target, old_rule = FaceTarget(), []
        picks = []
        for k in range(50):  # left and right person, sizes alternate by a few percent
            wobble = 0.01 if k % 2 else -0.01
            faces = [face(0.4, 0.5, 0.20 + wobble), face(0.6, 0.5, 0.20 - wobble)]
            chosen = target.choose(faces, k * 0.04)
            picks.append(centre_x(chosen))
            old_rule.append(centre_x(largest_in_centre(faces)))
        self.assertEqual(len(set(picks)), 1)            # always the same person
        self.assertGreater(len(set(old_rule)), 1)       # the old rule flips between them
        self.assertEqual(target.status()["switches"], 0)

    def test_short_dropout_is_held_and_long_loss_reacquires(self):
        target = FaceTarget(lost_after_s=1.0)
        left, right = face(0.35, 0.5, 0.2), face(0.65, 0.5, 0.25)
        self.assertIs(target.choose([left, right], 0.0), right)       # automatic: largest in centre
        self.assertIsNone(target.choose([left], 0.5))                  # target briefly missing: wait
        self.assertIs(target.choose([left, right], 0.7), right)        # same person comes back
        self.assertIs(target.choose([left], 2.0), left)                 # gone > 1 s: take who is there
        self.assertEqual(target.status()["switches"], 1)

    def test_click_and_next_select_a_person_who_stays_selected(self):
        target = FaceTarget(lost_after_s=0.5)
        left, right = face(0.3, 0.5, 0.2), face(0.7, 0.5, 0.2)
        target.choose([left, right], 0.0)
        target.select_point(0.25, 0.45)
        self.assertIs(target.choose([left, right], 0.1), left)
        self.assertEqual(target.status()["mode"], "selected")
        self.assertIsNone(target.choose([right], 2.0))                # selected person gone: do not swap
        self.assertIs(target.choose([left, right], 2.1), left)
        target.select_next()
        self.assertIs(target.choose([left, right], 2.2), right)       # left to right
        target.select_next()
        self.assertIs(target.choose([left, right], 2.3), left)        # wraps around
        target.clear()
        self.assertEqual(target.status()["mode"], "auto")

    def test_server_validates_select_face(self):
        from processor import server
        server.apply_control({"type": "select_face", "x": 0.2, "y": 0.8}, "test")
        server.apply_control({"type": "select_face", "next": True}, "test")
        server.apply_control({"type": "select_face", "clear": True}, "test")
        for bad in ({"x": 2, "y": 0.5}, {"x": True, "y": 0.5}, {}):
            with self.assertRaises(ValueError):
                server.apply_control({"type": "select_face", **bad}, "test")


if __name__ == "__main__":
    unittest.main()
