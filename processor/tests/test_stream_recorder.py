import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from processor.stream_recorder import StreamRecorder

RATE = 48000


def jpeg(value):
    import cv2
    ok, data = cv2.imencode(".jpg", np.full((96, 128, 3), value, np.uint8))
    return data.tobytes()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg not installed")
class StreamRecorderTests(unittest.TestCase):
    def test_writes_an_mp4_with_audio_and_video_in_sync_with_the_delay(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = StreamRecorder(root)
            recorder.update({"type": "set_stream_recording", "enabled": True})
            capture0 = 5_000_000
            for k in range(150):  # 3 s of 20 ms blocks
                first = 1000 * 960 + k * 960  # the phone's sample clock does not start at 0
                recorder.audio(np.full(960, 1000 * (k % 2), "<i2").tobytes(), first,
                               capture0 + k * 20_000, RATE)
                if k % 4 == 0:  # 12.5 fps video
                    recorder.video(jpeg(40 + k), capture0 + k * 20_000, 0.75)
            recorder.audio(bytes(1920), 1000 * 960 + 160 * 960, capture0 + 160 * 20_000, RATE)  # a gap
            self.assertTrue(recorder.status()["recording"])
            recorder.stop()
            deadline = time.time() + 60
            while recorder.status()["finishing"] and time.time() < deadline:
                time.sleep(0.05)
            out = Path(recorder.status()["last"])
            self.assertEqual(out.name, "lipsync-stream.mp4")
            probe = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration", "-of", "json",
                 str(out)], capture_output=True, text=True, check=True).stdout)
            kinds = sorted(s["codec_type"] for s in probe["streams"])
            self.assertEqual(kinds, ["audio", "video"])
            self.assertAlmostEqual(float(probe["format"]["duration"]), 3.22, delta=0.15)
            self.assertFalse((out.parent / "frames").exists())

    def test_rejects_bad_control_and_ignores_media_when_idle(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = StreamRecorder(root)
            with self.assertRaises(ValueError):
                recorder.update({"type": "set_stream_recording", "enabled": "yes"})
            recorder.audio(bytes(1920), 0, 0, RATE)
            recorder.video(jpeg(0), 0, 0.0)
            self.assertFalse(any(Path(root).iterdir()))


if __name__ == "__main__":
    unittest.main()
