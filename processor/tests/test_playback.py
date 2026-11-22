import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("LIPSYNC_LOG_FILE", "off")
from fastapi.testclient import TestClient

from processor.listener import ListenerOutbox
from processor.playback import Playback


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        for name, age in (("stream-1/lipsync-stream.mp4", 20), ("stream-2/lipsync-stream.mp4", 10)):
            path = self.root / name
            path.parent.mkdir()
            path.write_bytes(b"\x00" * 1000)
            os.utime(path, (time.time() - age, time.time() - age))
        (self.root / "secret.txt").write_text("x")

    def tearDown(self):
        self.dir.cleanup()

    def test_plays_the_newest_recording_and_tracks_position(self):
        playback = Playback(self.root)
        self.assertEqual(playback.files(), ["stream-2/lipsync-stream.mp4", "stream-1/lipsync-stream.mp4"])
        playback.update({"enabled": True})
        status = playback.status()
        self.assertEqual((status["active"], status["file"]), (True, "stream-2/lipsync-stream.mp4"))
        self.assertGreaterEqual(status["position_s"], 0)
        playback.update({"enabled": True, "file": "stream-1/lipsync-stream.mp4"})
        self.assertEqual(playback.status()["file"], "stream-1/lipsync-stream.mp4")
        playback.update({"enabled": False})
        self.assertFalse(playback.status()["active"])

    def test_refuses_files_outside_recordings(self):
        playback = Playback(self.root)
        for bad in ("../x.mp4", "secret.txt", "missing.mp4", 5, "/etc/passwd.mp4"):
            self.assertIsNone(playback.resolve(bad))
        with self.assertRaises(ValueError):
            playback.update({"enabled": True, "file": "../../outside.mp4"})
        with self.assertRaises(ValueError):
            playback.update({"enabled": "yes"})
        with self.assertRaises(ValueError):
            Playback(self.root / "empty").update({"enabled": True})

    def test_listeners_get_only_the_playback_summary(self):
        outbox = ListenerOutbox()
        outbox.put({"type": "pipeline", "layers": {}, "test": {},
                    "playback": {"active": True, "file": "a.mp4", "position_s": 3.0, "files": ["a.mp4", "b.mp4"]}})
        self.assertEqual(outbox.pipeline["playback"], {"active": True, "file": "a.mp4", "position_s": 3.0})

    def test_media_endpoint_serves_ranges_and_hides_other_files(self):
        from processor import server
        with patch.object(server, "playback", Playback(self.root)):
            with TestClient(server.app) as client:
                whole = client.get("/media/stream-2/lipsync-stream.mp4")
                self.assertEqual((whole.status_code, len(whole.content)), (200, 1000))
                part = client.get("/media/stream-2/lipsync-stream.mp4", headers={"Range": "bytes=100-199"})
                self.assertEqual((part.status_code, len(part.content)), (206, 100))
                self.assertEqual(client.get("/media/secret.txt").status_code, 404)
                self.assertEqual(client.get("/media/..%2Fsecret.txt").status_code, 404)


if __name__ == "__main__":
    unittest.main()
