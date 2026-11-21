"""Run with: .venv-local/Scripts/python -m unittest discover -s processor/tests -v"""
import asyncio
import base64
import unittest
import threading
from unittest.mock import patch

from fastapi.testclient import TestClient
from processor.bridge import Hub, Outbox, Session
from processor.server import app, hub


def hello():
    return {"type": "start", "protocol_version": 1, "device_name": "Test phone",
            "sample_rate": 48000, "channels": 1, "format": "pcm_s16le"}


def audio(session, seq=0):
    return {"type": "audio", "session_id": session, "seq": seq,
            "capture_time_us": 1_000_000 + seq * 20_000, "timing": "device",
            "first_sample": seq * 960, "pcm_base64": base64.b64encode(bytes(range(256)) * 7 + bytes(range(128))).decode()}


class ProtocolTests(unittest.TestCase):
    def test_pcm_bytes_and_sample_order(self):
        session = Session.start(hello())
        data = audio(session.id)
        self.assertEqual(len(session.validate(data)), 1920)
        with self.assertRaisesRegex(ValueError, "sequence/sample gap"):
            session.validate(data)

    def test_reject_invalid_format_and_payload(self):
        for changed in ({"channels": 2}, {"sample_rate": 8000}, {"protocol_version": 2}):
            with self.assertRaises(ValueError):
                Session.start({**hello(), **changed})
        for changed in ({"pcm_base64": "bad!"}, {"first_sample": -1}, {"session_id": "old"}, {"seq": True}):
            session = Session.start(hello())
            with self.assertRaises(ValueError):
                session.validate({**audio(session.id), **changed})

    def test_slow_subscriber_drops_old_audio_and_video(self):
        box = Outbox()
        for seq in range(25):
            box.put(audio("a", seq))
            box.put({"type": "video", "seq": seq})
        self.assertLessEqual(len(box.audio), 10)
        self.assertEqual(box.audio[-1]["seq"], 24)
        self.assertEqual(box.video["seq"], 24)
        self.assertEqual(box.controls[0]["type"], "discontinuity")

    def test_session_reset_drops_all_old_media(self):
        relay = Hub()
        viewer = Outbox()
        relay.viewers.add(viewer)
        first, second = Session.start(hello()), Session.start(hello())
        relay.begin(first, Outbox())
        relay.publish(audio(first.id))
        relay.end(first)
        self.assertFalse(viewer.audio)
        relay.begin(second, Outbox())
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(asyncio.run(viewer.get())["session_id"], second.id)


class SocketTests(unittest.TestCase):
    def test_audio_flows_while_video_worker_is_busy(self):
        entered, release = threading.Event(), threading.Event()
        def slow_video(*args):
            entered.set()
            release.wait(3)
            return b"jpeg", {"face_found": False, "openness": 0}
        try:
            with patch("processor.server.FrameProcessor.process", side_effect=slow_video):
                with TestClient(app) as client, client.websocket_connect("/ws/capture") as phone:
                    phone.send_json(hello())
                    session = phone.receive_json()["session_id"]
                    phone.send_json({"type": "video", "session_id": session, "seq": 0,
                                     "capture_time_us": 100, "timing": "estimated", "jpeg_base64": "anBlZw=="})
                    self.assertTrue(entered.wait(2))
                    message = audio(session)
                    phone.send_json(message)
                    self.assertEqual(phone.receive_json(), message)
                    release.set()
                    self.assertEqual(phone.receive_json()["type"], "video_ack")
        finally:
            release.set()

    def test_local_health_and_audio_fanout(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/").json()["mode"], "raw_passthrough")
            with client.websocket_connect("/ws/viewer") as a, client.websocket_connect("/ws/viewer") as b:
                self.assertEqual(a.receive_json()["type"], "waiting")
                self.assertEqual(b.receive_json()["type"], "waiting")
                with client.websocket_connect("/ws/capture") as phone:
                    phone.send_json(hello())
                    status = phone.receive_json()
                    self.assertEqual(a.receive_json(), status)
                    self.assertEqual(b.receive_json(), status)
                    message = audio(status["session_id"])
                    phone.send_json(message)
                    self.assertEqual(phone.receive_json(), message)
                    self.assertEqual(a.receive_json(), message)
                    self.assertEqual(b.receive_json(), message)
                    with client.websocket_connect("/ws/capture") as other:
                        other.send_json(hello())
                        self.assertIn("already capturing", other.receive_json()["message"])
                    phone.send_json({"type": "stop"})
                    self.assertEqual(a.receive_json()["type"], "waiting")
                    self.assertEqual(b.receive_json()["type"], "waiting")
        self.assertIsNone(hub.session)
        self.assertFalse(hub.viewers)

    def test_bad_json_and_sample_gap_close_capture(self):
        with TestClient(app) as client:
            with client.websocket_connect("/ws/capture") as phone:
                phone.send_text("[]")
                self.assertEqual(phone.receive_json()["type"], "error")
            with client.websocket_connect("/ws/capture") as phone:
                phone.send_json(hello())
                session = phone.receive_json()["session_id"]
                phone.send_json(audio(session, 1))
                self.assertIn("sequence/sample gap", phone.receive_json()["message"])

    def test_video_processing_and_legacy_endpoint(self):
        result = (b"processed", {"face_found": True, "openness": 0.2, "mouth_box": [1, 2, 3, 4]})
        with patch("processor.server.FrameProcessor.process", return_value=result):
            with TestClient(app) as client:
                with client.websocket_connect("/ws") as browser:
                    browser.send_bytes(b"jpeg")
                    self.assertTrue(browser.receive_json()["face_found"])
                    self.assertEqual(browser.receive_bytes(), b"processed")
                with client.websocket_connect("/ws/viewer") as viewer, client.websocket_connect("/ws/capture") as phone:
                    viewer.receive_json()
                    phone.send_json(hello())
                    session = phone.receive_json()["session_id"]
                    viewer.receive_json()
                    phone.send_json({"type": "video", "session_id": session, "seq": 0,
                                     "capture_time_us": 100, "timing": "estimated", "jpeg_base64": "anBlZw=="})
                    self.assertEqual(phone.receive_json()["type"], "video_ack")
                    self.assertEqual(base64.b64decode(viewer.receive_json()["jpeg_base64"]), b"processed")


if __name__ == "__main__":
    unittest.main()
