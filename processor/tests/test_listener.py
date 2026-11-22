"""Audience fan-out, sanitization, queue bounds, and control isolation."""
import asyncio
import base64
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from processor.bridge import Hub, Session
from processor.listener import ListenerOutbox
from processor.server import app, hub, send_loop
from processor.tests.test_bridge import hello, audio, receive


class ListenerQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_socket_is_closed_instead_of_blocking_capture(self):
        class SlowSocket:
            closed = None

            async def send_json(self, message):
                await asyncio.Future()

            async def close(self, code, reason):
                self.closed = code

        sock, out = SlowSocket(), ListenerOutbox()
        out.put({'type': 'waiting'})
        await asyncio.wait_for(send_loop(sock, out), 2)
        self.assertEqual(sock.closed, 1013)

    async def test_public_status_excludes_private_details_and_tracks_source(self):
        out = ListenerOutbox()
        out.put({'type': 'session', 'session_id': 's', 'device_name': 'private phone', 'sample_rate': 48000})
        self.assertNotIn('device_name', await out.get())
        out.put({'type': 'pipeline', 'log': [{'message': '/private/path'}], 'error': 'private detail',
                 'test': {'enabled': True, 'state': 'running', 'monitor': 'focus',
                          'live': {'delay_ms': 750}, 'last_recording': '/recordings/private'},
                 'layers': {'voice_band': True, 'auto_volume': True}})
        status = await out.get()
        self.assertEqual(status['source'], 'focus')
        self.assertEqual(status['source_delay_ms'], 750)
        self.assertNotIn('log', status)
        self.assertNotIn('/private', str(status))
        self.assertNotIn('test', status)
        out.put({'type': 'pipeline', 'test': {'enabled': False, 'monitor': 'voice2'}})
        self.assertEqual((await out.get())['source'], 'original')

    async def test_video_is_throttled_and_session_reset_drops_old_media(self):
        out = ListenerOutbox()
        with patch('processor.listener.time.monotonic', side_effect=[1, 1.04, 1.125]):
            for seq in range(3):
                out.put({'type': 'video', 'session_id': 's', 'seq': seq, 'jpeg_base64': 'x', 'info': {'warning': 'private'}})
        video = await out.get()
        self.assertEqual(video['seq'], 2)
        self.assertNotIn('info', video)
        for seq in range(11):
            out.put({'type': 'audio', 'session_id': 's', 'seq': seq, 'first_sample': seq*960, 'pcm_base64': 'x'})
        self.assertEqual(len(out.audio), 1)
        self.assertEqual((await out.get())['type'], 'discontinuity')
        out.reset({'type': 'session', 'session_id': 'new', 'sample_rate': 16000})
        self.assertEqual((await out.get())['session_id'], 'new')
        self.assertFalse(out.audio)
        self.assertIsNone(out.video)


class ListenerSocketTests(unittest.TestCase):
    def test_controls_are_rejected_and_never_applied(self):
        with TestClient(app) as client, client.websocket_connect('/ws/listen') as listener:
            self.assertEqual(listener.receive_json()['type'], 'waiting')
            self.assertEqual(listener.receive_json()['type'], 'pipeline')
            with patch('processor.server.apply_control') as control:
                listener.send_json({'type': 'set_layer', 'layer': 'voice_band', 'enabled': False})
                with self.assertRaises(WebSocketDisconnect) as closed:
                    listener.receive_json()
                self.assertEqual(closed.exception.code, 1008)
                control.assert_not_called()
        self.assertFalse(hub.listeners)

    def test_five_judges_get_final_pcm_and_video_then_clear_on_reconnect(self):
        processed, final = bytes([1, 2])*960, bytes([3, 4])*960
        with TestClient(app) as client, ExitStack() as stack:
            listeners = [stack.enter_context(client.websocket_connect('/ws/listen')) for _ in range(5)]
            for listener in listeners:
                self.assertEqual(listener.receive_json()['type'], 'waiting')
                listener.receive_json()
            for run in range(2):
                with client.websocket_connect('/ws/capture') as phone:
                    phone.send_json(hello())
                    session = receive(phone)['session_id']
                    for listener in listeners:
                        self.assertEqual(receive(listener)['session_id'], session)
                    with patch('processor.server.AudioChain.process', return_value=processed), \
                            patch('processor.server.AudioChain.finish', return_value=final):
                        phone.send_json(audio(session))
                        self.assertEqual(base64.b64decode(receive(phone)['pcm_base64']), final)
                        for listener in listeners:
                            received = receive(listener)
                            self.assertEqual(received['session_id'], session)
                            self.assertEqual(base64.b64decode(received['pcm_base64']), final)
                    with patch('processor.server.FrameProcessor.process', return_value=(b'jpeg', {'face_found': True, 'openness': .2})):
                        phone.send_json({'type': 'video', 'session_id': session, 'seq': 0,
                                         'capture_time_us': 100, 'timing': 'estimated', 'jpeg_base64': 'anBlZw=='})
                        self.assertEqual(receive(phone)['type'], 'video_ack')
                        for listener in listeners:
                            self.assertEqual(receive(listener)['jpeg_base64'], 'anBlZw==')
                    phone.send_json({'type': 'stop'})
                    for listener in listeners:
                        self.assertEqual(receive(listener)['type'], 'waiting')
        self.assertFalse(hub.listeners)


if __name__ == '__main__':
    unittest.main()
