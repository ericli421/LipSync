"""Generated media through the real Next proxy; requires idle running services."""
import argparse
import asyncio
import base64
from contextlib import AsyncExitStack
import json
import math
import struct

import cv2
import numpy as np
import websockets


async def receive(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), 8))


async def verify(args):
    async with AsyncExitStack() as stack:
        listeners = [await stack.enter_async_context(websockets.connect(args.url))
                     for _ in range(5)]
        for ws in listeners:
            if (await receive(ws)).get('type') != 'waiting':
                raise RuntimeError('Phone capture is active; stop it before this synthetic test.')
            status = await receive(ws)
            assert status['type'] == 'pipeline' and 'source_label' in status
            assert 'test' not in status and 'log' not in status

        # Assert the proxy leads to the dedicated read-only endpoint.
        async with websockets.connect(args.url) as probe:
            await receive(probe)
            await receive(probe)
            await probe.send(json.dumps({'type': 'set_layer', 'layer': 'voice_band', 'enabled': False}))
            try:
                await probe.recv()
                raise AssertionError('Listener endpoint accepted controls')
            except websockets.ConnectionClosed as closed:
                assert closed.code == 1008

        image = np.full((180, 320, 3), 30, np.uint8)
        cv2.putText(image, 'Generated LAN test', (15, 95), cv2.FONT_HERSHEY_SIMPLEX, .6, (80, 240, 160), 1)
        ok, jpeg = cv2.imencode('.jpg', image)
        assert ok
        encoded = base64.b64encode(jpeg).decode()
        previous_session = None
        for run, seconds in enumerate((args.seconds, 1)):
            async with websockets.connect(args.capture_url) as phone:
                await phone.send(json.dumps({'type': 'start', 'protocol_version': 1,
                    'device_name': 'Generated judge transport test', 'sample_rate': 48000,
                    'channels': 1, 'format': 'pcm_s16le'}))
                initial = await receive(phone)
                assert initial['type'] == 'session', initial
                session = initial['session_id']
                assert session != previous_session
                previous_session = session
                for ws in listeners:
                    message = await receive(ws)
                    assert message['type'] == 'session' and message['session_id'] == session
                    assert 'device_name' not in message

                async def collect(ws, audience, audio):
                    video, statuses = 0, 0
                    while True:
                        message = await receive(ws)
                        kind = message['type']
                        if kind == 'waiting':
                            return audio, video, statuses
                        if kind == 'audio':
                            assert message['session_id'] == session
                            audio[message['seq']] = message['pcm_base64']
                            if not audience and len(audio) == count:
                                return audio, video, statuses
                        elif kind == 'video':
                            assert audience and message['session_id'] == session
                            assert 'info' not in message
                            video += 1
                        elif kind == 'pipeline':
                            if audience:
                                assert 'log' not in message and 'test' not in message
                                assert 'source_label' in message
                            statuses += 1

                captured = [{} for _ in range(6)]
                readers = [asyncio.create_task(collect(ws, True, captured[index]))
                           for index, ws in enumerate(listeners)]
                owner = asyncio.create_task(collect(phone, False, captured[-1]))
                try:
                    count = math.ceil(seconds * 50)
                    clock = asyncio.get_running_loop()
                    began = clock.time()
                    for seq in range(count):
                        await asyncio.sleep(max(0, began + seq * .02 - clock.time()))
                        pcm = b''.join(struct.pack('<h', round(160 * math.sin(
                            (seq * 960 + sample) * 2 * math.pi * 333 / 48000))) for sample in range(960))
                        await phone.send(json.dumps({'type': 'audio', 'session_id': session,
                            'seq': seq, 'first_sample': seq * 960,
                            'capture_time_us': 1000000 + seq * 20000, 'timing': 'estimated',
                            'pcm_base64': base64.b64encode(pcm).decode()}))
                        if seq % 10 == 0:
                            await phone.send(json.dumps({'type': 'video', 'session_id': session,
                                'seq': seq // 10, 'capture_time_us': 1000000 + seq * 20000,
                                'timing': 'estimated', 'jpeg_base64': encoded}))
                    # Read every final block before stop resets subscriber queues.
                    async with asyncio.timeout(8):
                        while not all(len(audio) == count for audio in captured):
                            for task in readers:
                                if task.done():
                                    await task  # surface transport/validation errors
                                    raise AssertionError('Capture ended before all audio arrived')
                            await asyncio.sleep(.01)
                    await phone.send(json.dumps({'type': 'stop'}))
                    final = (await owner)[0]
                    results = await asyncio.gather(*readers)
                    assert len(final) == count, (len(final), count)
                    for audio, video, statuses in results:
                        assert audio == final, f'PCM mismatch or missing blocks: {len(audio)}/{count}'
                        assert video > 0 and statuses > 0
                    print(f'Session {run + 1}: five listeners, {count} identical final PCM blocks each, video/status OK', flush=True)
                finally:
                    for task in [owner, *readers]:
                        task.cancel()
                    await asyncio.gather(owner, *readers, return_exceptions=True)
    print('PASS: same-origin media, five-client fan-out, read-only access, two clean sessions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:3000/live/listen')
    parser.add_argument('--capture-url', default='ws://127.0.0.1:8000/ws/capture')
    parser.add_argument('--seconds', type=float, default=5)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120:
        parser.error('--seconds must be between 1 and 120')
    asyncio.run(verify(args))
