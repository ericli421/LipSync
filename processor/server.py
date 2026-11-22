"""Run: uvicorn processor.server:app --host 0.0.0.0 --port 8000"""
import asyncio
import base64
from contextlib import suppress
import json
import time
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from processor.audio_chain import AudioChain
from processor.bridge import Hub, MAX_MESSAGE_BYTES, MODE, Outbox, Session
from processor.test_mode import TestMode

app = FastAPI(title="LipSync local processor")
hub = Hub()
loop = None
CHAIN_CONTROLS = ("set_layer", "set_band")


def pipeline_status():
    status = hub.chain.status() if hub.chain is not None else hub.idle_pipeline()
    return {**status, "test": test_mode.status()}


def notify_viewers():
    """Called from the model-loading thread when test mode changes state."""
    if loop is not None:
        loop.call_soon_threadsafe(lambda: hub.publish(pipeline_status()))


test_mode = TestMode(on_change=notify_viewers)

class FrameProcessor:
    """Load vision lazily; a missing face model does not stop raw audio."""
    def __init__(self):
        self.landmarker = None
        self.last_ts = -1
        self.warning = None
        self.lock = threading.Lock()

    def process(self, jpeg, capture_us):
        with self.lock:
            return self._process(jpeg, capture_us)

    def _process(self, jpeg, capture_us):
        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode JPEG")
        if max(frame.shape[:2]) > 1920:
            raise ValueError("frame exceeds 1920 pixels")
        info = {"face_found": False, "openness": 0, "mouth_box": None}
        if self.warning is None:
            try:
                from test2 import create_landmarker, process_frame
                if self.landmarker is None:
                    self.landmarker = create_landmarker()
                timestamp = max(capture_us // 1000, self.last_ts + 1)
                self.last_ts = timestamp
                frame, info = process_frame(self.landmarker, frame, timestamp)
            except Exception as exc:
                self.warning = f"Face tracking unavailable: {exc}"
        if self.warning:
            info["warning"] = self.warning
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("could not encode JPEG")
        return encoded.tobytes(), info

    def close(self):
        # A cancelled socket task may leave its worker thread finishing a frame.
        with self.lock:
            if self.landmarker is not None:
                self.landmarker.close()

@app.get("/")
def health():
    return {"status": "ok", "protocol_version": 1, "mode": MODE}

async def receive_json(sock):
    event = await sock.receive()
    if event["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(event.get("code", 1000))
    text = event.get("text")
    if not isinstance(text, str):
        raise ValueError("expected JSON text message")
    if len(text) > MAX_MESSAGE_BYTES:
        raise ValueError("message too large")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value

async def send_loop(sock, outbox):
    try:
        while True:
            await asyncio.wait_for(sock.send_json(await outbox.get()), timeout=0.5)
    except (TimeoutError, RuntimeError, WebSocketDisconnect):
        with suppress(Exception):
            await sock.close(code=1013, reason="connection too slow; reconnect")

@app.websocket("/ws/viewer")
async def viewer(sock: WebSocket):
    global loop
    await sock.accept()
    loop = asyncio.get_running_loop()
    outbox = Outbox()
    hub.viewers.add(outbox)
    outbox.put(hub.status())
    outbox.put(pipeline_status())
    sender = asyncio.create_task(send_loop(sock, outbox))
    try:
        while True:
            text = await sock.receive_text()
            try:
                message = json.loads(text) if len(text) <= 4096 else None
                if not isinstance(message, dict):
                    raise ValueError("expected JSON object")
                if message.get("type") in CHAIN_CONTROLS:
                    hub.settings.update(message)
                else:
                    test_mode.update(message)
            except ValueError as exc:  # JSONDecodeError is a ValueError
                outbox.put({"type": "control_error", "message": str(exc)})
                continue
            hub.publish(pipeline_status())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.viewers.discard(outbox)
        sender.cancel()
        with suppress(asyncio.CancelledError):
            await sender

@app.websocket("/ws/capture")
async def capture(sock: WebSocket):
    global loop
    await sock.accept()
    loop = asyncio.get_running_loop()
    session = None
    sender = worker = None
    processor = FrameProcessor()
    pending = asyncio.Queue(maxsize=1)
    outbox = Outbox()

    async def process_video():
        while True:
            message, jpeg = await pending.get()
            try:
                encoded, info = await run_in_threadpool(processor.process, jpeg, message["capture_time_us"])
                if hub.session is session:
                    test_mode.add_mouth(message["capture_time_us"], info)
                    hub.publish({**message, "jpeg_base64": base64.b64encode(encoded).decode(), "info": info})
                    outbox.put({"type": "video_ack", "seq": message["seq"], "session_id": session.id})
            except Exception as exc:
                outbox.put({"type": "video_ack", "seq": message["seq"], "session_id": session.id, "error": str(exc)})
            finally:
                pending.task_done()
    try:
        session = Session.start(await asyncio.wait_for(receive_json(sock), 5))
        chain = AudioChain(session.sample_rate, hub.settings)
        hub.begin(session, outbox)
        hub.chain = chain
        test_mode.attach(session.id, session.sample_rate)
        sender = asyncio.create_task(send_loop(sock, outbox))
        worker = asyncio.create_task(process_video())
        while True:
            message = await receive_json(sock)
            if message.get("type") == "stop":
                break
            data = session.validate(message)
            if message["type"] == "audio":
                try:
                    processed = chain.process(data)
                except Exception as exc:  # never stop the relay: fall back to raw audio
                    chain.error, processed = f"Audio chain failed: {exc}", data
                test_mode.feed(processed, message["first_sample"], message["capture_time_us"])
                try:  # test mode: a separated voice or the lip-focused mix, delayed
                    processed = test_mode.output(len(processed) // 2) or processed
                except Exception as exc:
                    test_mode.error = f"Test mode output failed: {exc}"
                hub.publish({**message, "pcm_base64": base64.b64encode(processed).decode()}, include_phone=True)
                if message["seq"] % 5 == 0:  # meters every 100 ms
                    hub.publish({**pipeline_status(), "session_id": session.id})
            elif not pending.full():
                pending.put_nowait((message, data))
            else:
                outbox.put({"type": "video_ack", "seq": message["seq"], "session_id": session.id, "error": "video queue full"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    except (ValueError, TimeoutError) as exc:
        with suppress(Exception):
            await sock.send_json({"type": "error", "message": str(exc)})
            await sock.close(code=1008)
    finally:
        if session:
            if hub.session is session:
                hub.chain = None
            test_mode.detach(session.id)
            hub.end(session)
        for task in (sender, worker):
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        await run_in_threadpool(processor.close)

@app.websocket("/ws")
async def browser_debug(sock: WebSocket):
    """Compatibility: binary JPEG in, metadata then annotated JPEG out."""
    await sock.accept()
    processor = FrameProcessor()
    try:
        while True:
            jpeg = await sock.receive_bytes()
            try:
                out, info = await run_in_threadpool(processor.process, jpeg, time.monotonic_ns() // 1000)
                await sock.send_json(info)
                await sock.send_bytes(out)
            except ValueError as exc:
                await sock.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        await run_in_threadpool(processor.close)
