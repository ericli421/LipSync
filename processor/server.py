"""
WebSocket frame processor for the LipSync dashboard.

The browser sends JPEG-encoded camera frames; each one is run through
test2.process_frame (MediaPipe face landmarker + lip annotations) and sent back as:
    1. a text message with JSON info  ({"face_found", "openness", "mouth_box"})
    2. a binary message with the annotated JPEG frame
If a frame can't be handled, only a text message {"error": "..."} is sent instead.

Run from the repo root:
    pip install -r processor/requirements.txt
    uvicorn processor.server:app --port 8000

The dashboard connects to ws://127.0.0.1:8000/ws (not "localhost": Firefox resolves
that to IPv6 ::1 first, which uvicorn does not listen on by default).
"""

import os
import sys
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

# Make the repo root importable so we can reuse test2.py as-is.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test2 import create_landmarker, process_frame  # noqa: E402

JPEG_QUALITY = 80

app = FastAPI(title="LipSync processor")


@app.get("/")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def process_stream(sock: WebSocket):
    await sock.accept()
    # One landmarker per connection: VIDEO mode requires increasing timestamps per stream.
    landmarker = create_landmarker()
    start = time.monotonic()
    last_ts = -1
    try:
        while True:
            data = await sock.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                # Reply anyway so the client (which waits for each reply) keeps going.
                await sock.send_json({"error": "could not decode frame"})
                continue

            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts
            # MediaPipe is blocking; keep it off the event loop.
            out, info = await run_in_threadpool(process_frame, landmarker, frame, ts)

            ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                await sock.send_json({"error": "could not encode frame"})
                continue
            await sock.send_json(info)
            await sock.send_bytes(jpg.tobytes())
    except WebSocketDisconnect:
        pass
    finally:
        landmarker.close()
