"use client";

import { useEffect, useState, type RefObject } from "react";

export const PROCESSOR_URL =
  process.env.NEXT_PUBLIC_PROCESSOR_URL ?? "ws://127.0.0.1:8000/ws";

/** Frames are downscaled to at most this width before being sent. */
const MAX_SEND_WIDTH = 640;
const JPEG_QUALITY = 0.8;

export type ProcessorInfo = {
  face_found: boolean;
  openness: number;
  mouth_box: [number, number, number, number] | null;
};

export type ProcessorState = "idle" | "connecting" | "streaming" | "error";

/**
 * Streams frames from `videoRef` to the Python processor (processor/server.py)
 * over a WebSocket and paints the annotated frames it returns onto `canvasRef`.
 *
 * Only one frame is in flight at a time: the next frame is sent after the
 * previous reply arrives, so latency never builds up if Python is slower than
 * the camera.
 */
export function useFrameProcessor(
  videoRef: RefObject<HTMLVideoElement | null>,
  canvasRef: RefObject<HTMLCanvasElement | null>,
  enabled: boolean,
  /** Bump to force a reconnect (e.g. after an error). */
  attempt = 0,
) {
  const [state, setState] = useState<ProcessorState>("idle");
  const [info, setInfo] = useState<ProcessorInfo | null>(null);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setState("idle");
      setInfo(null);
      setFps(0);
      return;
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    const capture = document.createElement("canvas");
    const captureCtx = capture.getContext("2d");
    if (!ctx || !captureCtx) return;

    let closed = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let frames = 0;
    const fpsTimer = setInterval(() => {
      setFps(frames);
      frames = 0;
    }, 1000);

    setState("connecting");
    const ws = new WebSocket(PROCESSOR_URL);
    ws.binaryType = "blob";

    const sendFrame = () => {
      if (closed || ws.readyState !== WebSocket.OPEN) return;
      // Video not producing frames yet — try again shortly.
      if (video.readyState < 2 || !video.videoWidth) {
        retryTimer = setTimeout(sendFrame, 50);
        return;
      }
      const scale = Math.min(1, MAX_SEND_WIDTH / video.videoWidth);
      capture.width = Math.round(video.videoWidth * scale);
      capture.height = Math.round(video.videoHeight * scale);
      captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
      capture.toBlob(
        (blob) => {
          if (closed || ws.readyState !== WebSocket.OPEN) return;
          if (blob) ws.send(blob);
          else retryTimer = setTimeout(sendFrame, 50);
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    };

    ws.onopen = () => sendFrame();

    ws.onmessage = async (e: MessageEvent<string | Blob>) => {
      if (closed) return;
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);
        if ("error" in msg) sendFrame(); // server skipped this frame; keep going
        else setInfo(msg as ProcessorInfo);
        return;
      }
      const bitmap = await createImageBitmap(e.data);
      if (closed) {
        bitmap.close();
        return;
      }
      if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
      if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      frames++;
      setState("streaming");
      sendFrame();
    };

    ws.onerror = () => !closed && setState("error");
    ws.onclose = () => !closed && setState("error");

    return () => {
      closed = true;
      clearTimeout(retryTimer);
      clearInterval(fpsTimer);
      ws.close();
    };
  }, [enabled, attempt, videoRef, canvasRef]);

  return { state, info, fps };
}
