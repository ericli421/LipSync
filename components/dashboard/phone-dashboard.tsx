"use client";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { viewerUrl } from "@/lib/processor-url";

type Session = { session_id: string; device_name: string; sample_rate: number };
type Info = { face_found: boolean; openness: number; warning?: string };
export function PhoneDashboard() {
  const [state, setState] = useState("Connecting to backend…");
  const [session, setSession] = useState<Session | null>(null);
  const [info, setInfo] = useState<Info | null>(null);
  const [fps, setFps] = useState(0);
  const [level, setLevel] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const [monitor, setMonitor] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const player = useRef<{
    context: AudioContext;
    node: AudioWorkletNode;
  } | null>(null);
  const audioGeneration = useRef(0);
  const currentSession = useRef<Session | null>(null);
  const stopAudio = () => {
    audioGeneration.current++;
    const old = player.current;
    player.current = null;
    if (old) {
      old.node.disconnect();
      void old.context.close();
    }
    setMonitor(false);
  };
  useEffect(() => {
    let closed = false,
      frames = 0,
      lastLevel = 0,
      drawing = false;
    let socket: WebSocket;
    currentSession.current = null;
    setSession(null);
    setInfo(null);
    setLevel(0);
    setFps(0);
    setState("Connecting to backend…");
    const clear = () => {
      currentSession.current = null;
      setSession(null);
      setInfo(null);
      setLevel(0);
      setFps(0);
      stopAudio();
      const target = canvas.current;
      if (target)
        target.getContext("2d")?.clearRect(0, 0, target.width, target.height);
    };
    try {
      socket = new WebSocket(viewerUrl());
    } catch {
      setState("Invalid processor address. Check NEXT_PUBLIC_PROCESSOR_URL.");
      return;
    }
    const timer = setInterval(() => {
      setFps(frames);
      frames = 0;
    }, 1000);
    socket.onopen = () => setState("Waiting for phone");
    socket.onmessage = async (event) => {
      if (closed) return;
      const message = JSON.parse(event.data);
      if (message.type === "session") {
        clear();
        currentSession.current = message;
        setSession(message);
        setState("Phone connected");
      } else if (message.type === "waiting") {
        clear();
        setState("Waiting for phone");
      } else if (message.session_id !== currentSession.current?.session_id)
        return;
      else if (message.type === "discontinuity")
        player.current?.node.port.postMessage({ reset: true });
      else if (message.type === "audio") {
        const bytes = Uint8Array.from(atob(message.pcm_base64), (c) =>
          c.charCodeAt(0),
        );
        const view = new DataView(bytes.buffer),
          samples = new Float32Array(bytes.length / 2);
        let sum = 0;
        for (let i = 0; i < samples.length; i++) {
          samples[i] = view.getInt16(i * 2, true) / 32768;
          sum += samples[i] ** 2;
        }
        if (performance.now() - lastLevel > 100) {
          setLevel(Math.sqrt(sum / samples.length));
          lastLevel = performance.now();
        }
        player.current?.node.port.postMessage(
          {
            samples: samples.buffer,
            rate: currentSession.current!.sample_rate,
          },
          [samples.buffer],
        );
      } else if (message.type === "video" && !drawing) {
        drawing = true;
        try {
          const bytes = Uint8Array.from(atob(message.jpeg_base64), (c) =>
            c.charCodeAt(0),
          );
          const bitmap = await createImageBitmap(
            new Blob([bytes], { type: "image/jpeg" }),
          );
          const target = canvas.current;
          if (
            !closed &&
            target &&
            message.session_id === currentSession.current?.session_id
          ) {
            target.width = bitmap.width;
            target.height = bitmap.height;
            target.getContext("2d")?.drawImage(bitmap, 0, 0);
            setInfo(message.info);
            frames++;
          }
          bitmap.close();
        } finally {
          drawing = false;
        }
      }
    };
    const disconnected = () => {
      if (!closed) {
        clear();
        setState("Backend disconnected. Check the server and retry.");
      }
    };
    socket.onerror = disconnected;
    socket.onclose = disconnected;
    const visibility = () => {
      if (document.hidden) stopAudio();
    };
    document.addEventListener("visibilitychange", visibility);
    return () => {
      closed = true;
      clearInterval(timer);
      socket.close();
      stopAudio();
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [attempt]);

  async function toggleAudio() {
    if (player.current || monitor) {
      stopAudio();
      return;
    }
    setAudioError(null);
    const generation = ++audioGeneration.current;
    const sessionId = currentSession.current?.session_id;
    const context = new AudioContext();
    try {
      await context.resume();
      await context.audioWorklet.addModule("/pcm-player.js");
      if (
        generation !== audioGeneration.current ||
        sessionId !== currentSession.current?.session_id
      ) {
        await context.close();
        return;
      }
      const node = new AudioWorkletNode(context, "pcm-player");
      node.connect(context.destination);
      player.current = { context, node };
      setMonitor(true);
    } catch (error) {
      await context.close();
      setAudioError(
        error instanceof Error ? error.message : "Audio playback unavailable",
      );
    }
  }
  return (
    <section className="overflow-hidden rounded-xl border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b p-5">
        <div>
          <h2 className="font-semibold">
            {session?.device_name ?? "Android phone"}
          </h2>
          <p role="status" className="mt-1 text-sm text-muted-foreground">
            {state}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => setAttempt((a) => a + 1)}>
            Reconnect
          </Button>
          <Button
            disabled={!session}
            aria-pressed={monitor}
            onClick={toggleAudio}
          >
            {monitor ? "Mute monitor" : "Listen · headphones"}
          </Button>
        </div>
      </div>
      <div className="relative flex min-h-72 items-center justify-center bg-black aspect-video">
        <canvas
          ref={canvas}
          aria-label="Phone camera with face tracking"
          className="max-h-[65vh] max-w-full object-contain"
        />
        {!session && (
          <p className="absolute max-w-sm px-6 text-center text-sm text-white/75">
            Start capture in the APK using this laptop’s local network address.
            No laptop camera access is needed.
          </p>
        )}
      </div>
      <div className="grid gap-5 p-5 text-sm sm:grid-cols-4">
        <div>
          <p className="text-muted-foreground">Audio mode</p>
          <p className="mt-1 font-medium">Raw passthrough</p>
        </div>
        <div>
          <label htmlFor="phone-level" className="text-muted-foreground">
            Microphone level
          </label>
          <meter
            id="phone-level"
            value={level}
            min={0}
            max={1}
            className="mt-2 block w-full"
          />
        </div>
        <div>
          <p className="text-muted-foreground">Face / mouth openness</p>
          <p className="mt-1 font-mono">
            {info
              ? `${info.face_found ? "Detected" : "No face"} / ${info.openness.toFixed(2)}`
              : "Waiting for video"}
          </p>
        </div>
        <div>
          <p className="text-muted-foreground">Processed FPS</p>
          <p className="mt-1 font-mono">{fps}</p>
        </div>
      </div>
      <p className="border-t px-5 py-3 text-sm text-muted-foreground">
        Use headphones to avoid feedback. Voice isolation and captions are not
        enabled in this milestone.
      </p>
      {(info?.warning || audioError) && (
        <p role="alert" className="border-t px-5 py-3 text-sm text-destructive">
          {audioError ?? info?.warning}
        </p>
      )}
    </section>
  );
}
