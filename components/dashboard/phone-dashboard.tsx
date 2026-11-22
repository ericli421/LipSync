"use client";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { viewerUrl } from "@/lib/processor-url";
import {
  SeparationTestPanel,
  type TestStatus,
} from "@/components/dashboard/separation-test-panel";

type Session = { session_id: string; device_name: string; sample_rate: number };
type Info = { face_found: boolean; openness: number; warning?: string };
type Pipeline = {
  layers: { voice_band: boolean; auto_volume?: boolean };
  volume_gain_db?: number;
  band: { low_hz: number; high_hz: number; effective_high_hz: number | null };
  in_db: number | null;
  out_db: number | null;
  process_ms: number | null;
  error: string | null;
  test?: TestStatus;
  log?: {
    seq: number;
    time: string;
    level: string;
    source: string;
    message: string;
  }[];
};
const BANDS = [
  { label: "Voice 80 Hz – 8 kHz", low_hz: 80, high_hz: 8000 },
  { label: "Narrow 150 Hz – 5 kHz", low_hz: 150, high_hz: 5000 },
  { label: "Telephone 300 Hz – 3.4 kHz", low_hz: 300, high_hz: 3400 },
];
const db = (value: number | null) =>
  value === null ? "—" : `${value.toFixed(1)} dB`;
export function PhoneDashboard() {
  const [state, setState] = useState("Connecting to backend…");
  const [session, setSession] = useState<Session | null>(null);
  const [info, setInfo] = useState<Info | null>(null);
  const [fps, setFps] = useState(0);
  const [level, setLevel] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const [monitor, setMonitor] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<Pipeline | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const control = (message: object) => {
    setControlError(null);
    if (socketRef.current?.readyState === WebSocket.OPEN)
      socketRef.current.send(JSON.stringify(message));
    else setControlError("Backend not connected");
  };
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
      socketRef.current = socket;
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
      if (message.type === "pipeline") {
        setPipeline(message);
        return;
      }
      if (message.type === "control_error") {
        setControlError(message.message);
        return;
      }
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
          <p className="mt-1 font-medium">
            {pipeline?.test?.state === "running" &&
            pipeline.test.monitor !== "original"
              ? `Test · ${pipeline.test.monitor} (delayed)`
              : pipeline?.layers.voice_band
                ? "Voice band filter"
                : "Raw passthrough"}
          </p>
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
      <div className="border-t p-5 text-sm">
        <h3 className="font-semibold">Audio pipeline</h3>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <Button
            variant={pipeline?.layers.voice_band ? "default" : "outline"}
            aria-pressed={pipeline?.layers.voice_band ?? false}
            disabled={!pipeline}
            onClick={() =>
              control({
                type: "set_layer",
                layer: "voice_band",
                enabled: !pipeline?.layers.voice_band,
              })
            }
          >
            Voice band · {pipeline?.layers.voice_band ? "On" : "Off"}
          </Button>
          <Button
            variant={pipeline?.layers.auto_volume ? "default" : "outline"}
            aria-pressed={pipeline?.layers.auto_volume ?? false}
            disabled={!pipeline}
            onClick={() =>
              control({
                type: "set_layer",
                layer: "auto_volume",
                enabled: !pipeline?.layers.auto_volume,
              })
            }
          >
            Auto volume · {pipeline?.layers.auto_volume ? "On" : "Off"}
            {pipeline?.layers.auto_volume && pipeline.volume_gain_db != null
              ? ` (${pipeline.volume_gain_db > 0 ? "+" : ""}${pipeline.volume_gain_db.toFixed(0)} dB)`
              : ""}
          </Button>
          <label className="flex items-center gap-2">
            <span className="text-muted-foreground">Band</span>
            <select
              className="rounded-md border bg-background px-2 py-1"
              disabled={!pipeline}
              value={BANDS.findIndex(
                (b) =>
                  b.low_hz === pipeline?.band.low_hz &&
                  b.high_hz === pipeline?.band.high_hz,
              )}
              onChange={(event) => {
                const band = BANDS[Number(event.target.value)];
                if (band)
                  control({
                    type: "set_band",
                    low_hz: band.low_hz,
                    high_hz: band.high_hz,
                  });
              }}
            >
              <option value={-1} disabled>
                Custom
              </option>
              {BANDS.map((band, index) => (
                <option key={band.label} value={index}>
                  {band.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <dl className="mt-3 grid gap-3 font-mono sm:grid-cols-4">
          <div>
            <dt className="font-sans text-muted-foreground">Input</dt>
            <dd>{db(pipeline?.in_db ?? null)}</dd>
          </div>
          <div>
            <dt className="font-sans text-muted-foreground">Output</dt>
            <dd>{db(pipeline?.out_db ?? null)}</dd>
          </div>
          <div>
            <dt className="font-sans text-muted-foreground">Removed</dt>
            <dd>
              {pipeline?.in_db != null && pipeline.out_db != null
                ? `${(pipeline.in_db - pipeline.out_db).toFixed(1)} dB`
                : "—"}
            </dd>
          </div>
          <div>
            <dt className="font-sans text-muted-foreground">Cost / 20 ms</dt>
            <dd>
              {pipeline?.process_ms != null
                ? `${pipeline.process_ms.toFixed(2)} ms`
                : "—"}
            </dd>
          </div>
        </dl>
        {pipeline?.band.effective_high_hz === null && (
          <p className="mt-2 text-muted-foreground">
            Phone is capturing at {session?.sample_rate} Hz: only the low cut is
            applied.
          </p>
        )}
      </div>
      {pipeline?.test && (
        <SeparationTestPanel
          test={pipeline.test}
          connected={!!session}
          control={control}
        />
      )}
      <details className="border-t px-5 py-3 text-sm">
        <summary className="cursor-pointer font-semibold">
          Debug log (laptop server)
        </summary>
        <ol className="mt-2 max-h-72 overflow-y-auto font-mono text-xs">
          {[...(pipeline?.log ?? [])].reverse().map((line) => (
            <li
              key={line.seq}
              className={
                line.level === "error"
                  ? "text-destructive"
                  : line.level === "warn"
                    ? "text-amber-600"
                    : undefined
              }
            >
              {line.time} {line.level.toUpperCase()} [{line.source}]{" "}
              {line.message}
            </li>
          ))}
          {!pipeline?.log?.length && (
            <li className="text-muted-foreground">No messages yet.</li>
          )}
        </ol>
      </details>
      <p className="border-t px-5 py-3 text-sm text-muted-foreground">
        Use headphones to avoid feedback. Enable Separation test to try the two
        voices and lip-picked Focus. Falcon and captions are not enabled.
      </p>
      {(info?.warning || audioError || controlError || pipeline?.error) && (
        <p role="alert" className="border-t px-5 py-3 text-sm text-destructive">
          {audioError ?? controlError ?? pipeline?.error ?? info?.warning}
        </p>
      )}
    </section>
  );
}
