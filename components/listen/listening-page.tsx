"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import Link from "next/link";
import { ChevronDown, Play, RefreshCw, Volume2, VolumeX } from "lucide-react";
import { BufferedPcmPlayer, decodePcm16 } from "@/lib/buffered-pcm-player";
import { listenUrl } from "@/lib/processor-url";
import styles from "./listening-page.module.css";

type Session = { session_id: string; sample_rate: number };
type Summary = { source?: string; source_label: string; source_delay_ms: number; separation: boolean; state: string; layers: { voice_band?: boolean; auto_volume?: boolean } };
type Playback = { active: boolean; file: string | null; position_s: number | null; at: number };
const sourceNames: Record<string, string> = { original: "Original", focus: "Focus", voice1: "Voice 1", voice2: "Voice 2" };

export function ListeningPage() {
  const [connection, setConnection] = useState("Connecting to the presentation…");
  const [session, setSession] = useState<Session | null>(null);
  const [hasFrame, setHasFrame] = useState(false);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [listening, setListening] = useState(false);
  const [volume, setVolume] = useState(0.75);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [audioStats, setAudioStats] = useState({ buffering: true, targetMs: 200, underruns: 0 });
  const canvas = useRef<HTMLCanvasElement>(null);
  const currentSession = useRef<Session | null>(null);
  const player = useRef<BufferedPcmPlayer | null>(null);
  const generation = useRef(0);
  // Prerecorded fallback: the presenter can switch every listener to a saved recording.
  const [playback, setPlayback] = useState<Playback>({ active: false, file: null, position_s: null, at: 0 });
  const video = useRef<HTMLVideoElement>(null);
  // Whether this judge wants the recording's sound. Kept across phone disconnects
  // (when the presenter pivots to the recording); only the Listen/Mute button changes it.
  const [videoSound, setVideoSound] = useState(false);
  const wantsSound = useRef(false);
  const [soundBlocked, setSoundBlocked] = useState(false);

  function stopAudio() {
    generation.current++;
    player.current?.close();
    player.current = null;
    setListening(false);
  }

  useEffect(() => {
    let disposed = false;
    let activeSocket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let retryDelay = 1000;
    let drawing = false;

    const clear = () => {
      currentSession.current = null;
      setSession(null);
      setHasFrame(false);
      setSummary(null);
      stopAudio();
      const target = canvas.current;
      if (target) target.getContext("2d")?.clearRect(0, 0, target.width, target.height);
    };
    const connect = () => {
      if (disposed) return;
      const socket = new WebSocket(listenUrl());
      activeSocket = socket;
      socket.onopen = () => {
        retryDelay = 1000;
        setConnection("Waiting for the presenter’s phone…");
      };
      socket.onmessage = async event => {
        if (disposed || activeSocket !== socket) return;
        try {
          const message = JSON.parse(event.data);
          if (message.type === "session") {
            clear();
            currentSession.current = message;
            setSession(message);
            setConnection("Live presentation connected");
            setError(null);
          } else if (message.type === "waiting") {
            clear();
            setConnection("Waiting for the presenter’s phone…");
          } else if (message.type === "pipeline") {
            if (message.playback) {
              const next = message.playback;
              setPlayback(previous => previous.active === next.active && previous.file === next.file
                ? previous : { active: next.active, file: next.file, position_s: next.position_s, at: performance.now() });
            }
            if (!message.session_id || message.session_id === currentSession.current?.session_id) setSummary(message);
          } else if (message.session_id !== currentSession.current?.session_id) {
            return;
          } else if (message.type === "discontinuity") {
            player.current?.reset();
          } else if (message.type === "audio" && player.current) {
            player.current.push(decodePcm16(message.pcm_base64), currentSession.current!.sample_rate, message.seq, message.session_id);
          } else if (message.type === "video" && !drawing) {
            drawing = true;
            try {
              const bytes = Uint8Array.from(atob(message.jpeg_base64), char => char.charCodeAt(0));
              const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }));
              const target = canvas.current;
              if (!disposed && activeSocket === socket && target && message.session_id === currentSession.current?.session_id) {
                target.width = bitmap.width;
                target.height = bitmap.height;
                target.getContext("2d")?.drawImage(bitmap, 0, 0);
                setHasFrame(true);
              }
              bitmap.close();
            } finally { drawing = false; }
          }
        } catch (problem) {
          setError(problem instanceof Error ? problem.message : "Could not read the live feed");
          stopAudio();
        }
      };
      socket.onerror = () => setConnection("Cannot reach the presentation. Checking the connection…");
      socket.onclose = () => {
        if (disposed || activeSocket !== socket) return;
        activeSocket = null;
        clear();
        setConnection("Presentation disconnected. Reconnecting automatically…");
        retry = setTimeout(connect, retryDelay);
        retryDelay = Math.min(8000, retryDelay * 2);
      };
    };
    connect();
    const timer = setInterval(() => {
      if (player.current) setAudioStats(player.current.stats);
    }, 500);
    const visibility = () => {
      if (document.hidden) stopAudio();
    };
    document.addEventListener("visibilitychange", visibility);
    return () => {
      disposed = true;
      clearTimeout(retry);
      clearInterval(timer);
      activeSocket?.close();
      currentSession.current = null;
      stopAudio();
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [attempt]);

  // Start the recording where everyone else is, with sound only after Listen.
  useEffect(() => {
    const element = video.current;
    if (!element) return;
    if (!playback.active || !playback.file) { element.pause(); return; }
    const sync = () => {
      element.currentTime = (playback.position_s ?? 0) + (performance.now() - playback.at) / 1000;
      element.muted = !wantsSound.current;
      element.play().then(() => setSoundBlocked(false)).catch(() => {
        // The browser refused sound without a click on this page: play silently and ask for one.
        element.muted = true;
        setSoundBlocked(true);
        void element.play().catch(() => undefined);
      });
    };
    if (element.readyState >= 1) sync();
    else element.addEventListener("loadedmetadata", sync, { once: true });
    return () => element.removeEventListener("loadedmetadata", sync);
  }, [playback]);
  useEffect(() => {
    if (video.current) {
      video.current.muted = !videoSound || !playback.active;
      video.current.volume = volume;
    }
    player.current?.setVolume(playback.active ? 0 : volume);  // the live feed stays silent meanwhile
  }, [videoSound, playback.active, volume]);

  // Listen/Mute while the recording plays: runs inside the click, so browsers allow sound.
  function toggleVideoSound(force?: boolean) {
    const element = video.current;
    const next = force ?? (!videoSound || soundBlocked);
    wantsSound.current = next;
    setVideoSound(next);
    if (!element) return;
    element.muted = !next;
    element.volume = volume;
    if (next) element.play().then(() => setSoundBlocked(false)).catch(() => setSoundBlocked(true));
  }

  async function toggleListening() {
    if (playback.active) { toggleVideoSound(); return; }
    if (player.current || listening) { wantsSound.current = false; setVideoSound(false); stopAudio(); return; }
    const selected = currentSession.current;
    if (!selected) return;
    wantsSound.current = true;  // keep sound if the presenter later switches to the recording
    setVideoSound(true);
    const epoch = ++generation.current;
    setError(null);
    let next: BufferedPcmPlayer | null = null;
    try {
      next = new BufferedPcmPlayer();
      await next.resume();
      if (epoch !== generation.current || selected !== currentSession.current) { next.close(); return; }
      next.setSession(selected.session_id);
      next.setVolume(volume);
      player.current = next;
      setAudioStats(next.stats);
      setListening(true);
    } catch (problem) {
      next?.close();
      setError(problem instanceof Error ? problem.message : "Audio is unavailable in this browser. Try Chrome, Edge, Firefox or Safari.");
    }
  }

  const hearing = playback.active ? videoSound && !soundBlocked : listening;
  const connectionLabel = playback.active ? "Recorded demo" : session ? "Live session"
    : connection.startsWith("Waiting") ? "Waiting for presenter"
    : connection.startsWith("Connecting") ? "Connecting" : "Reconnecting";
  const sourceName = session && summary ? sourceNames[summary.source ?? ""] ?? summary.source_label : "—";

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link href="/" className={styles.wordmark} aria-label="LipSync home">lipsync</Link>
        <p className={styles.connection} role="status" aria-label={connection}>
          <span>{connectionLabel}</span><span className={styles.dot} data-live={!!session || playback.active} aria-hidden="true" />
        </p>
      </header>
      <main className={styles.main}>
        <h1 className="sr-only">Live listening</h1>
        <section className={styles.video} aria-label="Live camera preview">
          <canvas ref={canvas} aria-label="Presenter camera with selected person highlighted" hidden={playback.active} />
          <video ref={video} hidden={!playback.active} playsInline preload="auto" aria-label="Prerecorded LipSync demo"
            onClick={() => toggleVideoSound(true)}
            src={playback.active && playback.file ? `/live/media/${playback.file.split("/").map(encodeURIComponent).join("/")}` : undefined}
 />
          {playback.active && (!videoSound || soundBlocked) && (
            <button type="button" className={styles.soundPrompt} onClick={() => toggleVideoSound(true)}>
              <Volume2 size={18} aria-hidden="true" /> Click for sound
            </button>
          )}
          {!hasFrame && !playback.active && <div className={styles.empty}>
            <p>{session ? "Waiting for the camera" : "Waiting for the presenter"}</p>
            <span>{session ? "The live picture will appear here." : "The live picture will appear when their phone connects."}</span>
          </div>}
        </section>
        <div className={styles.controls} role="group" aria-label="Personal listening controls">
          <div className={styles.playback}>
            <button type="button" className={styles.listen} disabled={!session && !playback.active} onClick={toggleListening} aria-pressed={hearing}>
              <span className={styles.playIcon} aria-hidden="true">{hearing ? <VolumeX size={17} /> : <Play size={16} fill="currentColor" />}</span>
              <span>{hearing ? "Mute" : "Listen"}</span>
            </button>
            <label className={styles.volume}>
              {volume === 0 ? <VolumeX size={18} aria-hidden="true" /> : <Volume2 size={18} aria-hidden="true" />}
              <span className="sr-only">Your volume</span>
              <input type="range" min="0" max="1" step="0.01" value={volume} aria-valuetext={`${Math.round(volume * 100)}%`}
                style={{ "--volume": `${volume * 100}%` } as CSSProperties}
                onChange={event => { const value = Number(event.target.value); setVolume(value); player.current?.setVolume(value); }} />
            </label>
          </div>
          <div className={styles.output}>
            <p className={styles.source} aria-live="polite"><span>Hearing</span><span>{sourceName}</span></p>
            <button type="button" className={styles.reconnect} aria-label="Reconnect" onClick={() => setAttempt(value => value + 1)}><RefreshCw size={20} aria-hidden="true" /></button>
          </div>
        </div>
        {listening && audioStats.buffering && <p className={styles.notice} role="status">Buffering audio…</p>}
        {error && <p className={styles.error} role="alert">{error}</p>}
        <footer className={styles.footer}>
          <p>Use headphones for the live demo.</p>
          <details className={styles.details}>
            <summary>Details <ChevronDown size={12} aria-hidden="true" /></summary>
            <div className={styles.detailContent}>
              <p>{connection}</p>
              <p>{summary?.source_label ?? "Waiting for presentation"}{summary ? ` · Separation ${summary.separation ? "enabled" : "off"}` : ""}{summary?.layers.voice_band ? " · Speech-band filter" : ""}{summary?.layers.auto_volume ? " · Auto volume" : ""}</p>
              {listening && <p role="status">{audioStats.buffering ? "Buffering audio…" : "Listening to the presenter’s selected output."} Listening buffer: {audioStats.targetMs} ms. Processing adds approximately {summary?.source_delay_ms ?? 0} ms, plus network and device delay.</p>}
              <p>Your volume and mute affect only this device. The presenter controls the audio.</p>
              <p>The camera is a live preview; processed audio arrives later. After changing tabs or reconnecting the phone, click Listen again.</p>
              {!session && <p>Check that you and the presenter are on the same network and that local device connections are allowed.</p>}
            </div>
          </details>
        </footer>
      </main>
    </div>
  );
}
