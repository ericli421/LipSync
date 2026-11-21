"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ChevronDown,
  Loader2,
  Power,
  PowerOff,
  RefreshCw,
  ScanFace,
  Video,
  VideoOff,
  Webcam,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { PROCESSOR_URL, useFrameProcessor } from "./use-frame-processor";

type Status =
  | "loading"
  | "needs-permission"
  | "ready"
  | "unsupported"
  | "error";

const cameraLabel = (d: MediaDeviceInfo, i: number) =>
  d.label || `Camera ${i + 1}`;

export function CameraDashboard() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [listError, setListError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [on, setOn] = useState(false);
  const [starting, setStarting] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);

  // Send frames through the Python processor (processor/server.py) before display.
  const [processing, setProcessing] = useState(true);
  const [procAttempt, setProcAttempt] = useState(0);

  const refresh = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setStatus("unsupported");
      return;
    }
    try {
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter(
        (d) => d.kind === "videoinput",
      );
      // Without permission, browsers hide device ids/labels.
      const hidden =
        cams.length > 0 && cams.every((d) => !d.deviceId || !d.label);
      setDevices(cams);
      setStatus(hidden ? "needs-permission" : "ready");
      // Keep the current selection if it still exists, otherwise pick the first.
      setSelectedId((cur) =>
        cur && cams.some((d) => d.deviceId === cur)
          ? cur
          : (cams[0]?.deviceId ?? null),
      );
    } catch (e: unknown) {
      setListError(e instanceof Error ? e.message : "Unable to list cameras");
      setStatus("error");
    }
  }, []);

  const requestAccess = async () => {
    setStatus("loading");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach((t) => t.stop());
      await refresh();
    } catch (e: unknown) {
      setListError(
        e instanceof Error ? e.message : "Camera permission was denied",
      );
      setStatus("error");
    }
  };

  useEffect(() => {
    refresh();
    const md = navigator.mediaDevices;
    md?.addEventListener?.("devicechange", refresh);
    return () => md?.removeEventListener?.("devicechange", refresh);
  }, [refresh]);

  // Start/stop the stream whenever the toggle or the selected camera changes.
  useEffect(() => {
    if (!on || !selectedId) return;
    const video = videoRef.current;
    let stream: MediaStream | null = null;
    let cancelled = false;
    setStarting(true);
    setStreamError(null);

    navigator.mediaDevices
      .getUserMedia({ video: { deviceId: { exact: selectedId } } })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        if (video) video.srcObject = s;
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setStreamError(
          e instanceof Error ? e.message : "Unable to open camera",
        );
        setOn(false);
      })
      .finally(() => !cancelled && setStarting(false));

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
      if (video) video.srcObject = null;
    };
  }, [on, selectedId]);

  const selectCamera = (id: string) => {
    setStreamError(null);
    setSelectedId(id);
  };

  const selectedIndex = devices.findIndex((d) => d.deviceId === selectedId);
  const selected = selectedIndex >= 0 ? devices[selectedIndex] : null;
  const live = on && !starting && !streamError;
  const hasCameras = status === "ready" && devices.length > 0;

  const proc = useFrameProcessor(
    videoRef,
    canvasRef,
    live && processing,
    procAttempt,
  );
  const showProcessed = live && processing && proc.state === "streaming";

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Camera</h1>
          <p className="mt-1 text-muted-foreground">
            Choose a camera from the list and turn it on to begin.
          </p>
        </div>
        <Button variant="outline" onClick={refresh}>
          <RefreshCw />
          Refresh
        </Button>
      </div>

      {status === "loading" && (
        <div className="flex items-center justify-center gap-2 py-24 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Looking for cameras…
        </div>
      )}

      {hasCameras && (
        <section
          className={cn(
            "mx-auto flex w-full max-w-4xl flex-col overflow-hidden rounded-xl border bg-card transition-shadow",
            live && "border-primary/50 shadow-lg shadow-primary/15",
          )}
        >
          {/* Controls */}
          <div className="flex flex-col gap-3 border-b p-4 sm:flex-row sm:items-center sm:justify-between">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-between sm:w-80"
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <Webcam className="text-primary-deep" />
                    <span className="truncate">
                      {selected
                        ? cameraLabel(selected, selectedIndex)
                        : "Select a camera"}
                    </span>
                  </span>
                  <ChevronDown className="opacity-60" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="start"
                className="w-[var(--radix-dropdown-menu-trigger-width)]"
              >
                <DropdownMenuLabel>
                  Available cameras ({devices.length})
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuRadioGroup
                  value={selectedId ?? undefined}
                  onValueChange={selectCamera}
                >
                  {devices.map((d, i) => (
                    <DropdownMenuRadioItem
                      key={d.deviceId || i}
                      value={d.deviceId}
                    >
                      <span className="truncate">{cameraLabel(d, i)}</span>
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>

            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => setProcessing((v) => !v)}
                aria-pressed={processing}
                className={cn(
                  "flex-1 sm:flex-none",
                  processing && "border-primary/60 bg-primary/10",
                )}
              >
                <ScanFace className={cn(processing && "text-primary-deep")} />
                Processing {processing ? "on" : "off"}
              </Button>
              <Button
                variant={on ? "outline" : "default"}
                onClick={() => setOn((v) => !v)}
                aria-pressed={on}
                disabled={!selectedId}
                className="flex-1 sm:flex-none"
              >
                {on ? <PowerOff /> : <Power />}
                {on ? "Turn off" : "Turn on"}
              </Button>
            </div>
          </div>

          {/* Feed */}
          <div className="relative aspect-video bg-[hsl(205_50%_6%)]">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={cn(
                "h-full w-full object-cover",
                (!on || showProcessed) && "hidden",
              )}
            />
            {/* Annotated frames returned by the Python processor */}
            <canvas
              ref={canvasRef}
              className={cn(
                "h-full w-full object-contain",
                !showProcessed && "hidden",
              )}
            />
            {live && processing && proc.state === "connecting" && (
              <div className="absolute inset-x-0 bottom-0 flex items-center justify-center gap-2 bg-black/60 px-4 py-2 text-xs text-slate-200 backdrop-blur">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Connecting to processor…
              </div>
            )}
            {live && processing && proc.state === "error" && (
              <div className="absolute inset-x-0 bottom-0 flex flex-col items-center justify-between gap-2 bg-black/70 px-4 py-2.5 text-xs text-slate-200 backdrop-blur sm:flex-row">
                <span className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 shrink-0 text-amber-400" />
                  <span>
                    Can&apos;t reach the processor at{" "}
                    <code className="text-primary">{PROCESSOR_URL}</code> —
                    showing the raw feed. Start it with{" "}
                    <code className="text-primary">
                      uvicorn processor.server:app --port 8000
                    </code>
                  </span>
                </span>
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-7 shrink-0"
                  onClick={() => setProcAttempt((n) => n + 1)}
                >
                  Retry
                </Button>
              </div>
            )}
            {!on && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-6 text-center text-slate-400">
                {streamError ? (
                  <>
                    <AlertCircle className="h-10 w-10 text-red-400" />
                    <p className="text-sm text-slate-300">
                      Couldn&apos;t start camera
                    </p>
                    <p className="text-xs">{streamError}</p>
                  </>
                ) : (
                  <>
                    <VideoOff className="h-10 w-10" />
                    <p className="text-sm">Camera is off</p>
                  </>
                )}
              </div>
            )}
            {on && starting && (
              <div className="absolute inset-0 flex items-center justify-center text-slate-300">
                <Loader2 className="h-7 w-7 animate-spin" />
              </div>
            )}
            {live && (
              <span className="absolute left-3 top-3 flex items-center gap-1.5 rounded-md bg-black/50 px-2 py-1 text-xs font-medium text-white backdrop-blur">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-500" />
                LIVE
                {showProcessed && (
                  <span className="text-primary">· Processed</span>
                )}
              </span>
            )}
          </div>

          {/* Results from the Python processor */}
          {showProcessed && (
            <div className="grid grid-cols-3 gap-4 border-t p-4 text-sm">
              <div>
                <p className="text-xs text-muted-foreground">Face</p>
                <p className="font-medium">
                  {proc.info?.face_found ? "Detected" : "Not found"}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Mouth openness</p>
                <div className="mt-1.5 flex items-center gap-2">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary transition-[width] duration-100"
                      style={{
                        width: `${Math.min(100, (proc.info?.openness ?? 0) * 200)}%`,
                      }}
                    />
                  </div>
                  <span className="w-9 text-right font-mono text-xs">
                    {(proc.info?.openness ?? 0).toFixed(2)}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground">Processed FPS</p>
                <p className="font-mono font-medium">{proc.fps}</p>
              </div>
            </div>
          )}
        </section>
      )}

      {status !== "loading" && !hasCameras && (
        <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-dashed bg-card px-6 py-20 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/15 text-primary-deep">
            {status === "error" || status === "unsupported" ? (
              <AlertCircle className="h-7 w-7" />
            ) : (
              <Video className="h-7 w-7" />
            )}
          </div>
          {status === "needs-permission" && (
            <>
              <div>
                <p className="text-lg font-semibold">Allow camera access</p>
                <p className="mt-1 max-w-md text-sm text-muted-foreground">
                  LipSync needs permission to see the cameras connected to this
                  device.
                </p>
              </div>
              <Button onClick={requestAccess}>Allow access</Button>
            </>
          )}
          {status === "ready" && (
            <div>
              <p className="text-lg font-semibold">No cameras found</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Plug in a camera and it will appear here automatically.
              </p>
            </div>
          )}
          {status === "unsupported" && (
            <div>
              <p className="text-lg font-semibold">Camera access unavailable</p>
              <p className="mt-1 max-w-md text-sm text-muted-foreground">
                Your browser doesn&apos;t support camera access here. Open the
                app over https or on localhost.
              </p>
            </div>
          )}
          {status === "error" && (
            <>
              <div>
                <p className="text-lg font-semibold">
                  Couldn&apos;t access cameras
                </p>
                <p className="mt-1 max-w-md text-sm text-muted-foreground">
                  {listError}
                </p>
              </div>
              <Button onClick={requestAccess}>Try again</Button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
