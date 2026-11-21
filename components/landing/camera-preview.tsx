import { Camera, Laptop, Headphones } from "lucide-react";
export function CameraPreview() {
  return (
    <div className="rounded-2xl border bg-card p-8 shadow-lg">
      <p className="mb-6 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
        Local capture bridge
      </p>
      <div
        className="flex items-center justify-between gap-3"
        aria-hidden="true"
      >
        <Camera size={36} />
        <span>→</span>
        <Laptop size={36} />
        <span>→</span>
        <Headphones size={36} />
      </div>
      <h2 className="mt-8 text-xl font-semibold">
        Phone → laptop → headphones
      </h2>
      <p className="mt-3 leading-relaxed text-muted-foreground">
        Live camera frames and microphone audio travel over your local network.
        The dashboard shows mouth movement and lets you monitor the audio.
      </p>
      <p className="mt-5 rounded-lg bg-muted p-3 text-sm">
        Current milestone: raw passthrough. Voice isolation and captions are
        next.
      </p>
    </div>
  );
}
