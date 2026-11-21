"use client";

import { useEffect, useState } from "react";
import { Camera, Radio } from "lucide-react";

const LINES = [
  { speaker: "Speaker 1", text: "Can everyone see the slides on their end?" },
  { speaker: "Speaker 2", text: "Yes, looks great. Let's start with the roadmap." },
  { speaker: "Speaker 1", text: "Perfect — first item is the camera rollout." },
  { speaker: "Speaker 3", text: "We connected the lobby feeds this morning." },
];

/** Decorative mock of a camera feed with a live, typed-out transcript. */
export function CameraPreview() {
  const [lineIdx, setLineIdx] = useState(0);
  const [chars, setChars] = useState(0);

  useEffect(() => {
    const current = LINES[lineIdx].text;
    if (chars < current.length) {
      const t = setTimeout(() => setChars((c) => c + 1), 38);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setLineIdx((i) => (i + 1) % LINES.length);
      setChars(0);
    }, 1600);
    return () => clearTimeout(t);
  }, [chars, lineIdx]);

  const history = [1, 2]
    .map((back) => LINES[(lineIdx - back + LINES.length) % LINES.length])
    .reverse();

  return (
    <div className="relative mx-auto w-full max-w-xl animate-float">
      {/* Glow */}
      <div className="absolute -inset-6 -z-10 rounded-[2rem] bg-primary/25 blur-3xl" />

      <div className="overflow-hidden rounded-2xl border border-primary/30 bg-[hsl(205_50%_6%)] text-slate-100 shadow-2xl shadow-primary/20">
        {/* Window chrome */}
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-2.5 text-xs">
          <div className="flex items-center gap-2 text-slate-300">
            <Camera className="h-3.5 w-3.5" />
            Conference Room A · CAM-02
          </div>
          <span className="flex items-center gap-1.5 rounded-full bg-red-500/15 px-2 py-0.5 font-medium text-red-400">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-500" />
            LIVE
          </span>
        </div>

        {/* "Video" */}
        <div className="relative aspect-video overflow-hidden bg-[radial-gradient(ellipse_at_center,hsl(200_40%_16%),hsl(205_50%_6%)_70%)]">
          <div className="bg-grid absolute inset-0 opacity-40 [--foreground:190_80%_80%]" />
          {/* scan line */}
          <div className="absolute inset-x-0 top-0 h-1/4 animate-scan bg-gradient-to-b from-transparent via-primary/15 to-transparent" />

          {/* Face silhouette */}
          <svg
            viewBox="0 0 320 180"
            className="absolute inset-0 h-full w-full"
            aria-hidden
          >
            <ellipse
              cx="160"
              cy="88"
              rx="52"
              ry="66"
              fill="hsl(200 30% 22% / 0.55)"
              stroke="hsl(190 74% 59% / 0.25)"
            />
            <circle cx="140" cy="72" r="3" fill="hsl(190 74% 70% / 0.6)" />
            <circle cx="180" cy="72" r="3" fill="hsl(190 74% 70% / 0.6)" />
            {/* tracking brackets around mouth */}
            <g
              stroke="hsl(190 74% 59%)"
              strokeWidth="2"
              fill="none"
              strokeLinecap="round"
            >
              <path d="M126 104 v-8 h8" />
              <path d="M194 104 v-8 h-8" />
              <path d="M126 128 v8 h8" />
              <path d="M194 128 v8 h-8" />
            </g>
          </svg>

          {/* Animated lips with landmark dots */}
          <div className="absolute left-1/2 top-[64.5%] -translate-x-1/2 -translate-y-1/2">
            <svg viewBox="0 0 60 24" className="h-6 w-[3.75rem]" aria-hidden>
              <g className="origin-center animate-lip-move [transform-box:fill-box]">
                <path
                  d="M4 12 C14 4 22 5 30 8 C38 5 46 4 56 12 C46 20 38 21 30 21 C22 21 14 20 4 12 Z"
                  fill="hsl(190 74% 59% / 0.18)"
                  stroke="hsl(190 74% 59%)"
                  strokeWidth="1.5"
                />
                {[
                  [4, 12],
                  [17, 6.5],
                  [30, 8],
                  [43, 6.5],
                  [56, 12],
                  [43, 19.5],
                  [30, 21],
                  [17, 19.5],
                ].map(([x, y]) => (
                  <circle key={`${x}-${y}`} cx={x} cy={y} r="1.4" fill="white" />
                ))}
              </g>
            </svg>
          </div>

          <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-md bg-black/40 px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-primary backdrop-blur">
            <Radio className="h-3 w-3" />
            Tracking · 98% confidence
          </div>
        </div>

        {/* Transcript */}
        <div className="space-y-2 border-t border-white/10 bg-black/30 px-4 py-3 font-mono text-[13px] leading-relaxed">
          {history.map((l, i) => (
            <p key={`${lineIdx}-${i}`} className="truncate text-slate-500">
              <span className="text-slate-400">{l.speaker}:</span> {l.text}
            </p>
          ))}
          <p className="min-h-[1.5em] text-slate-100">
            <span className="text-primary">{LINES[lineIdx].speaker}:</span>{" "}
            {LINES[lineIdx].text.slice(0, chars)}
            <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-blink bg-primary" />
          </p>
        </div>
      </div>
    </div>
  );
}
