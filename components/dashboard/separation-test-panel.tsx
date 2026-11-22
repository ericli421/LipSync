"use client";
import { Button } from "@/components/ui/button";

export type TestLive = {
  window_s: number;
  hop_s: number;
  delay_ms: number;
  infer_ms: number | null;
  infer_p95_ms: number | null;
  rtf: number | null;
  keeping_up: boolean | null;
  hops: number;
  skipped_hops: number;
  swaps_fixed: number;
  underruns: number;
  behind_ms: number;
  voice_db: [number, number];
  pick: number;
  pick_reason: string;
  lip_scores: [number | null, number | null];
  recorded_s: number;
  error: string | null;
  identity?: {
    identity_swaps: number;
    decision?: string;
    voiceprints: [boolean, boolean];
    ms: number | null;
  } | null;
};
export type TestStatus = {
  enabled: boolean;
  state: "off" | "loading" | "ready" | "running" | "error";
  error: string | null;
  model: string;
  info: {
    model: string;
    label: string;
    device: string;
    gpu: string | null;
  } | null;
  window_s: number;
  hop_s: number;
  monitor: "original" | "voice1" | "voice2" | "focus";
  focus_db: number;
  duck_db: number;
  focus_style?: "mask" | "mix";
  lip_offset_ms: number;
  recording: boolean;
  last_recording: string | null;
  identity?: boolean;
  identity_ready?: boolean;
  live: TestLive | null;
};

const MODELS = [
  { value: "auto", label: "Auto (GPU: WHAMR 16k, CPU: ReSepFormer)" },
  {
    value: "sepformer-whamr16k",
    label: "SepFormer-WHAMR 16k · noise + reverb, 16 kHz",
  },
  { value: "sepformer-whamr", label: "SepFormer-WHAMR 8k · noise + reverb" },
  {
    value: "resepformer-wsj02mix",
    label: "ReSepFormer 8k · fast, clean-trained",
  },
];
const MONITORS = [
  { value: "original", label: "Original (live)" },
  { value: "voice1", label: "Voice 1" },
  { value: "voice2", label: "Voice 2" },
  { value: "focus", label: "Focus (person on camera)" },
] as const;

const meter = (db: number) => Math.min(1, Math.max(0, (db + 60) / 60));
const ms = (value: number | null) =>
  value === null ? "—" : `${Math.round(value)} ms`;

function stateText(test: TestStatus, connected: boolean) {
  if (test.state === "loading")
    return "Loading model… (first run downloads it)";
  if (test.state === "error") return test.error ?? "Error";
  if (test.state === "off") return "Off";
  const where = test.info
    ? `${test.info.model} on ${test.info.gpu ?? test.info.device}`
    : "";
  if (test.state === "ready")
    return connected
      ? `Ready · ${where} · starting with the next audio block`
      : `Ready · ${where} · start capture on the phone`;
  return `Running · ${where}`;
}

export function SeparationTestPanel({
  test,
  connected,
  control,
}: {
  test: TestStatus;
  connected: boolean;
  control: (message: object) => void;
}) {
  const live = test.live;
  const configure = (changes: object) =>
    control({
      type: "set_test_mode",
      enabled: test.enabled,
      model: test.model,
      window_s: test.window_s,
      hop_s: test.hop_s,
      identity: test.identity ?? false,
      ...changes,
    });
  const select = "rounded-md border bg-background px-2 py-1";
  return (
    <div className="border-t p-5 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-semibold">Separation test</h3>
          <p
            className={
              test.state === "error"
                ? "mt-1 text-destructive"
                : "mt-1 text-muted-foreground"
            }
          >
            {stateText(test, connected)}
          </p>
        </div>
        <Button
          variant={test.enabled ? "default" : "outline"}
          aria-pressed={test.enabled}
          onClick={() => configure({ enabled: !test.enabled })}
        >
          Test mode · {test.enabled ? "On" : "Off"}
        </Button>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Button
          size="sm"
          variant={(test.identity ?? false) ? "default" : "outline"}
          aria-pressed={test.identity ?? true}
          onClick={() => configure({ identity: !(test.identity ?? false) })}
        >
          Voice lock · {(test.identity ?? false) ? "On" : "Off"}
        </Button>
        <span className="text-muted-foreground">
          {!(test.identity ?? false)
            ? "Voices can swap after a pause."
            : test.identity_ready
              ? `Speaker voiceprints keep each person in one voice${
                  live?.identity
                    ? ` · ${live.identity.identity_swaps} corrections · ${live.identity.decision ?? ""}`
                    : ""
                }`
              : "Needs the GPU; see the debug log."}
        </span>
      </div>

      <div className="mt-3 flex flex-wrap gap-3">
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground">Model</span>
          <select
            className={select}
            value={test.model}
            onChange={(e) => configure({ model: e.target.value })}
          >
            {MODELS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground">Window</span>
          <select
            className={select}
            value={test.window_s}
            onChange={(e) => configure({ window_s: Number(e.target.value) })}
          >
            {[1, 1.5, 2].map((v) => (
              <option key={v} value={v}>
                {v} s
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground">Hop</span>
          <select
            className={select}
            value={test.hop_s}
            onChange={(e) => configure({ hop_s: Number(e.target.value) })}
          >
            {[0.25, 0.5].map((v) => (
              <option key={v} value={v}>
                {v} s
              </option>
            ))}
          </select>
        </label>
      </div>

      <fieldset className="mt-4">
        <legend className="text-muted-foreground">
          Listen (phone headphones and this page)
        </legend>
        <div className="mt-2 flex flex-wrap gap-2">
          {MONITORS.map((m) => (
            <Button
              key={m.value}
              size="sm"
              variant={test.monitor === m.value ? "default" : "outline"}
              aria-pressed={test.monitor === m.value}
              onClick={() => control({ type: "set_monitor", source: m.value })}
            >
              {m.label}
            </Button>
          ))}
        </div>
      </fieldset>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        {[0, 1].map((i) => (
          <div
            key={i}
            className={
              live?.pick === i
                ? "rounded-lg border-2 border-primary p-3"
                : "rounded-lg border p-3"
            }
          >
            <div className="flex justify-between">
              <span className="font-medium">
                Voice {i + 1}
                {live?.pick === i && " · on camera"}
              </span>
              <span className="font-mono">
                {live ? `${live.voice_db[i].toFixed(1)} dB` : "—"}
              </span>
            </div>
            <meter
              aria-label={`Voice ${i + 1} level`}
              value={live ? meter(live.voice_db[i]) : 0}
              min={0}
              max={1}
              className="mt-2 block w-full"
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Lip match:{" "}
              {live?.lip_scores[i] != null
                ? live.lip_scores[i]!.toFixed(2)
                : "—"}
            </p>
          </div>
        ))}
      </div>
      <p className="mt-2 text-muted-foreground">
        Lip pick: {live ? live.pick_reason : "—"}
        {live && live.voice_db
          ? ` · level gap ${Math.abs(live.voice_db[0] - live.voice_db[1]).toFixed(1)} dB`
          : ""}
      </p>

      <dl className="mt-4 grid gap-3 font-mono sm:grid-cols-4">
        <div>
          <dt className="font-sans text-muted-foreground">
            Model time / budget
          </dt>
          <dd>
            {ms(live?.infer_ms ?? null)} / {live ? ms(live.hop_s * 1000) : "—"}
          </dd>
        </div>
        <div>
          <dt className="font-sans text-muted-foreground">Keeping up</dt>
          <dd
            className={
              live?.keeping_up === false ? "text-destructive" : undefined
            }
          >
            {live?.keeping_up == null
              ? "—"
              : live.keeping_up
                ? `Yes (p95 ${ms(live.infer_p95_ms)})`
                : `No · ${live.skipped_hops} skipped`}
          </dd>
        </div>
        <div>
          <dt className="font-sans text-muted-foreground">Audio delay</dt>
          <dd>{live ? `${live.delay_ms} ms + Wi-Fi` : "—"}</dd>
        </div>
        <div>
          <dt className="font-sans text-muted-foreground">Order swaps fixed</dt>
          <dd>{live ? `${live.swaps_fixed} / ${live.hops} chunks` : "—"}</dd>
        </div>
      </dl>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground">Focus sound</span>
        {(
          [
            ["mask", "Natural (lower the others)"],
            ["mix", "Mix (separated voice)"],
          ] as const
        ).map(([style, label]) => (
          <Button
            key={style}
            size="sm"
            variant={
              (test.focus_style ?? "mask") === style ? "default" : "outline"
            }
            aria-pressed={(test.focus_style ?? "mask") === style}
            onClick={() => control({ type: "set_focus", style })}
          >
            {label}
          </Button>
        ))}
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <label className="block">
          <span className="text-muted-foreground">
            Boost on-camera voice: +{test.focus_db} dB
          </span>
          <input
            type="range"
            min={0}
            max={12}
            step={1}
            value={test.focus_db}
            onChange={(e) =>
              control({ type: "set_focus", focus_db: Number(e.target.value) })
            }
            className="mt-1 block w-full"
          />
        </label>
        <label className="block">
          <span className="text-muted-foreground">
            Lower the rest: {test.duck_db} dB
          </span>
          <input
            type="range"
            min={-40}
            max={0}
            step={1}
            value={test.duck_db}
            onChange={(e) =>
              control({ type: "set_focus", duck_db: Number(e.target.value) })
            }
            className="mt-1 block w-full"
          />
        </label>
        <label className="block">
          <span className="text-muted-foreground">
            Lip/audio offset: {test.lip_offset_ms} ms
          </span>
          <input
            type="range"
            min={-500}
            max={500}
            step={20}
            value={test.lip_offset_ms}
            onChange={(e) =>
              control({
                type: "set_focus",
                lip_offset_ms: Number(e.target.value),
              })
            }
            className="mt-1 block w-full"
          />
        </label>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button
          variant={test.recording ? "destructive" : "outline"}
          disabled={!test.enabled}
          onClick={() =>
            control({ type: "set_recording", enabled: !test.recording })
          }
        >
          {test.recording
            ? `Stop recording (${live?.recorded_s ?? 0} s)`
            : "Record test clip"}
        </Button>
        {test.last_recording && (
          <span className="text-muted-foreground">
            Saved: <span className="font-mono">{test.last_recording}</span>
          </span>
        )}
      </div>
      <p className="mt-3 text-muted-foreground">
        How to test: take turns talking. Each person should light up only one
        voice meter (a level gap of 10 dB or more is a clear split). Then talk
        at the same time and listen to Voice 1 and Voice 2. Point the camera at
        one person and choose Focus: their voice should be louder and the other
        voice and noise quieter. Separated audio is 8 kHz (telephone quality)
        and arrives with the delay shown.
      </p>
      {live?.error && (
        <p role="alert" className="mt-2 text-destructive">
          {live.error}
        </p>
      )}
    </div>
  );
}
