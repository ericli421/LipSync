# Falcon / SepFormer comparison handoff

This offline experiment asks whether Falcon improves the target voice when given
the original mixture or either anonymous SepFormer candidate. It does not change
the live app, choose a speaker automatically, or establish live latency.

## Environment

Follow [the Falcon setup](model-validation.md) for the separate Python 3.11
environment, explicit PyTorch CPU/CUDA installation, and pinned `external/Falcon`
checkout. Then install the comparison dependencies:

```powershell
.\.venv-model\Scripts\python.exe -m pip install -r requirements-falcon-comparison.txt
```

FFmpeg and ffprobe must be on PATH. Use the separate audio environment described
in [audio-only separation](audio-only-separation.md) for SepFormer; the comparison
process never imports or loads SepFormer. Nothing is added to normal app startup.

## Prepare the exact input for your teammate

Record 20–30 seconds with a stationary phone and two unobstructed mouths: A alone,
B alone, both speaking independently, and a pause where the target stops while
the other person continues. Choose a fixed square mouth crop in upright decoded
pixels. The coordinates below are examples; replace them with your measured crop.

```powershell
.\.venv-model\Scripts\python.exe scripts/compare_falcon.py `
  --video recordings/two-speakers.mp4 --mouth-box 200 140 80 `
  --prepare-only --out-dir validation-output/falcon-input-a
```

Inspect `mouth_preview.mp4` first. If the mouth leaves the crop, record again.
Preparation exports float32 mono **8 kHz `input.wav`** on the common retained
audio/video timeline. `report.json` records the video hash, common start/end,
mouth box, frame count, and input hash. Preparation needs neither PyTorch nor
weights. It resamples with SciPy's polyphase band-limited filter.

Have SepFormer separate this exact `input.wav`. With the current offline runner:

```powershell
.\.venv-audio\Scripts\python.exe scripts/separate_audio.py `
  --input validation-output/falcon-input-a/input.wav `
  --device cuda --out-dir validation-output/sepformer-a
```

Your teammate can instead export from his upcoming implementation. The testing
handoff is a directory containing:

| File | Contract |
| --- | --- |
| `mixture.wav` | Prepared mixture, mono PCM16, attenuated by the shared export gain |
| `candidate_0.wav`, `candidate_1.wav` | Mono PCM16, same rate/count/origin as mixture; no independent normalization or unreported delay |
| `metrics.json` | Top-level integer `sample_rate`, `export.shared_gain` in `(0,1]`, plus available model pins/device/timing provenance |

This does not prescribe his live API. Rates from 8–48 kHz are supported by the
reader, but the default prepared input is 8 kHz, matching the current separator.
For a different rate, resample the retained native source with the same polyphase
method. Candidate indices are not identities and must not swap within an exported
file. The exporter must compensate known processing delays before exporting.
Do not use a full-video soundtrack with different start/end bounds.

## Run all four Falcon branches

```powershell
.\.venv-model\Scripts\python.exe scripts/compare_falcon.py `
  --video recordings/two-speakers.mp4 --mouth-box 200 140 80 `
  --separation-dir validation-output/sepformer-a `
  --falcon-dir external/Falcon --device cuda `
  --out-dir validation-output/falcon-compare-a
```

Use the same video and crop as preparation. CPU must be explicitly requested
with `--device cpu`; it never passes the CUDA throughput gate. Choose a fresh
output directory. Repeat with B's crop to test the other target; the same
separation can be reused because changing the crop does not change the audio.

The harness verifies the mixture samplewise against prepared audio at the
declared rate, after applying the export gain (absolute tolerance two PCM16 LSBs).
Wrong recordings, rates, lengths, gains, and offsets are rejected. Very quiet
exports that cannot establish source agreement are rejected. Samplewise mixture
agreement does not prove candidate alignment: that remains the exporter's contract.
The shared attenuation is undone before candidate inference.

Falcon loads once. Independent slicer instances process the original mixture,
the original down/up-sampled through the separator rate, candidate 0, and candidate
1 with the **same mouth crops**. The round-trip control exposes the bandwidth
cost of an 8 kHz separator; upsampling cannot recover missing frequencies.

## Outputs and decisions

Eight aligned analysis WAVs contain those four unrefined inputs and their four
`*_falcon.wav` results. All use one shared clipping-protection gain recorded in
`report.json`. If the upstream slicer clips any output, the comparison aborts
rather than publishing distorted evidence; reduce source recording levels and
repeat preparation/separation. The existing standalone validation CLI retains its
original clipping/counting behaviour.

`listening_previews/` contains separately labelled two-pass EBU R128 previews
targeting −23 LUFS and −2 dBTP. Their measurements, actual output loudness, and
normalization type are reported. Silence/unmeasurable tracks are copied and
explicitly marked, never described as loudness-matched. Analysis WAVs are not
independently normalized. Review words, interference, distortion, and target
silence in addition to volume. The mouth preview covers the entire prepared
input; use the report's source interval when comparing it to the trimmed WAVs.

Only complete two-second windows are processed at 500 ms hops, with upstream
80 ms right context and crossfading. Outputs start 1.42 seconds after the common
origin; a 20-second prepared input yields 37 windows and 18.5 seconds of aligned
output. These positions are source timing, **not measured live listening delay**.

The report includes input/output hashes, pins, actual packages/device/driver,
loading and preparation times, per-branch processing p50/p95/max, overruns and
peak allocated CUDA memory. Each branch needs at least 30 measured CUDA windows
and p95 total processing below 400 ms against a 500 ms hop. This is Falcon-only
throughput: upstream SepFormer time, file decoding/export, network, playback,
combined GPU contention, and end-to-end latency are not covered.

Two teammates should independently compare target intelligibility, interfering
speech leakage, distortion, and the target-silent interval. Record observations
for both targets. `quality_status` stays `pending_listening_review`; there is no
automatic winning architecture or accuracy score. Clean references are needed
before claiming reference-based separation metrics. A3000 acceptance and speech
quality remain pending until real recordings and hardware runs are reviewed.

## Automated verification

```powershell
.\.venv-model\Scripts\python.exe -m unittest discover -s tests -p test_validate_extraction.py -v
.\.venv-model\Scripts\python.exe -m unittest discover -s tests -p test_falcon_comparison.py -v
```

Generated media and injected model/slicer fixtures test correctness, publication
failure, gain/rate conversion, alignment, and listening exports without weights
or GPU. They do not validate real speech quality. The real CLI always uses the
pinned Falcon runtime and upstream output slicer.
