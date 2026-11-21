# Validate lip-conditioned voice extraction

This contribution tests Falcon Small on a recorded, stationary target before
integrating live phone capture. A successful extraction run creates an aligned
original/isolated audio pair, the mouth-crop video, and a machine-readable report.
It does not train a model, track moving people, or measure live listening latency.

## Current evidence

The development computer has an AMD Radeon RX 6600, not the team's NVIDIA A3000.
The A3000 benchmark and two-speaker listening review are **not yet measured**.
Synthetic-video tests exercise synchronization, rotation, output alignment, and
benchmark decisions. Synthetic images/tones are not evidence of speech quality.

On November 21, 2026, a real-checkpoint smoke run completed on Windows with Python
3.11.18 and PyTorch 2.9.1+cpu. A three-second generated test-pattern/sine-wave video
produced three measured windows and a finite 1.5-second extracted WAV aligned to
the original source interval `[1.42, 2.92)`. Processing p50/p95/max were
562.5/583.0/585.3 ms. The report correctly marked this `not_validated_cpu`.
This short synthetic run proves loading and output generation only. CUDA
compatibility, memory usage, speech quality, and throughput still need the A3000.
A separate-process rerun with `HF_HUB_OFFLINE=1` also completed using cached
weights. Explicit CUDA execution on this computer failed with an actionable
error and did not publish an output directory.

## Reproducible setup on the A3000 laptop

Run these commands in PowerShell from the repository root. Prerequisites: Git,
Python 3.11, FFmpeg/ffprobe on PATH, and a working NVIDIA driver. Confirm
`nvidia-smi` sees the A3000. The PyTorch wheel includes its CUDA runtime; installing
a separate CUDA toolkit is not required for this harness.

```powershell
py -3.11 -m venv .venv-model
$modelPython = Join-Path (Get-Location) '.venv-model\Scripts\python.exe'
& $modelPython -m pip install --upgrade pip
& $modelPython -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
& $modelPython -m pip install -r requirements-model.lock.txt
& $modelPython -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CUDA UNAVAILABLE')"

git clone --filter=blob:none --sparse https://github.com/JusperLee/Falcon.git external/Falcon
git -C external/Falcon sparse-checkout set studio
git -C external/Falcon checkout --detach 5c5d4ff6ce99a63742101a3a7992592068df496c
```

If Python 3.11 is not installed, install it first; do not use the system's 3.13 or
3.14 environment for this experiment. An alternative with [uv](https://docs.astral.sh/uv/)
is `uv venv --python 3.11 .venv-model`; then use the same `$modelPython` and pip
commands above. If the checkout already exists, skip cloning and check out the
pinned commit in that clean external checkout.

This inference environment deliberately pins PyTorch 2.9.1, which has Windows
CUDA 12.6 wheels. It uses its own dependency lock rather than upstream Studio's
newer dependency minimums and unconditional `uvloop` lock entry. A real-checkpoint
CPU smoke run verifies this combination locally; CUDA execution still needs the
A3000 run. Update the NVIDIA driver if the CUDA wheel cannot initialize. Do not
switch to CPU and report that result as GPU performance.

For CPU-only correctness checks on another computer, replace the PyTorch install
command with:

```powershell
& $modelPython -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

`model-validation.lock.json` pins both source and checkpoint:

| Item | Pin |
| --- | --- |
| Falcon source commit | `5c5d4ff6ce99a63742101a3a7992592068df496c` |
| Model repository | `JusperLee/Falcon` |
| Model revision | `e50af93425c65dfabf15a7f6ea62991c5b632838` |
| Checkpoint | `falcon.pt` |
| Checkpoint SHA-256 | `a1888b5a3484a38a68c42d1a0e73334b1f2e7924c6e6444b2cfdd27becce928c` |

The first run downloads the approximately 47 MB public checkpoint into the Hugging
Face cache; no account is required. Later runs reuse that cache. The harness checks
the commit, tracked Studio changes, and checkpoint hash before inference. Every
report includes actual installed package versions and the input video hash.

## Record and run the experiment

1. Put the phone on a stationary support. Record about **20 seconds** of two
   visible people speaking simultaneously, with their mouths unobstructed and
   large enough to crop. Include a pause where A stops and B continues. Use the
   phone's microphone, not separate microphone tracks for each person.
2. Save the file under ignored `recordings/`, for example `recordings/two-speakers.mp4`.
3. Export an upright frame to measure a square mouth box. Coordinates are pixels
   in this frame, with X/Y measured from its upper-left corner:

   ```powershell
   ffmpeg -i recordings/two-speakers.mp4 -frames:v 1 recordings/first-frame.png
   ```

4. Choose a square containing A's mouth with a little surrounding context. Keep
   the entire square inside the frame. Run the harness; **the box below is only
   an example and must be replaced with your measured coordinates**:

   ```powershell
   & $modelPython scripts/validate_extraction.py `
     --falcon-dir external/Falcon `
     --video recordings/two-speakers.mp4 `
     --mouth-box 200 140 80 `
     --device cuda `
     --out-dir validation-output/speaker-a
   ```

5. Repeat with B's mouth box and a new output directory, such as
   `validation-output/speaker-b`. The mixed audio must be the same recording.
6. Inspect each `mouth_preview.mp4` before judging the extracted speech. If a mouth
   moves out of its fixed box, record again; tracking belongs to the next milestone.

The CLI defaults to CUDA and fails if CUDA is unavailable. Explicit `--device cpu`
is supported for correctness checks. Output directories must not already exist.
A completed run exits zero even if the throughput gate fails: read `metrics.json`.
Input, dependency, or runtime errors exit nonzero and do not publish a new run.

## Outputs and integration contract

| Artifact | Contents |
| --- | --- |
| `mixture.wav` | 16 kHz mono PCM16 original over the emitted source interval |
| `target.wav` | Sample-aligned 16 kHz mono PCM16 extracted speech |
| `mouth_preview.mp4` | Prepared 96×96 grayscale mouth sequence at 25 fps, including model context |
| `metrics.json` | Pins, source/crop provenance, device, package versions, per-window timings, memory, and gate status |

FFmpeg preserves source timestamps, trims to the common audio/video interval,
autorotates phone video, resamples audio, and samples video on a 25 fps grid. Both
streams use the **same timeline origin**; they are not independently reset. The
report records original starts, common bounds, and the final retained bound.

The model receives float32 audio shaped `(32000,)` and uint8 mouth video shaped
`(50, 96, 96)` per two-second window. It loads once, warms up, and advances by
8000 samples (500 ms). Video frames are chosen on the timeline, accounting for the
12.5-frame hop. The upstream `OutputSlicer` supplies its 80 ms right context and
20 ms crossfade; no extraction or join code is copied into this repository.

Only full windows are processed, with no synthetic start or tail padding. The first
output therefore begins 1.42 seconds into the retained input. At exactly 20 seconds,
37 windows emit 18.5 seconds covering `[1.42, 19.92)` relative to the common origin.
Both WAVs cover precisely that range. These source positions are **not live latency**.

## Decide whether to integrate

- Require at least **30 measured windows** on CUDA. Three extra real-input warm-up
  calls and the runtime's initialization warm-up are excluded from timing.
- Require **p95 total window processing below 400 ms**, against a 500 ms hop budget.
  Inspect maximum time and the count of windows at or above 500 ms as well.
- Inference timing includes the upstream preprocessing and CPU/GPU transfer inside
  `runtime.infer`. Total window timing additionally includes window assembly,
  validation, and output slicing. CUDA is synchronized before/after measured work.
- `preparation_ms` separately measures recorded-file probing/decoding/resampling.
  File encoding, network, capture, playback, and the live pipeline are excluded
  from window timings. Peak CUDA memory reports allocated tensor memory after
  warm-up, including resident weights; it is not total driver-reserved VRAM.
- Listen at comparable loudness. Each of your two teammates independently checks
  whether A is clearer with A selected, B with B selected, and whether B leaks
  through during A's pause. Record actual observations rather than inferring
  quality from model output or timing alone.

`throughput_status` is one of `not_validated_cpu`, `insufficient_windows`,
`throughput_insufficient`, or `throughput_gate_passed`. `quality_status` remains
`pending_listening_review`: software cannot declare the listening test passed.

Append the following completed report to this document in your A3000 results PR:

| Evidence | Result |
| --- | --- |
| GPU / driver / PyTorch CUDA build | Not measured |
| Measured windows / processing p50, p95, max | Not measured |
| Windows over 500 ms / peak allocated VRAM | Not measured |
| A selected: reviewers' observations | Not reviewed |
| B selected: reviewers' observations | Not reviewed |
| Target-silent interval | Not reviewed |
| Repeat after restarting process | Not run |
| Conclusion | Validation incomplete |

Choose **ready for integration** only after both throughput and listening checks
pass; otherwise record **quality insufficient**, **throughput insufficient**, or
the specific setup blocker. Keep recordings/weights/generated audio out of Git.

## Automated checks

```powershell
& $modelPython -m unittest discover -s tests -v
```

Tests use generated media and an injected test runtime, with no checkpoint download
or GPU requirement. They cover 30-to-25 fps and stereo-to-mono resampling, delayed
audio timestamps, portrait rotation, half-frame hops, aligned WAVs, invalid inputs,
non-finite output, CUDA synchronization, and throughput decisions. The injected
runtime is only in the tests; the CLI always loads the real pinned model.

## Upstream credit

Model implementation, preprocessing, checkpoint loading, and output slicing come
from [Falcon](https://github.com/JusperLee/Falcon) under its
[Apache-2.0 license](https://github.com/JusperLee/Falcon/blob/5c5d4ff6ce99a63742101a3a7992592068df496c/LICENSE).
See the [Studio pipeline documentation](https://github.com/JusperLee/Falcon/blob/5c5d4ff6ce99a63742101a3a7992592068df496c/studio/README.md)
and [PyTorch installation guide](https://pytorch.org/get-started/locally/).
