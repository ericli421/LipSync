# Recorded-video lip-to-voice matching

This experiment takes synchronized video/audio, separates two candidate voices,
tracks two visible faces, and compares their mouth-opening rhythms with the
candidate speech envelopes. It uses MediaPipe face landmarks and an audio-only
SepFormer; it does not use a pretrained lip-reading model.

The matcher is an **unvalidated heuristic**. "Tentative" means correlation and
margin thresholds were met, not that a speaker identity was verified. It can
make wrong assignments or stay uncertain. Its reports deliberately contain no
accuracy percentage without independent ground truth.

## Setup and run

Use Windows x64, Python 3.11, FFmpeg/ffprobe on PATH, and the tracked
`face_landmarker.task`. Create a separate environment or install the additional
landmark dependencies into the existing audio environment:

```powershell
py -3.11 -m venv .venv-lip-match
$matchPython = Join-Path (Get-Location) '.venv-lip-match\Scripts\python.exe'
& $matchPython -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
& $matchPython -m pip install -r requirements-lip-match.lock.txt
```

For the A3000, use the matching `cu126` PyTorch/Torchaudio wheel index instead
of `cpu`. The development checkout's `.venv-audio` also has the required
dependencies installed. The Falcon environment is independent.

Run the supplied debate clip:

```powershell
& $matchPython scripts/match_lips.py `
  --video video-samples/debate.mp4 `
  --roi 0 640 1080 640 `
  --out-dir validation-output/my-debate-run
```

The ROI is an upright pixel crop: X, Y, width, height. These values select the
actual split-screen broadcast in this 1080×1920 clip and exclude its blurred
decorative background. Omit `--roi` for a normal video with sufficiently large
faces. It is not a target-identity override; both people are detected and tracked.

The CLI runs the pinned audio-only model automatically. You can reuse a completed
separation with `--separation-dir validation-output/debate-audio`. Before reuse,
the runner verifies that the prepared source audio agrees with the cached mixture
after its recorded shared export gain, and that candidates share its timeline.
Wrong recordings or incompatible cached results are rejected.

Other options: `--device auto|cpu|cuda`, `--face-model PATH`, `--min-score 0.30`,
and `--margin 0.10`. Always choose a fresh output directory. Processing is local;
only missing public model assets are downloaded. Source videos, generated
recordings, and results are ignored by Git.

## Outputs

| Output | Purpose |
| --- | --- |
| `annotated.mp4` | Cropped video, tracked faces, rolling scores, original mixture audio |
| `face_0.mp4`, `face_1.mp4` | Same annotated video, with the tentative per-face audio selection |
| `face_0.wav`, `face_1.wav` | Per-face selected audio, with uncertainty muted |
| `separation/` | Original mixture, two candidate WAVs, and audio-only metrics |
| `matches.json` | Per-window scores, selected source, hold/uncertainty state, and estimated lag |
| `tracking.json` | Measured mouth-opening ratios, mouth outlines, and face boxes |
| `report.json` | Source/timing provenance, coverage, parameters, and shifted-video control |
| `prepared.mp4`, `input.wav` | Timestamp-aligned analysis inputs for debugging |

Face 0 initially means the leftmost detected face; Face 1 is the other face.
The tracker associates nearby face positions across frames even when MediaPipe's
output order changes. This is geometry tracking, not biometric identification
across camera cuts, occlusion, or people swapping places.

Preview MP4 audio is loudness-normalized for listening; saved analysis WAVs are
unchanged. Compare words and interfering speech, not perceived volume alone.

## Matching behavior

- Decode both media streams onto a common timestamp origin. Analyze video at
  25 fps and audio at 8 kHz, retaining complete synchronized frame intervals.
- Measure inner-lip gap divided by mouth width, which reduces sensitivity to
  face size and in-plane head rotation.
- Compute 40 ms RMS energy envelopes for the two candidates. Compare their
  log-energy fluctuations with mouth opening using a 0.7–7 Hz bandpass.
- Use two-second windows, updated every 200 ms, searching audiovisual lags up
  to ±240 ms. Require correlation at least 0.30 and a margin of 0.10 over the
  alternative. These are provisional, configurable thresholds.
- Reject weak/no mouth motion and missing landmarks. A candidate cannot be
  assigned to both faces simultaneously. Require two successive decisions to
  change an established candidate association.
- Hold a recent association for up to one second through weak evidence. Mute
  after that, or immediately when the face is lost; use 20 ms audio crossfades.
- Repeat matching with the mouth traces shifted by one second. Report how much
  synchronized best-correlation scores exceed the shifted scores. This is a
  diagnostic comparison, not a significance test, clean reference, or accuracy.

The analysis is offline: interpolation, zero-phase filtering, and positive lag
search use future context. It must not be described as a causal streaming
implementation. The first two seconds mostly remain uncertain while context
accumulates. Scores are correlations and may be misleading on short windows;
they are not probabilities.

## Result on the supplied debate

The clip is 32.73 seconds of picture with slightly longer audio. The runner
retains **818 frames / 32.72 seconds**, tracking both faces throughout. The
audio-only CPU separation took about **20.86 seconds** in its measured pass.

The first lip-matching result is **inconclusive**: synchronized best-correlation
scores exceeded the one-second-shift control by only about **0.045** on average,
below the fixed 0.05 diagnostic threshold. A face-specific diagnostic can differ
from the aggregate; neither establishes correct identity.

The clip is mostly one person speaking while another listens, with brief
turn-taking. Both candidate streams can contain leakage and the heuristic can
switch between them. It is useful for checking detection, synchronization,
selection, and playback, but does **not** establish reliable separation during
sustained overlapping speech. Review `annotated.mp4`, both raw candidates, and
the per-face previews before drawing quality conclusions.

A stronger next test is a stationary, synchronized video with two different
people speaking independently at the same time, plus solo sections or separate
reference recordings for labeling. Do not adjust thresholds solely until the
debate clip looks convincing; evaluate changes on a separate recording.

## Verification

```powershell
& $matchPython -m unittest discover -s tests -v
```

Tests cover known synthetic face/source associations, lag recovery, shifted-video
degradation, detector-order changes, face loss, stationary mouths, duplicate-source
ambiguity, unique assignments, muted output, crossfades, source timestamps, and
cached-audio mismatch rejection. Generated fixtures do not validate real-world
speech quality. The real clip is separately run through the actual landmark and
audio models.
