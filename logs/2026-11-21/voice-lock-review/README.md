# Voice-lock testing and recording review

Snapshots taken after pulling `7bfea43`, verifying Voice lock, and reviewing the
two latest recordings. The audio/video recordings themselves remain local.
Active log files continue to change independently of these snapshots.

- `backend-error.log`, `backend.log`, `dashboard-error.log`, `dashboard.log`:
  latest launcher/server log snapshots. Backend stderr includes normal diagnostic
  messages, not just errors.
- `lipsync-debug.log`: dedicated app log snapshot. During inspection, a normal
  backend import also reproduced the test-detection condition that disables this
  file handler, so the backend stderr snapshot was used for the recording review.
- `voice-lock-gpu-smoke.json`: latest six-second generated-input GPU check with
  voice lock after the local two-second warm-up adjustment. Seventeen windows,
  p95 separator inference 149.1 ms, median identity processing 10.9 ms, no skipped
  hops or playback underruns. The warm-up source changes are not part of this
  logs-only commit.
- `voice-lock-gpu-warm-smoke.json`: preceding diagnostic run with explicit extra
  warm-up, kept separately from the final smoke result.
- `recording-window-benchmark.json`: five measured passes each on a fixed excerpt
  of `live-test-20261121-233254/mix.wav`, after warm-up, using WHAMR 16k on the
  A3000. Inference p95: 142.2 ms for two-second windows and 268.1 ms for four-second
  windows. This excludes voice lock, networking, playback, and export; it does not
  establish improved speech quality for the longer window.
- `recording-metadata/`: metadata from the two most recent recordings when this
  snapshot was created. No speech samples are included.

The review examined the 65.75-second `233136` and 48.5-second `233254` recordings
through waveform/timing analysis, not a human listening review. Neither had
fully silent 20 ms blocks after the first two seconds or full-scale clipping.
The latest session logged zero server-side playback gaps, but network arrival
bursts and intermittent face detection remain relevant. The newest metadata had
59.3% unassigned lip picks; held picks are not independently verified identities.

Recommended follow-up: investigate client jitter buffering and Android partial
writes, compare smoother inter-window joins, and evaluate four-second context
with a 500 ms hop. These are proposals, not implemented changes or validated
quality improvements. No reference-based leakage/accuracy score is claimed.
