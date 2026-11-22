# GPU verification and subsequent session logs

Snapshots captured after pulling `21b31bc` and testing the status-update and
sample-rate fixes. Active logs remain under ignored `.tools/startup/`.

- `smoke-separator.json`: pinned SepFormer-WHAMR 16k loaded on the NVIDIA RTX
  A3000 with PyTorch 2.9.1+cu126; one generated 1.5-second input took about 109 ms.
- `live-separation-gpu-smoke.json`: six seconds of generated audio through the
  streaming separator, 2-second windows and 250 ms hops. Seventeen windows ran,
  p95 inference was 149.8 ms, with no skipped hops or playback underruns.
- `lipsync-debug.log`: cumulative app diagnostics, including test fixtures and
  subsequent phone sessions. It includes other model selections and actual
  network/face-tracking conditions; do not interpret the generated smoke result
  as a result for every live session.
- `backend*.log` and `dashboard*.log`: launcher/server snapshots. Informational
  backend output is also written to stderr; its filename does not imply failure.

Generated tones do not establish speech isolation quality or lip-selection
accuracy. The phone-session tail reports arrival gaps, playback gaps, and
face-not-visible decisions that should be investigated separately from GPU speed.

Checks completed: 28 backend tests, 51 offline tests, 3 PCM playback tests,
TypeScript, ESLint, Next.js build, Android debug assembly/unit tests/lint, and
launcher startup/shutdown. Falcon remains disabled in the live app.
