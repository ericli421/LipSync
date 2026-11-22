# Focus audio dropout investigation

Investigated November 22, 2026, using the active
`.tools/startup/lipsync-debug.log` and recordings
`live-test-20261122-040734` and `live-test-20261122-041651`.

## Finding and fix

Natural Focus could remain completely silent after a processing stall because
playback advanced an output buffer's start past the samples actually produced.
When separation resumed, the worker appended the skipped-hop padding and new
audio at that advanced position, shifting its output timeline into the future.
Natural Focus then requested original samples that had not arrived, received
zeros, and cached silence. The individual separated voices could still contain
sound because their samples came directly from the model, although their timing
was also shifted.

`StreamingSeparator.read` and `_read_mask` now retire at most the samples present
in their buffers. An empty buffer stays at the worker's actual output position,
so resumed output and skipped-hop padding keep their original timestamps.
Missing Focus output continues to fall back to the delayed original.

## Evidence

- At 04:13:33, Focus was selected after a period with late hops. At 04:13:52,
  playback was still -120 dBFS even though the original was -23.5 dBFS, the two
  separated voices were -33.0/-28.7 dBFS, the face was found in 100% of frames,
  and inference was back to 149/153 ms median/p95 with no recent late hops.
  Persistent silence outlasted the processing slowdown.
- A deterministic replay with a two-second worker stall moved the old output
  timeline ahead by 1,480 ms and made Focus remain silent after recovery, even
  with no face selection (where Focus should pass the original through).
- Both recent recordings contain substantial speech in the mix and separated
  tracks. The 04:16:51 recording also has 14 completely silent 250 ms blocks in
  each voice track, consistent with the logged late-hop padding. Those transient
  gaps are a separate effect from Focus remaining silent after recovery.
- The 04:07 session held voice 2 while voice 1 sometimes became much louder;
  selecting a quiet voice can lower the audible talker. That can explain quieter
  passages, but does not explain exact, persistent silence at the default -8 dB
  duck setting.

The recordings save mix/voice1/voice2 and aggregate pick fractions, not the played
Focus track, video, or per-hop picks. The exact original Focus output cannot be
reconstructed from those files alone.

## Validation

Replayed each recording through the output path with the recorded separated
tracks supplying model output, the mix expanded to 48 kHz, a forced stall from
4 to 6 seconds, and no face pick. In the checked tail beginning at 10 seconds:

| Recording | Old code: silent 20 ms blocks | Fixed code: silent 20 ms blocks |
| --- | ---: | ---: |
| 04:07:34 | 837 / 837 | 0 / 837 |
| 04:16:51 | 787 / 787 | 0 / 787 |

The fixed replay had zero timeline drift and matched the delayed original to
floating-point precision. This is a controlled recovery test using recorded
audio, not an exact replay of the live session's timing or camera decisions.

Added regression tests cover recovery with and without a picked voice, switching
from Voice 1 or Original to Focus after the stall, and target boost versus the
other voice after recovery. They fail with the old buffer logic.

All 28 tests passed:

```powershell
.venv-local/Scripts/python.exe -m unittest processor.tests.test_separation processor.tests.test_lock_in processor.tests.test_face_picker -v
```

Restart the Python backend to load the fix. A fresh phone/headphone test is still
needed to confirm the live experience. Occasional inference/Wi-Fi stalls and
incorrect camera-to-voice selection can still cause transient gaps or reduced
target volume; this change addresses the persistent buffer-induced silence.
