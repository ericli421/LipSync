# LipSync
A project for CodeJam16

LipSync explores selective listening: separate candidate voices, then match their
speech rhythms to the selected person's lip motion. The current hardware client
is a native Android APK; the laptop runs a local Python backend and web dashboard.

- Prepare the next model experiment with [the Falcon/SepFormer comparison](docs/falcon-comparison.md):
  export aligned input for SepFormer, then compare Falcon on the mix and both
  candidates without changing the live pipeline.

- Start with [the local Android demonstration](docs/local-android-demo.md) for
  live phone camera capture, mouth tracking, and raw audio monitoring. No account
  or Supabase setup is needed. Live voice separation is a subsequent milestone.

- Start with [the audio-only separation guide](docs/audio-only-separation.md):
  run the public two-speaker mixture and export two candidate voices. No microphone
  or personal audio data is needed.
- Try [the recorded-video lip-matching experiment](docs/lip-matching.md) to track
  two faces, inspect correlation scores, and play per-face selections.
- The earlier [Falcon model-validation guide](docs/model-validation.md) documents
  the independent lip-conditioned baseline and recorded-video experiment.
