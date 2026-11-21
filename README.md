# LipSync
A project for CodeJam16

The project explores selective listening through glasses: separate candidate
voices from audio, then match their speech rhythms to the selected person's lip
motion. The current milestone is recorded audio-only separation; lip matching
and live glasses capture follow afterward.

- Start with [the audio-only separation guide](docs/audio-only-separation.md):
  run the public two-speaker mixture and export two candidate voices. No microphone
  or personal audio data is needed.
- The earlier [Falcon model-validation guide](docs/model-validation.md) documents
  the independent lip-conditioned baseline and recorded-video experiment.
