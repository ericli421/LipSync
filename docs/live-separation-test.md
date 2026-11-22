# Live separation test mode

Test, in real time, whether a pretrained separator can split two people talking
into two voices, and whether lip tracking picks the person the camera is on.
The phone captures as usual; the laptop separates. Nothing is trained.

**Run this on the A3000 laptop.** On an i7-1260P CPU (no GPU) the loop does not
keep up: ReSepFormer, 15 s streamed in real time through the server, measured
137–573 ms median per 1 s window against a 250 ms budget, skipping 8–33 of
~54 hops. SepFormer-WHAMR took 2.2 s per 1 s window on that CPU.

## Setup (A3000)

### One-click setup and local CPU test

Double-click **Setup-Models.cmd** once to install the optional separation
dependencies and cache/smoke-test the CPU model. It does not enable or run Falcon.
Then double-click **Start-LipSync.cmd** to start the backend and dashboard.
Normal startup does not install or load model weights.

Install the updated `Lipsync/app/build/outputs/apk/debug/app-debug.apk` to get
the phone's SepFormer controls and debug log. Existing APKs can still capture;
use the dashboard controls if you have not updated the phone yet.

On this PC use **Auto** (ReSepFormer on CPU), window **1 s**, hop **0.5 s**
for an initial test. This trades context and update rate for more CPU headroom;
it does not guarantee real-time performance. Enable **Test mode**, wait for the
model to load, and try **Voice 1**, **Voice 2**, then **Focus** with headphones.
Check skipped hops and the "Keeping up" indicator before judging quality.

For a new A3000 setup, run `Setup-Models.cmd -Device cuda`. If CPU Torch already
exists, replacement requires the explicit `-ReplaceTorch` switch. Existing CUDA
Torch is retained by a normal setup run. The manual setup below remains available.

Falcon's standalone offline harness is preserved separately. The unfinished live
integration is parked under ignored `.tools/falcon-integration-draft/` and is not
imported or shown by the app.

```powershell
py -3.11 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r processor/requirements.txt
.\.venv-local\Scripts\python.exe -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-local\Scripts\python.exe -m pip install speechbrain==1.1.2 soundfile==0.14.0 huggingface_hub==2.4.0
.\.venv-local\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name. Then start the server and
dashboard exactly as in [the Android demo](local-android-demo.md). Models download
on first use (pinned revisions, SHA-256 verified) into the Hugging Face cache.

## Run the two-person test

1. Start capture in the APK (48 kHz is required; the APK prefers it).
2. In the dashboard, **Separation test → Test mode · On**. Model **Auto** picks
   SepFormer-WHAMR on CUDA (trained with noise and reverb) and ReSepFormer on CPU.
   Start with window **1.5 s**, hop **0.25 s**.
3. Check **Keeping up: Yes**. If it says No, try hop 0.5 s or window 1 s.
4. **Take turns talking.** Each person should move only one voice meter. A level
   gap of about 10 dB or more is a clear split; under ~3 dB it is not separating.
5. **Talk at the same time.** Listen to *Voice 1* and *Voice 2* on headphones.
6. **Point the camera at one person** and choose *Focus*. Their stream should be
   marked "on camera", be boosted (+3 dB default) and everything else lowered
   (−15 dB default). Swap the camera to the other person: the pick should follow
   within about two hops. If the pick is wrong or "ambiguous", adjust the
   **Lip/audio offset** (phone audio and video timestamps are not verified to be
   synchronized).
7. Play background noise and repeat 4–6.
8. **Record test clip** saves `recordings/live-test-*/` with `mix.wav`,
   `voice1.wav`, `voice2.wav` (8 kHz, aligned) and `meta.json` (settings, model,
   share of time each voice was picked) for offline scoring.

## How it works and its limits

- Every hop, the last *window* seconds (resampled to 8 kHz) are separated into
  two candidates. Each chunk's candidates are reordered and sign-corrected to
  match the previous chunk on their overlap ("order swaps fixed" counts these),
  then crossfaded. Only audio ending 0.25 s before the newest sample is used.
- Server delay is fixed at lookahead + 2 hops (0.75 s at hop 0.25 s, 1.25 s at
  hop 0.5 s), plus Wi-Fi and phone playback buffering. If the model is slower than
  one hop, hops are skipped (heard as gaps) rather than letting delay grow.
- Focus mix = boost × target + duck × (delayed original − target). The target is
  the candidate whose loudness envelope best correlates with the on-camera mouth
  openness (the logic of `scripts/lip_matching.py`, per chunk, with ±0.24 s lag,
  minimum correlation 0.30, margin 0.10, two-hop switch confirmation). Until a
  confident match exists, the original passes through unchanged.
- Separated voices are 8 kHz (telephone bandwidth); the ducked remainder keeps
  the full band. Both models output exactly two voices: a third talker or loud
  music leaks into one of them. Lip video on the dashboard is not delayed.
- `LIPSYNC_TORCH_THREADS` optionally caps CPU threads; capping 4–8 threads was
  slower than torch's default on the i7-1260P.
