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
dependencies and cache/smoke-test the model. **On a PC with an NVIDIA GPU (the
A3000 laptop) it installs CUDA PyTorch automatically**, replacing a CPU build if
one is installed, and smoke-tests on the GPU. It does not enable or run Falcon.
Then double-click **Start-LipSync.cmd** to start the backend and dashboard.
Normal startup does not install or load model weights.

Install the updated `Lipsync/app/build/outputs/apk/debug/app-debug.apk` to get
the phone's SepFormer controls and debug log. Existing APKs can still capture;
use the dashboard controls if you have not updated the phone yet.

**Start-LipSync.cmd** prints where separation will run. On the A3000 it must say
*Separation will run on the GPU*. If an NVIDIA GPU is present but PyTorch cannot
use it, startup prints a red warning and test mode refuses to run on the CPU
(set `LIPSYNC_ALLOW_CPU=1` only to test on the CPU on purpose).

Choose the model in the phone app or dashboard and keep the best one:

| Choice | Model | Notes |
| --- | --- | --- |
| Auto | WHAMR 16k on GPU, ReSepFormer on CPU | default |
| WHAMR 16k | SepFormer-WHAMR 16 kHz | noise + reverb training, 16 kHz audio; GPU only (7.5 s per 2 s window on an i7-1260P) |
| WHAMR 8k | SepFormer-WHAMR 8 kHz | noise + reverb training, telephone bandwidth; GPU only |
| ReSepFormer | ReSepFormer 8 kHz | fastest; trained on clean speech, weakest in a real room |

Defaults are window **2 s**, hop **0.25 s** (0.75 s delay). Without a GPU use
ReSepFormer, window 1 s, hop 0.5 s; that does not guarantee real-time performance. Enable **Test mode**, wait for the
model to load, and try **Voice 1**, **Voice 2**, then **Focus** with headphones.
Check skipped hops and the "Keeping up" indicator before judging quality.

`Setup-Models.cmd -Device cpu` or `-Device cuda` forces a build; `-ReplaceTorch`
reinstalls it. Existing CUDA Torch is retained by a normal setup run. The manual
setup below remains available.

### Logs to share after a test

Everything the phone's **Debug log** shows is also written to
`.tools/startup/lipsync-debug.log` (all levels, rotated at 2 MB). Every 5 s it
records a **stream** line (audio blocks received, largest Wi-Fi arrival gap,
input/output level) and, while separating, a **test** line: model and device,
window/hop/delay, model time median/p95, hops, hops dropped as late in the last
10 s, playback gaps, voice levels and their average gap, the share of lip-pick
outcomes (matched, ambiguous, face not visible, lips not moving), the current
pick and what you are listening to. Dropped hops say whether the model was too
slow or the audio arrived late. Send that file, or tap **Copy** in the app.

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
2. In the phone app tap **Start SepFormer test** (or in the dashboard,
   **Separation test → Test mode · On**). Model **Auto** picks SepFormer-WHAMR 16k
   on CUDA (trained with noise and reverb) and ReSepFormer on CPU. Start with
   window **2 s**, hop **0.25 s**; switch models with the Model chips to compare.
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
  hop 0.5 s), plus Wi-Fi and phone playback buffering. A hop is dropped (heard as
  a gap) only when its output would already be late, so delay never grows; audio
  that arrives in a burst but is still in time is processed back to back. The log
  says whether drops came from a slow model or from late audio.
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
