# Audio-only separation: first runnable experiment

Run a pretrained audio-only separator on a public speech mixture and listen to
its two candidate outputs. No microphone, camera, personal recording, or A3000 is
required for the first CPU run. This is the first step toward separating audio
and then selecting a voice using lip motion; the lip matcher is not implemented yet.

## Setup

The dependency lock targets **Windows x64 and Python 3.11**. Keep this environment
separate from the earlier Falcon environment. From the repository root, run:

```powershell
py -3.11 -m venv .venv-audio
$audioPython = Join-Path (Get-Location) '.venv-audio\Scripts\python.exe'
& $audioPython -m pip install --upgrade pip
& $audioPython -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
& $audioPython -m pip install -r requirements-audio-only.lock.txt
```

If Python 3.11 is not registered with the Windows launcher, install it first or
use `uv venv --python 3.11 .venv-audio`, then continue with `$audioPython` above.
The developer checkout already has a working `.venv-audio` environment.

On the **A3000 laptop**, use matching CUDA wheels instead of the CPU install line:

```powershell
& $audioPython -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
```

Use a fresh environment when changing builds, and ensure the NVIDIA driver
supports the wheel's CUDA runtime. Both PyTorch and Torchaudio must use the same
version and CPU/CUDA build. The lock's `==2.9.1` accepts the `+cpu`/`+cu126` suffixes.
SoundFile handles WAV I/O, so this runner does not require FFmpeg or TorchCodec.

## Run the public sample

```powershell
& $audioPython scripts/separate_audio.py --out-dir validation-output/audio-public
```

The default device is `auto`: CUDA if available, otherwise CPU. Specify
`--device cpu` or `--device cuda` to require a device. An unavailable explicit
CUDA request fails rather than silently falling back.

The first run downloads approximately **113 MB** of inference weights plus a
small configuration and sample. No account is needed for these public files.
Later runs reuse the Hugging Face cache. All model files and the sample have
pinned revisions and SHA-256 checks in `audio-separation.lock.json`. SpeechBrain
loads the verified local files without creating Windows symlinks or fetching
additional model assets.

The model is `speechbrain/sepformer-whamr` at revision
`9950fd8055b38679d2af5bef2366b5dba600ec66`. The public sample is
`test_mixture.wav` from `speechbrain/sepformer-wsj02mix` at revision
`3a2826343a10e2d2e8a75f79aeab5ff3a2473531`. It contains **16,544 samples of 8 kHz
mono floating-point WAV**, or **2.068 seconds**. The sample comes from a separate
model repository; it is an initial smoke test, not a WHAMR evaluation set.

Each output directory must be new. To verify cached operation in a separate
PowerShell process without network access:

```powershell
$env:HF_HUB_OFFLINE = '1'
& $audioPython scripts/separate_audio.py --out-dir validation-output/audio-public-offline
Remove-Item Env:HF_HUB_OFFLINE
```

## Inspect the outputs

| File | Contents |
| --- | --- |
| `mixture.wav` | Prepared mono input |
| `candidate_0.wav` | First separated source estimate |
| `candidate_1.wav` | Second separated source estimate |
| `metrics.json` | Asset hashes/pins, actual software/device, input conversion, timing, and export gain |

All three WAVs are 8 kHz mono PCM16 with identical sample counts. The runner
uses `separate_batch` to preserve model output levels, then applies **one shared
attenuation factor** to all three files only if needed to cap their peaks at 0.99.
It does not normalize the candidates independently. Separation models can have
arbitrary source gains, so this shared clipping protection is not a loudness match
between the mixture and estimated sources. Compare intelligibility, not volume alone.

Candidate indices are anonymous output channels. They do not identify people or
guarantee continuity across separate runs or future processing chunks.

The model loads once, runs one full-clip warm-up, and then one measured full-clip
pass. Inference timing includes tensor assembly, transfers, separation, return to
CPU, and finite-output validation. CUDA is synchronized around timed work. Asset
fetching, model loading, input decoding/resampling, and WAV export are excluded
from inference time. Their relevant setup timings are recorded separately.

`real_time_factor` is inference duration divided by audio duration. A value below
one on this short file does **not** establish streaming suitability or
capture-to-headphone delay. The report deliberately keeps `quality_status` as
`pending_listening_review` and `streaming_suitability` as `not_evaluated`. No
SI-SDR or other reference-based quality score is reported because the supplied
sample does not include separate clean reference tracks.

## Local verification

On November 21, 2026, Windows/Python 3.11.18 with PyTorch/Torchaudio 2.9.1+cpu,
SpeechBrain 1.1.2, and four CPU threads completed the real-checkpoint public sample:

| Run | Inference | Real-time factor | Output length per WAV |
| --- | --- | --- | --- |
| Initial public sample | 1243.0 ms | 0.601 | 16,544 samples |
| New process, `HF_HUB_OFFLINE=1` | 1204.9 ms | 0.583 | 16,544 samples |

These are single measured passes after warm-up. A3000 execution and human
listening quality have not been validated. Generated files live under ignored
`validation-output/`; weights remain in the Hugging Face cache.

## Run your own recording later

```powershell
& $audioPython scripts/separate_audio.py `
  --input recordings/two-speakers.wav `
  --device cuda `
  --out-dir validation-output/audio-custom
```

The runner accepts PCM or floating-point WAV, averages input channels into mono,
and resamples to 8 kHz using a band-limited polyphase resampler. It leaves the
original recording untouched. Empty, too-short, non-finite, corrupt, and non-WAV
inputs fail clearly; failed inference/export does not publish a partial run.

No audio from you is needed now. The next useful data is a **20–30-second recording
of two different people**: A alone, B alone, then both speaking simultaneously.
For the later lip matcher, keep the original synchronized video with both mouths
clearly visible. Live microphone capture, lip matching, training, and chunked
streaming are outside this first experiment.

## Tests

```powershell
& $audioPython -m unittest discover -s tests -p test_separate_audio.py -v
```

The tests generate WAV fixtures and inject a small test model; they need no model
download or GPU. They cover float WAV decoding, downmixing/resampling, pinned
asset verification, device selection, timing synchronization, output alignment,
shared gain, errors, and atomic publication. The CLI always loads the real model.
In the older Falcon environment, these tests skip if audio-only dependencies are
missing; the existing Falcon tests remain independently runnable.

Model/configuration and sample credit: [SpeechBrain SepFormer-WHAMR](https://huggingface.co/speechbrain/sepformer-whamr)
and [SepFormer-WSJ02Mix public sample](https://huggingface.co/speechbrain/sepformer-wsj02mix/blob/3a2826343a10e2d2e8a75f79aeab5ff3a2473531/test_mixture.wav).
See the [SpeechBrain separation interface](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.separation.html)
and [PyTorch installation guide](https://pytorch.org/get-started/locally/).
