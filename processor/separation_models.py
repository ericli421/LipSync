"""Pinned pretrained separators for live test mode. torch is imported lazily so
the relay runs without it; run Setup-Models.cmd (CUDA torch on NVIDIA PCs)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "sepformer-whamr16k": {
        "label": "SepFormer-WHAMR 16k (noise + reverb, 16 kHz; GPU)",
        "pin": {
            "repository": "speechbrain/sepformer-whamr16k",
            "revision": "21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5",
            "sample_rate": 16000, "sources": 2,
            "files": {
                "hyperparams.yaml": "eeef8d1450e1e06417eda07aa87ded71c6d9f3394852a92805e6aaf5229a50c6",
                "encoder.ckpt": "31d23d395a408b887b8f6ac01e477f00bcba27f95785426359fb52d52f1dc6ed",
                "decoder.ckpt": "2d959cb46de5b15f008bf5476cf7c19bc680b310c7e10883e3eddfdbde533cb8",
                "masknet.ckpt": "e5fb8c690668e5d1bbbc9a8256974577093a09cd08e845e9f30024fdd33472ce",
            },
        },
    },
    "sepformer-whamr": {
        "label": "SepFormer-WHAMR 8k (noise + reverb; GPU)",
        "pin": None,  # read from audio-separation.lock.json
    },
    "resepformer-wsj02mix": {
        "label": "ReSepFormer 8k (fast on CPU; clean-speech trained)",
        "pin": {
            "repository": "speechbrain/resepformer-wsj02mix",
            "revision": "b8e127bf2b3585c95eebbe7b786e9d3f16675156",
            "sample_rate": 8000, "sources": 2,
            "files": {
                "hyperparams.yaml": "37aa180ad4127ca20ae6cf4b410078aa01aee240797ba8e229cadf55660e5ca5",
                "encoder.ckpt": "3908eb3146ab80c464c4a05a5cd9824ddd341694f38fae641f2625d2bb3bd97f",
                "decoder.ckpt": "5504e2cecb7e0f94164a3bda231491bbef242e22d41762759e87f6f7699a9049",
                "masknet.ckpt": "c7d93084e4189c384fb46f93df477977813afc19b01b2d4aa632e70fe04de2ad",
            },
        },
    },
}
GPU_DEFAULT, CPU_DEFAULT = "sepformer-whamr16k", "resepformer-wsj02mix"


def _separate_audio():
    sys.path.insert(0, str(ROOT / "scripts"))
    import separate_audio
    return separate_audio


def resolve(name, cuda):
    if name == "auto":
        return GPU_DEFAULT if cuda else CPU_DEFAULT
    if name not in MODELS:
        raise ValueError(f"unknown model {name}")
    return name


def nvidia_gpu():
    """Name of an NVIDIA GPU reported by the driver, or None."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    try:
        result = subprocess.run([tool, "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    name = result.stdout.strip().splitlines()[0].strip() if result.returncode == 0 and result.stdout.strip() else None
    return name or None


def choose_device(torch, requested="auto", gpu_name=None):
    """Never fall back to CPU silently on a PC that has an NVIDIA GPU."""
    cuda = torch.cuda.is_available()
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False; run Setup-Models.cmd.")
    if requested != "auto":
        return requested
    if cuda:
        return "cuda"
    if gpu_name and os.environ.get("LIPSYNC_ALLOW_CPU") != "1":
        build = "CPU-only" if torch.version.cuda is None else f"CUDA {torch.version.cuda}"
        raise RuntimeError(
            f"This PC has an NVIDIA GPU ({gpu_name}) but PyTorch {torch.__version__} ({build}) cannot use it. "
            "Close LipSync, run Setup-Models.cmd (it installs CUDA PyTorch on NVIDIA PCs), then restart. "
            "To test on the CPU anyway, set LIPSYNC_ALLOW_CPU=1.")
    return "cpu"


def load(name="auto", device="auto"):
    """Return (separate(window float32 [n]) -> float [n, 2], info dict)."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Run Setup-Models.cmd.") from exc
    device = choose_device(torch, device, nvidia_gpu())
    # Optional CPU thread cap. On an i7-1260P, capping at 4-8 threads was slower
    # than torch's default in the live loop, so the default is left alone.
    threads = os.environ.get("LIPSYNC_TORCH_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    name = resolve(name, device == "cuda")
    sa = _separate_audio()
    pin = MODELS[name]["pin"] or json.loads(sa.PIN_FILE.read_text())["model"]
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(repo_id=pin["repository"], revision=pin["revision"], allow_patterns=list(pin["files"])))
    for file, expected in pin["files"].items():
        sa.verify_file(directory / file, expected)
    model = sa.load_model(directory, device, pin)
    rate = int(pin["sample_rate"])
    lock = threading.Lock()

    def separate(window):
        with lock, torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0).to(device)
            estimate = model.separate_batch(batch)[0]
            return estimate.float().cpu().numpy()

    separate(np.random.default_rng(0).normal(0, 0.05, int(1.5 * rate)).astype(np.float32))  # warm-up
    info = {"model": name, "label": MODELS[name]["label"], "device": device, "sample_rate": rate,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__}
    return separate, info
