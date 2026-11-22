"""Pinned pretrained separators for live test mode. torch is imported lazily so
the relay runs without it; install requirements-audio-only.txt to enable."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
CKPTS = ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt")

MODELS = {
    "sepformer-whamr": {
        "label": "SepFormer-WHAMR (noise + reverb; needs GPU)",
        "pin": None,  # read from audio-separation.lock.json
    },
    "resepformer-wsj02mix": {
        "label": "ReSepFormer (fast on CPU; clean-speech trained)",
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


def _separate_audio():
    sys.path.insert(0, str(ROOT / "scripts"))
    import separate_audio
    return separate_audio


def resolve(name, cuda):
    if name == "auto":
        return "sepformer-whamr" if cuda else "resepformer-wsj02mix"
    if name not in MODELS:
        raise ValueError(f"unknown model {name}")
    return name


def load(name="auto", device="auto"):
    """Return (separate(window float32 [n]) -> float [n, 2], info dict)."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Install torch/torchaudio and "
                           "requirements-audio-only.txt (CUDA wheels on the A3000).") from exc
    cuda = torch.cuda.is_available()
    if device == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False; install the cu126 torch wheels.")
    device = ("cuda" if cuda else "cpu") if device == "auto" else device
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
    lock = threading.Lock()

    def separate(window):
        with lock, torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0).to(device)
            estimate = model.separate_batch(batch)[0]
            return estimate.float().cpu().numpy()

    separate(np.random.default_rng(0).normal(0, 0.05, 12000).astype(np.float32))  # warm-up
    info = {"model": name, "label": MODELS[name]["label"], "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__}
    return separate, info
