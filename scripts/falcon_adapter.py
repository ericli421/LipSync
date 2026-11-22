"""Reusable pinned Falcon Small adapter; no live-server or model imports at import time.

Audio is mono float32 at 16 kHz; mouths are synchronized uint8 25 fps 96x96.
Use one loaded adapter and a fresh upstream slicer for each independent branch.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

try:
    import numpy as np
except ImportError:
    np = None

SR, FPS, SIDE = 16_000, 25, 96
WINDOW, HOP, GUARD = 32_000, 8_000, 1_280
MIN_WINDOWS, GATE_MS = 30, 400.0

class ValidationError(Exception):
    """An actionable media, dependency, or model failure."""

def command(args: list[str], *, input_bytes: bytes | None = None) -> bytes:
    """Run an argument vector, never a shell command assembled from user paths."""
    try:
        result = subprocess.run(args, input=input_bytes, capture_output=True, check=False)
    except OSError as exc:
        raise ValidationError(f"Cannot run {args[0]}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-2500:]
        raise ValidationError(f"{Path(args[0]).name} failed ({result.returncode}): {detail}")
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_engine(falcon_dir: Path, device: str, pins: dict):
    if device not in ("cpu", "cuda"):
        raise ValidationError("Choose an explicit cpu or cuda device.")
    studio = falcon_dir / "studio"
    if not (studio / "falcon_runtime.py").is_file():
        raise ValidationError("--falcon-dir must point to the pinned Falcon checkout, not its studio folder.")
    revision = command(["git", "-C", str(falcon_dir), "rev-parse", "HEAD"]).decode().strip()
    if revision != pins["falcon_commit"]:
        raise ValidationError(f"Wrong Falcon commit: {revision}. Check out {pins['falcon_commit']}.")
    dirty = command(["git", "-C", str(falcon_dir), "status", "--porcelain",
                     "--untracked-files=no", "--", "studio"])
    if dirty.strip():
        raise ValidationError("Falcon's tracked Studio files have local changes; use a clean pinned checkout.")
    try:
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ValidationError("Install PyTorch and requirements-model.lock.txt in the Python 3.11 environment.") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise ValidationError(
            "CUDA is unavailable. Run on the A3000 with a CUDA-enabled PyTorch build, "
            "or explicitly use --device cpu for a correctness check (not GPU validation)."
        )
    try:
        checkpoint = Path(hf_hub_download(
            repo_id=pins["model_repository"], filename=pins["checkpoint_filename"],
            revision=pins["model_revision"]
        ))
    except Exception as exc:
        raise ValidationError(f"Cannot obtain the pinned checkpoint: {exc}. Check connectivity or the Hugging Face cache.") from exc
    if sha256(checkpoint) != pins["checkpoint_sha256"]:
        raise ValidationError("Checkpoint SHA-256 does not match model-validation.lock.json; do not use this download.")
    sys.path.insert(0, str(studio.resolve()))
    try:
        runtime_module = importlib.import_module("falcon_runtime")
        spec = importlib.util.spec_from_file_location("lipsync_falcon_server", studio / "server.py")
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        runtime = runtime_module.FalconRuntime("small", checkpoint_path=checkpoint).load(device)
    except ImportError as exc:
        raise ValidationError(f"Falcon dependency import failed: {exc}. Install requirements-model.lock.txt.") from exc
    if runtime.device != device or not runtime.ready:
        raise ValidationError(f"Requested {device}, but runtime is not ready on that device.")
    # The upstream implementation is reused directly: no copied model or join code.
    return runtime, server.OutputSlicer(), torch, checkpoint


def mouth_window(mouths, start_sample: int):
    # Samples are selected at the center of each 40 ms bin. Resolve exact
    # nearest-frame ties toward the earlier frame, as Studio does. In particular,
    # odd 500 ms hops start halfway between frames: never stride by a fixed 12/13.
    first = (start_sample * FPS + SR - 1) // SR
    result = mouths[first:first + 50]
    if result.shape != (50, SIDE, SIDE):
        raise ValidationError("Not enough synchronized mouth frames for this audio window.")
    return np.ascontiguousarray(result)


def stats(values: list[float]) -> dict:
    return {"p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values)),
            "mean_ms": float(np.mean(values))}


def benchmark(audio, mouths, runtime, slicer, torch):
    if (not isinstance(audio, np.ndarray) or audio.dtype != np.float32
            or audio.ndim != 1 or not audio.flags.c_contiguous
            or len(audio) < WINDOW or not np.isfinite(audio).all()):
        raise ValidationError("Audio must be finite contiguous mono float32 with at least two seconds at 16 kHz.")
    if (not isinstance(mouths, np.ndarray) or mouths.dtype != np.uint8
            or mouths.ndim != 3 or mouths.shape[1:] != (SIDE, SIDE)):
        raise ValidationError("Mouths must be uint8 frames shaped (N, 96, 96) at 25 fps.")
    # Check the final full window before any expensive model work.
    last_endpoint = WINDOW + ((len(audio) - WINDOW) // HOP) * HOP
    mouth_window(mouths, last_endpoint - WINDOW)
    def synchronize():
        if runtime.device == "cuda":
            torch.cuda.synchronize()

    # Warm up real shapes, in addition to the runtime's own zero-input warm-up.
    for _ in range(3):
        runtime.infer(np.ascontiguousarray(audio[:WINDOW]), mouth_window(mouths, 0))
    synchronize()
    if runtime.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inference_times, processing_times, chunks = [], [], []
    clipped = 0
    endpoints = range(WINDOW, len(audio) + 1, HOP)
    last_end = None
    for endpoint in endpoints:
        synchronize()
        total_start = time.perf_counter()
        block = np.ascontiguousarray(audio[endpoint - WINDOW:endpoint])
        frames = mouth_window(mouths, endpoint - WINDOW)
        synchronize()
        infer_start = time.perf_counter()
        prediction = runtime.infer(block, frames)
        synchronize()
        inference_times.append((time.perf_counter() - infer_start) * 1000)
        if np.asarray(prediction).shape != (WINDOW,) or not np.isfinite(prediction).all():
            raise ValidationError("Falcon produced non-finite or incorrectly sized output; extraction aborted.")
        chunk, _, clip_count = slicer.take(prediction, endpoint)
        if np.asarray(chunk).shape != (HOP,) or not np.isfinite(chunk).all():
            raise ValidationError("Falcon slicer produced non-finite or incorrectly sized output.")
        chunks.append(chunk)
        clipped += clip_count
        synchronize()
        processing_times.append((time.perf_counter() - total_start) * 1000)
        last_end = endpoint
    if not chunks:
        raise ValidationError("No complete two-second processing window is available.")
    target = np.concatenate(chunks)
    first_sample = WINDOW - GUARD - HOP
    final_sample = last_end - GUARD
    mixture = audio[first_sample:final_sample]
    if len(mixture) != len(target):
        raise ValidationError("Extracted output and comparison mixture are not sample-aligned.")
    processing = stats(processing_times)
    if runtime.device != "cuda":
        gate = "not_validated_cpu"
    elif len(chunks) < MIN_WINDOWS:
        gate = "insufficient_windows"
    elif processing["p95_ms"] >= GATE_MS:
        gate = "throughput_insufficient"
    else:
        gate = "throughput_gate_passed"
    metrics = {
        "window_count": len(chunks), "warmup_windows_excluded": 3,
        "inference": stats(inference_times), "processing": processing,
        "hop_budget_ms": HOP / SR * 1000, "integration_gate_p95_ms": GATE_MS,
        "minimum_windows": MIN_WINDOWS, "throughput_status": gate,
        "quality_status": "pending_listening_review",
        "conclusion": "throughput_insufficient" if gate == "throughput_insufficient" else "validation_incomplete",
        "processing_overruns": sum(t >= HOP / SR * 1000 for t in processing_times),
        "inference_ms_by_window": inference_times, "processing_ms_by_window": processing_times,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if runtime.device == "cuda" else None,
        "target_clipped_samples": int(clipped),
        "mixture_clipped_samples": int(np.count_nonzero(np.abs(mixture) > 1)),
        "output_sample_range": [first_sample, final_sample],
        "output_duration_seconds": len(target) / SR,
        "timing_scope": "Decoded-window assembly, inference and output slicing; excludes file decoding, encoding, network and playback.",
        "windows": {"samples": WINDOW, "hop_samples": HOP, "right_context_samples": GUARD},
    }
    return mixture, target, metrics


class FalconAdapter:
    """Load once, run independent recordings sequentially (not thread-safe)."""
    def __init__(self, runtime, slicer_factory, torch, checkpoint):
        self.runtime = runtime
        self.slicer_factory = slicer_factory
        self.torch = torch
        self.checkpoint = checkpoint

    @classmethod
    def load(cls, falcon_dir, device, pins):
        runtime, slicer, torch, checkpoint = load_engine(falcon_dir, device, pins)
        return cls(runtime, type(slicer), torch, checkpoint)

    def extract(self, audio, mouths):
        return benchmark(audio, mouths, self.runtime, self.slicer_factory(), self.torch)
