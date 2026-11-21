"""Separate a public two-speaker sample or a local WAV into two candidate voices.

Examples:
  python scripts/separate_audio.py --out-dir validation-output/audio-public
  python scripts/separate_audio.py --input recording.wav --device cuda --out-dir validation-output/audio-custom

This is whole-clip offline separation, not lip matching or live inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PIN_FILE = ROOT / "audio-separation.lock.json"


class SeparationError(Exception):
    """An input, asset, dependency, or inference failure with an actionable message."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise SeparationError(f"SHA-256 mismatch for {path.name}; the cached asset does not match its pin.")


def fetch_assets(pins: dict, public_sample: bool):
    from huggingface_hub import hf_hub_download, snapshot_download

    model = pins["model"]
    try:
        directory = Path(snapshot_download(
            repo_id=model["repository"], revision=model["revision"],
            allow_patterns=list(model["files"]),
        ))
        for name, expected in model["files"].items():
            verify_file(directory / name, expected)
        sample = None
        if public_sample:
            sample_pin = pins["sample"]
            sample = Path(hf_hub_download(
                repo_id=sample_pin["repository"], revision=sample_pin["revision"],
                filename=sample_pin["filename"],
            ))
            verify_file(sample, sample_pin["sha256"])
    except SeparationError:
        raise
    except Exception as exc:
        raise SeparationError(
            f"Cannot obtain pinned assets: {exc}. Check connectivity; offline runs require a populated Hugging Face cache."
        ) from exc
    return directory, sample


def prepare_audio(path: Path, sample_rate: int):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        info = sf.info(str(path))
        if info.format not in {"WAV", "WAVEX", "RF64"}:
            raise SeparationError("Input must be a WAV file. Convert other formats to WAV first.")
        data, original_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, OSError) as exc:
        raise SeparationError(f"Cannot read WAV {path}: {exc}") from exc
    if len(data) == 0 or data.shape[1] == 0 or original_rate <= 0:
        raise SeparationError("Input WAV has no audio samples or an invalid sample rate.")
    if not np.isfinite(data).all():
        raise SeparationError("Input WAV contains non-finite samples.")
    original_samples, channels = data.shape
    # Accumulate in float64 to avoid overflowing when averaging float WAV input.
    audio = data.mean(axis=1, dtype=np.float64)
    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, original_rate // divisor)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise SeparationError("Audio conversion produced non-finite samples.")
    # The pinned encoder's kernel is 16 samples. Shorter inputs cannot be encoded.
    if len(audio) < 16:
        raise SeparationError("Input is too short for this model; supply at least 16 samples at 8 kHz.")
    metadata = {
        "filename": path.name, "sha256": sha256(path),
        "original_sample_rate": original_rate, "original_channels": channels,
        "original_samples": original_samples,
        "original_duration_seconds": original_samples / original_rate,
        "prepared_sample_rate": sample_rate, "prepared_samples": len(audio),
        "duration_seconds": len(audio) / sample_rate,
    }
    return audio, metadata


def choose_device(requested: str, torch) -> str:
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise SeparationError(
            "CUDA is unavailable. Install matching CUDA torch/torchaudio builds on the A3000, "
            "or explicitly use --device cpu for a local correctness check."
        )
    return ("cuda" if available else "cpu") if requested == "auto" else requested


def load_model(directory: Path, device: str, pins: dict):
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import FetchConfig, LocalStrategy

    # With no savedir, SpeechBrain reads the already-verified local cache paths.
    # COPY also prevents symlink creation if a future version collects files.
    # Disable its network access: all required files were fetched and pinned above.
    model = SepformerSeparation.from_hparams(
        source=str(directory), savedir=None,
        local_strategy=LocalStrategy.COPY,
        fetch_config=FetchConfig(allow_network=False),
        run_opts={"device": device},
    )
    if (model.hparams.sample_rate != pins["sample_rate"]
            or model.hparams.num_spks != pins["sources"]):
        raise SeparationError("Model configuration does not match the pinned sample rate/source count.")
    model.eval()
    return model


def measure_separation(model, audio, device: str, torch):
    import numpy as np

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def forward():
        with torch.inference_mode():
            batch = torch.from_numpy(audio).unsqueeze(0)
            estimate = model.separate_batch(batch)
            if tuple(estimate.shape) != (1, len(audio), 2):
                raise SeparationError(f"Expected model output (1, {len(audio)}, 2); got {tuple(estimate.shape)}.")
            result = estimate[0].detach().cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise SeparationError("Model produced non-finite output; no result will be published.")
        return result

    synchronize()
    started = time.perf_counter()
    forward()  # One full-clip warm-up, excluded from the measured inference.
    synchronize()
    warmup_ms = (time.perf_counter() - started) * 1000
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    estimates = forward()
    synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return estimates, {
        "warmup_ms": warmup_ms, "warmup_passes": 1,
        "inference_ms": elapsed_ms, "measured_passes": 1,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
        "timing_scope": "One full clip: tensor assembly, device transfer, separation, return to CPU and output validation; excludes downloads, model loading, resampling and WAV export.",
    }


def publish_outputs(output: Path, mixture, candidates, metrics: dict) -> None:
    import numpy as np
    import soundfile as sf

    if output.exists():
        raise SeparationError(f"Output directory already exists: {output}. Choose a fresh directory.")
    if candidates.shape != (len(mixture), 2) or not np.isfinite(candidates).all():
        raise SeparationError("Candidates must be finite and aligned with the mixture.")
    peak = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(candidates))))
    gain = min(1.0, 0.99 / peak) if peak else 1.0
    metrics["export"] = {
        "format": "WAV PCM16", "sample_rate": metrics["sample_rate"],
        "shared_gain": gain, "unscaled_global_peak": peak,
        "candidate_unscaled_peaks": [float(np.max(np.abs(candidates[:, i]))) for i in range(2)],
        "candidate_indices_are_speaker_identities": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lipsync-audio-", dir=output.parent) as temporary:
        stage = Path(temporary)
        rate = metrics["sample_rate"]
        sf.write(str(stage / "mixture.wav"), mixture * gain, rate, subtype="PCM_16")
        for index in range(2):
            sf.write(str(stage / f"candidate_{index}.wav"), candidates[:, index] * gain, rate, subtype="PCM_16")
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)


def run(args) -> dict:
    output = args.out_dir.resolve()
    if output.exists():
        raise SeparationError(f"Output directory already exists: {output}. Choose a fresh directory.")
    if args.input is not None and not args.input.is_file():
        raise SeparationError(f"Input WAV does not exist: {args.input}")
    try:
        import torch
        import soundfile  # Fail before downloading weights if audio dependencies are missing.
        import scipy
    except ImportError as exc:
        raise SeparationError("Install requirements-audio-only.lock.txt in the separate Python 3.11 environment.") from exc
    device = choose_device(args.device, torch)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    pins = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    print(f"Using {device}. Fetching/verifying pinned model assets...", flush=True)
    fetch_start = time.perf_counter()
    directory, sample = fetch_assets(pins, public_sample=args.input is None)
    asset_ms = (time.perf_counter() - fetch_start) * 1000
    source = args.input.resolve() if args.input is not None else sample
    prepare_start = time.perf_counter()
    audio, source_info = prepare_audio(source, pins["model"]["sample_rate"])
    preparation_ms = (time.perf_counter() - prepare_start) * 1000
    print(f"Prepared {source_info['duration_seconds']:.3f}s at 8 kHz. Loading SepFormer-WHAMR...", flush=True)
    load_start = time.perf_counter()
    model = load_model(directory, device, pins["model"])
    if device == "cuda":
        torch.cuda.synchronize()
    loading_ms = (time.perf_counter() - load_start) * 1000
    print("Running one warm-up and one measured whole-clip separation...", flush=True)
    candidates, timings = measure_separation(model, audio, device, torch)
    metrics = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_audio_only", "source": source_info,
        "source_kind": "public_test_mixture" if args.input is None else "local_wav",
        "pins": pins, "sample_rate": pins["model"]["sample_rate"], "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "cuda_version": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
        "python": sys.version, "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                     if d.metadata.get("Name")},
        "asset_fetch_and_verification_ms": asset_ms, "preparation_ms": preparation_ms,
        "model_loading_ms": loading_ms, **timings,
        "real_time_factor": timings["inference_ms"] / (source_info["duration_seconds"] * 1000),
        "quality_status": "pending_listening_review",
        "streaming_suitability": "not_evaluated",
    }
    publish_outputs(output, audio, candidates, metrics)
    print(f"Saved {output}")
    print(f"Inference {metrics['inference_ms']:.1f} ms; RTF {metrics['real_time_factor']:.3f}; "
          "two anonymous candidate voices. Listening review still required.")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Optional local WAV; omitted uses the pinned public sample")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for this run's four artifacts")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (SeparationError, OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Separation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
