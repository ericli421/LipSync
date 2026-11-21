"""Validate pinned Falcon Small extraction on a stationary, recorded target.

Run with --help. See docs/model-validation.md for environment setup and the
listening protocol. Timings measure offline window processing, not live latency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import wave

try:
    import numpy as np
except ImportError:
    np = None  # --help and argument errors work before installing dependencies.

ROOT = Path(__file__).resolve().parents[1]
SR, FPS, SIDE = 16_000, 25, 96
WINDOW, HOP, GUARD = 32_000, 8_000, 1_280
MIN_WINDOWS, GATE_MS = 30, 400.0


class ValidationError(Exception):
    """An actionable input, dependency, or runtime failure."""


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    start: float
    end: float
    video_start: float
    audio_start: float
    rotation: int
    video_index: int
    audio_index: int


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


def probe(video: Path, ffprobe: str) -> MediaInfo:
    data = json.loads(command([
        ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)
    ]))
    streams = data.get("streams", [])
    picture = next((s for s in streams if s.get("codec_type") == "video"
                    and not s.get("disposition", {}).get("attached_pic")), None)
    sound = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if picture is None or sound is None:
        raise ValidationError("Input must contain both a video stream and an audio stream.")
    container = data.get("format", {})

    def start_end(stream):
        start = float(stream.get("start_time", container.get("start_time", 0)))
        duration = stream.get("duration")
        end = (start + float(duration) if duration is not None else
               float(container.get("start_time", 0)) + float(container.get("duration", 0)))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValidationError("Cannot determine finite stream timestamps/duration; remux the clip first.")
        return start, end

    video_start, video_end = start_end(picture)
    audio_start, audio_end = start_end(sound)
    start, end = max(video_start, audio_start), min(video_end, audio_end)
    if end - start < 2:
        raise ValidationError("Need at least two seconds of overlapping audio and video; use a 20-second clip.")
    rotation = float(picture.get("tags", {}).get("rotate", 0))
    for side_data in picture.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = float(side_data["rotation"])
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 0.001:
        raise ValidationError("Only 0/90/180/270-degree video rotation is supported; export an upright clip.")
    rotation = round(rotation) % 360
    width, height = int(picture["width"]), int(picture["height"])
    if rotation in (90, 270):
        width, height = height, width
    return MediaInfo(width, height, start, end, video_start, audio_start, rotation,
                     int(picture["index"]), int(sound["index"]))


def prepare_media(video: Path, box: tuple[int, int, int], ffmpeg: str,
                  ffprobe: str):
    info = probe(video, ffprobe)
    x, y, size = box
    if x < 0 or y < 0 or size <= 0 or x + size > info.width or y + size > info.height:
        raise ValidationError(
            f"Mouth box {box} is outside the upright {info.width}x{info.height} video; "
            "use nonnegative X/Y and a positive square SIZE."
        )
    start, end = f"{info.start:.9f}", f"{info.end:.9f}"
    # Preserve input PTS. Subtract the SAME origin in both paths, rather than
    # independently using STARTPTS and accidentally erasing a real A/V offset.
    base = [ffmpeg, "-v", "error", "-nostdin", "-copyts", "-i", str(video)]
    audio_filter = (f"atrim=start={start}:end={end},asetpts=PTS-({start})/TB,"
                    f"aresample={SR}:async=1:first_pts=0")
    raw_audio = command(base + [
        "-map", f"0:{info.audio_index}", "-af", audio_filter, "-ac", "1", "-ar", str(SR),
        "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"
    ])
    count = math.floor((info.end - info.start) * FPS + 1e-6)
    video_filter = (f"trim=start={start}:end={end},setpts=PTS-({start})/TB,"
                    f"fps={FPS}:start_time=0:round=near,"
                    f"crop={size}:{size}:{x}:{y}:exact=1,"
                    f"scale={SIDE}:{SIDE}:flags=bicubic,format=gray")
    # FFmpeg autorotates phone video before applying the crop filter.
    raw_video = command(base + [
        "-map", f"0:{info.video_index}", "-an", "-vf", video_filter, "-frames:v", str(count),
        "-c:v", "rawvideo", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"
    ])
    if len(raw_audio) % 4 or len(raw_video) % (SIDE * SIDE):
        raise ValidationError("FFmpeg returned an incomplete audio sample or mouth frame.")
    audio = np.frombuffer(raw_audio, dtype="<f4").astype(np.float32)
    mouths = np.frombuffer(raw_video, dtype=np.uint8).reshape(-1, SIDE, SIDE)
    frames = min(len(mouths), len(audio) // (SR // FPS), count)
    if frames < WINDOW * FPS // SR:
        raise ValidationError("Decoded input is shorter than two synchronized seconds.")
    audio, mouths = audio[:frames * (SR // FPS)], mouths[:frames]
    if not np.isfinite(audio).all():
        raise ValidationError("Decoded audio contains non-finite samples.")
    return audio, mouths, info


def load_engine(falcon_dir: Path, device: str, pins: dict):
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


def write_wav(path: Path, samples):
    pcm = (np.clip(samples, -1, 1) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def package_versions() -> dict:
    return {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")}


def nvidia_driver_information() -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        return command([executable, "--query-gpu=index,name,driver_version,memory.total",
                        "--format=csv,noheader"]).decode().strip()
    except ValidationError:
        return None  # Driver metadata must not discard an otherwise valid run.


def run(args) -> dict:
    if np is None:
        raise ValidationError("NumPy is missing. Install requirements-model.lock.txt in the Python 3.11 environment.")
    video, falcon_dir, output = args.video.resolve(), args.falcon_dir.resolve(), args.out_dir.resolve()
    if not video.is_file():
        raise ValidationError(f"Input video does not exist: {video}")
    if output.exists():
        raise ValidationError(f"Output directory already exists: {output}. Choose a fresh directory for each run.")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValidationError("Install FFmpeg and put both ffmpeg and ffprobe on PATH.")
    pins = json.loads((ROOT / "model-validation.lock.json").read_text(encoding="utf-8"))
    print("Preparing synchronized audio and mouth crops...", flush=True)
    started = time.perf_counter()
    audio, mouths, media = prepare_media(video, tuple(args.mouth_box), ffmpeg, ffprobe)
    preparation_ms = (time.perf_counter() - started) * 1000
    print(f"Prepared {len(audio) / SR:.2f}s. Loading pinned Falcon Small on {args.device}...", flush=True)
    runtime, slicer, torch, checkpoint = load_engine(falcon_dir, args.device, pins)
    print("Warming up and measuring two-second windows at 500 ms hops...", flush=True)
    mixture, target, metrics = benchmark(audio, mouths, runtime, slicer, torch)
    metrics.update({
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pins": pins, "checkpoint_sha256": sha256(checkpoint),
        "device": runtime.device,
        "gpu": torch.cuda.get_device_name() if runtime.device == "cuda" else None,
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if runtime.device == "cuda" else None,
        "nvidia_smi": nvidia_driver_information() if runtime.device == "cuda" else None,
        "cuda_version": torch.version.cuda, "python": sys.version, "platform": platform.platform(),
        "packages": package_versions(), "ffmpeg": command([ffmpeg, "-version"]).decode().splitlines()[0],
        "source": {"filename": video.name, "sha256": sha256(video),
                   "upright_dimensions": [media.width, media.height], "rotation_degrees": media.rotation,
                   "mouth_box": args.mouth_box, "video_start_seconds": media.video_start,
                   "audio_start_seconds": media.audio_start,
                   "common_start_seconds": media.start, "common_end_seconds": media.end,
                   "retained_end_seconds": media.start + len(audio) / SR},
        "output_source_interval_seconds": [media.start + n / SR for n in metrics["output_sample_range"]],
        "preparation_ms": preparation_ms, "sample_rate": SR, "video_fps": FPS,
        "mouth_frame_count": len(mouths),
    })
    # Publish a complete run only; no previous outputs can be mistaken for a new run.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lipsync-validation-", dir=output.parent) as temporary:
        stage = Path(temporary)
        write_wav(stage / "mixture.wav", mixture)
        write_wav(stage / "target.wav", target)
        command([
            ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pixel_format", "gray",
            "-video_size", f"{SIDE}x{SIDE}", "-framerate", str(FPS), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(stage / "mouth_preview.mp4")
        ], input_bytes=mouths.tobytes())
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    print(f"Saved {output}")
    print(f"{metrics['window_count']} windows; processing p95 {metrics['processing']['p95_ms']:.1f} ms; "
          f"{metrics['throughput_status']}; listening review still required.")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--falcon-dir", type=Path, required=True, help="Pinned Falcon repository root")
    parser.add_argument("--video", type=Path, required=True, help="Recorded video with mixed microphone audio")
    parser.add_argument("--mouth-box", nargs=3, type=int, required=True, metavar=("X", "Y", "SIZE"),
                        help="Fixed square mouth crop in upright decoded video pixels")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Default: cuda; CPU runs never pass the GPU integration gate")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for this run's four artifacts")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValidationError, OSError, ValueError, RuntimeError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    return 0  # Completed measurements are successful even when the gate fails.


if __name__ == "__main__":
    raise SystemExit(main())
