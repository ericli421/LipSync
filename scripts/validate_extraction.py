"""Validate pinned Falcon Small extraction on a stationary, recorded target.

Run with --help. See docs/model-validation.md for environment setup and the
listening protocol. Timings measure offline window processing, not live latency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import wave

try:
    import numpy as np
except ImportError:
    np = None  # --help and argument errors work before installing dependencies.

ROOT = Path(__file__).resolve().parents[1]


try:
    from scripts.falcon_adapter import (ValidationError, load_engine, benchmark, mouth_window,
                                        stats, command, sha256, SR, FPS, SIDE, WINDOW, HOP,
                                        GUARD, MIN_WINDOWS, GATE_MS)
except ModuleNotFoundError:  # Direct python scripts/validate_extraction.py invocation.
    from falcon_adapter import (ValidationError, load_engine, benchmark, mouth_window,
                                stats, command, sha256, SR, FPS, SIDE, WINDOW, HOP,
                                GUARD, MIN_WINDOWS, GATE_MS)


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
