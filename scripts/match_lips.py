"""Separate recorded speech, associate two visible faces with voices, and render a demo.

Example for a split-screen debate:
  python scripts/match_lips.py --video video-samples/debate.mp4 --roi 0 640 1080 640 --out-dir validation-output/debate

This is an offline heuristic experiment, not a lip-reading or streaming model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from lip_matching import FaceTracker, MatchConfig, analyze, selected_audio
import separate_audio

ROOT = Path(__file__).resolve().parents[1]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              415, 310, 311, 312, 13, 82, 81, 80, 191]


class MatchError(Exception):
    pass


def command(arguments):
    result = subprocess.run(arguments, capture_output=True)
    if result.returncode:
        raise MatchError(f"{Path(arguments[0]).name}: {result.stderr.decode(errors='replace')[-2000:]}")
    return result.stdout


def prepare_video(video, stage, roi, fps):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise MatchError("Install FFmpeg and put ffmpeg/ffprobe on PATH.")
    metadata = json.loads(command([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)]))
    streams = metadata.get("streams", [])
    image = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    sound = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if image is None or sound is None:
        raise MatchError("Video must contain both picture and audio.")
    container = metadata.get("format", {})

    def bounds(stream):
        start = float(stream.get("start_time", container.get("start_time", 0)))
        end = (start + float(stream["duration"]) if "duration" in stream else
               float(container.get("start_time", 0)) + float(container.get("duration", 0)))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise MatchError("Cannot determine finite media timestamps.")
        return start, end

    video_start, video_end = bounds(image)
    audio_start, audio_end = bounds(sound)
    start, end = max(video_start, audio_start), min(video_end, audio_end)
    frames = math.floor((end-start)*fps + 1e-6)
    if frames < 2 * fps:
        raise MatchError("Need at least two seconds of overlapping audio/video; 20–30 seconds is preferable.")
    duration = frames / fps
    width, height = int(image["width"]), int(image["height"])
    rotation = float(image.get("tags", {}).get("rotate", 0))
    for side in image.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > .001:
        raise MatchError("Export arbitrary-angle rotated footage upright first.")
    if round(rotation) % 180:
        width, height = height, width
    if roi is None:
        roi = [0, 0, width, height]
    x, y, w, h = roi
    if min(x, y) < 0 or min(w, h) < 2 or x+w > width or y+h > height:
        raise MatchError(f"ROI is outside the upright {width}x{height} image.")
    out_width = min(960, w) // 2 * 2
    out_height = max(2, round(h * out_width / w / 2) * 2)
    prepared = stage / "prepared.mp4"
    audio = stage / "input.wav"
    base = [ffmpeg, "-v", "error", "-nostdin", "-copyts", "-i", str(video)]
    command(base + [
        "-map", f"0:{image['index']}", "-an", "-vf",
        f"trim=start={start:.9f}:end={start+duration:.9f},setpts=PTS-({start:.9f})/TB,"
        f"fps={fps}:start_time=0,crop={w}:{h}:{x}:{y}:exact=1,scale={out_width}:{out_height}",
        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-threads", "2", str(prepared),
    ])
    command(base + [
        "-map", f"0:{sound['index']}", "-vn", "-af",
        f"atrim=start={start:.9f}:end={start+duration:.9f},asetpts=PTS-({start:.9f})/TB,"
        "aresample=8000:async=1:first_pts=0", "-ac", "1", "-ar", "8000", "-c:a", "pcm_f32le", str(audio),
    ])
    return prepared, audio, {
        "filename": video.name, "sha256": separate_audio.sha256(video),
        "video_start_seconds": video_start, "audio_start_seconds": audio_start,
        "common_start_seconds": start, "retained_end_seconds": start+duration,
        "duration_seconds": duration, "frames": frames, "fps": fps,
        "roi_pixels": roi, "prepared_dimensions": [out_width, out_height],
    }


def track_faces(prepared, model_path, fps):
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(prepared))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2,
    )
    tracker = FaceTracker()
    traces, boxes = [], []
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(index*1000/fps))
                detections = []
                for face in result.face_landmarks:
                    points = np.array([(p.x*width, p.y*height) for p in face])
                    low, high = points.min(axis=0), points.max(axis=0)
                    mouth_width = np.linalg.norm(points[61]-points[291])
                    if mouth_width < 3:
                        continue
                    mouth = points[INNER_LIPS]
                    detections.append({
                        "center": ((low+high)/2 / [width, height]).tolist(),
                        "bbox": np.concatenate((low, high)).round().astype(int).tolist(),
                        "mouth": mouth.round().astype(int).tolist(),
                        "opening": float(np.linalg.norm(points[13]-points[14])/mouth_width),
                    })
                assigned = tracker.assign(detections)
                traces.append([item["opening"] if item else np.nan for item in assigned])
                boxes.append(assigned)
                index += 1
                if index % (fps*5) == 0:
                    print(f"Tracked {index/fps:.0f}s of video...", flush=True)
    finally:
        cap.release()
    return np.asarray(traces), boxes


def get_separation(audio_path, stage, cached, device):
    import soundfile as sf

    destination = stage / "separation"
    if cached is None:
        separate_audio.run(argparse.Namespace(input=audio_path, device=device, out_dir=destination))
    else:
        required = ["mixture.wav", "candidate_0.wav", "candidate_1.wav", "metrics.json"]
        if not all((cached / name).is_file() for name in required):
            raise MatchError("--separation-dir needs the four files from a completed separate_audio run.")
        source, rate = sf.read(str(audio_path), dtype="float32")
        mixture, mix_rate = sf.read(str(cached / "mixture.wav"), dtype="float32")
        report = json.loads((cached / "metrics.json").read_text(encoding="utf-8"))
        try:
            gain = float(report["export"]["shared_gain"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MatchError("Cached metrics need a valid export.shared_gain from separate_audio.py.") from exc
        if (rate != mix_rate or source.shape != mixture.shape
                or not np.allclose(source*gain, mixture, atol=2/32768, rtol=0)):
            raise MatchError("Cached separation is not aligned with this video's prepared audio.")
        destination.mkdir()
        for name in required:
            shutil.copy2(cached / name, destination / name)
    mixture, rate = sf.read(str(destination / "mixture.wav"), dtype="float32")
    sources = []
    for index in range(2):
        samples, source_rate = sf.read(str(destination / f"candidate_{index}.wav"), dtype="float32")
        if source_rate != rate or samples.shape != mixture.shape or not np.isfinite(samples).all():
            raise MatchError("Candidate outputs are not finite and sample-aligned.")
        sources.append(samples)
    return mixture, np.column_stack(sources), rate


def render_demo(stage, prepared, boxes, records, selections, statuses, fps, duration):
    import cv2

    cap = cv2.VideoCapture(str(prepared))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    silent = stage / "annotated-silent.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height+112))
    if not writer.isOpened():
        cap.release()
        raise MatchError("OpenCV could not create the annotated video.")
    record_index = -1
    index = 0
    colors = [(80, 220, 140), (240, 190, 70)]
    try:
        while index < len(boxes):
            ok, frame = cap.read()
            if not ok:
                break
            canvas = np.zeros((height+112, width, 3), np.uint8)
            canvas[:height] = frame
            while record_index+1 < len(records) and records[record_index+1]["time_seconds"] <= index/fps + 1e-6:
                record_index += 1
            for face in range(2):
                item = boxes[index][face]
                if item:
                    x0, y0, x1, y1 = item["bbox"]
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), colors[face], 2)
                    cv2.polylines(canvas, [np.array(item["mouth"], np.int32)], True, colors[face], 1)
                    cv2.putText(canvas, f"Face {face}", (max(0,x0), max(24,y0-8)), cv2.FONT_HERSHEY_SIMPLEX, .65, colors[face], 2)
                chosen = selections[index, face]
                label = f"Voice {chosen}" if chosen >= 0 else "uncertain / muted"
                scores = records[record_index]["scores"][face] if record_index >= 0 else [None,None]
                score_text = " / ".join("--" if value is None else f"{value:.2f}" for value in scores)
                state = "tentative" if statuses[index,face] == "matched" else statuses[index,face]
                text = f"Face {face}: {label} | {state} | r0/r1 {score_text}"
                cv2.putText(canvas, text, (12, height+32+face*34), cv2.FONT_HERSHEY_SIMPLEX, .54, colors[face], 1, cv2.LINE_AA)
            cv2.putText(canvas, "Offline lip-rhythm association | scores are correlations, not probabilities", (12,height+99), cv2.FONT_HERSHEY_SIMPLEX, .43, (210,210,210), 1, cv2.LINE_AA)
            writer.write(canvas)
            index += 1
    finally:
        cap.release()
        writer.release()
    ffmpeg = shutil.which("ffmpeg")
    command([ffmpeg, "-v", "error", "-nostdin", "-i", str(silent), "-i", str(stage/"separation"/"mixture.wav"),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
             "-threads", "2", "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-c:a", "aac", "-ar", "48000", "-b:a", "128k", "-t", str(duration), "-movflags", "+faststart", str(stage/"annotated.mp4")])
    for face in range(2):
        command([ffmpeg, "-v", "error", "-nostdin", "-i", str(stage/"annotated.mp4"), "-i", str(stage/f"face_{face}.wav"),
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-c:a", "aac", "-ar", "48000", "-b:a", "128k",
                 "-t", str(duration), "-movflags", "+faststart", str(stage/f"face_{face}.mp4")])
    silent.unlink()


def run(args):
    import soundfile as sf

    video, output = args.video.resolve(), args.out_dir.resolve()
    if not video.is_file():
        raise MatchError(f"Video does not exist: {video}")
    if output.exists():
        raise MatchError("Choose a fresh --out-dir; existing results will not be overwritten.")
    model_path = args.face_model.resolve()
    if not model_path.is_file():
        raise MatchError(f"Face landmark model is missing: {model_path}")
    config = MatchConfig(min_correlation=args.min_score, min_margin=args.margin)
    if not 0 <= config.min_correlation <= 1 or not 0 <= config.min_margin <= 2:
        raise MatchError("--min-score must be 0..1 and --margin 0..2.")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".lipsync-match-", dir=output.parent) as temporary:
        stage = Path(temporary)
        print("Preparing synchronized audio/video...", flush=True)
        prepared, audio, media = prepare_video(video, stage, args.roi, config.fps)
        print("Separating the two audio candidates...", flush=True)
        mixture, candidates, rate = get_separation(audio, stage, args.separation_dir, args.device)
        print("Tracking visible faces and measuring mouth opening...", flush=True)
        mouths, boxes = track_faces(prepared, model_path, config.fps)
        if not np.isfinite(mouths).any():
            raise MatchError("No usable mouth landmarks were detected. Crop the visible faces with --roi or use clearer footage.")
        frames = min(len(mouths), len(candidates)//(rate//config.fps), media["frames"])
        if frames < config.fps * 2:
            raise MatchError("Too few decoded, synchronized frames for matching.")
        mouths, boxes = mouths[:frames], boxes[:frames]
        media["processed_frames"] = frames
        candidates = candidates[:frames*(rate//config.fps)]
        print("Matching lip rhythms and running the shifted-video negative control...", flush=True)
        selections, statuses, records, summary = analyze(mouths, candidates, rate, config)
        selected = selected_audio(candidates, selections, rate, config.fps)
        for face in range(2):
            sf.write(str(stage/f"face_{face}.wav"), selected[:,face], rate, subtype="PCM_16")
        (stage/"tracking.json").write_text(json.dumps({
            "fps": config.fps, "mouth_opening": [[None if not np.isfinite(v) else float(v) for v in row] for row in mouths],
            "faces": boxes,
        }, allow_nan=False)+"\n", encoding="utf-8")
        (stage/"matches.json").write_text(json.dumps(records, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print("Rendering the original-audio and per-face demonstration videos...", flush=True)
        render_demo(stage, prepared, boxes, records, selections, statuses, config.fps, frames/config.fps)
        report = {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": media, "face_model_sha256": separate_audio.sha256(model_path),
            "tracking": "Two geometry-based slots, initially ordered left-to-right; not biometric identity tracking across cuts.",
            "duration_seconds": frames/config.fps, "sample_rate": rate, **summary,
            "uncertain_audio_policy": "Mute after a one-second association hold; 20 ms crossfades at transitions.",
            "preview_audio": "Video previews use FFmpeg loudness normalization for listening only; analysis WAVs are unchanged.",
            "total_processing_seconds": time.perf_counter()-started,
        }
        (stage/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        stage.rename(output)
    print(f"Saved {output}")
    for face in report["faces"]:
        print(f"Face {face['face_id']}: detected {face['detection_fraction']:.1%}, matched {face['matched_fraction']:.1%}, "
              f"held {face['held_fraction']:.1%}, uncertain {face['uncertain_fraction']:.1%}")
    print("Shifted-video timing check:", report["negative_control"]["timing_check"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X","Y","WIDTH","HEIGHT"), help="Optional upright video crop, in pixels")
    parser.add_argument("--face-model", type=Path, default=ROOT/"face_landmarker.task")
    parser.add_argument("--separation-dir", type=Path, help="Reuse existing separation after checking audio alignment")
    parser.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    parser.add_argument("--min-score", type=float, default=.30)
    parser.add_argument("--margin", type=float, default=.10)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (MatchError, separate_audio.SeparationError, OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Lip matching failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
