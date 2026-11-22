"""Record the live stream exactly as listeners get it: final audio plus the tracked video.

Audio is the processed PCM sent to headphones (after separation/Focus and auto
volume). Video is the annotated camera stream. Separated audio plays about 0.75 s
late, so each frame is placed at its capture time plus the audio delay at that
moment: lips stay in sync with the voice in the file. On stop, ffmpeg joins them
into recordings/stream-<time>/lipsync-stream.mp4 (audio.wav is kept as well).
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

MAX_SECONDS = 15 * 60


class StreamRecorder:
    def __init__(self, root, log=None):
        self.root = Path(root).resolve()  # ffmpeg resolves the frame list relative to itself
        self.log = log or (lambda *args, **kwargs: None)
        self.lock = threading.Lock()
        self.active = False
        self.folder = None
        self.last = None
        self.finishing = False

    # ---- control ----
    def start(self):
        with self.lock:
            if self.active:
                return
            self.folder = self.root / f"stream-{datetime.now():%Y%m%d-%H%M%S}"
            (self.folder / "frames").mkdir(parents=True, exist_ok=True)
            self.wav = None
            self.rate = None
            self.start_sample = self.next_sample = None
            self.t0_us = None
            self.frames = []  # (seconds in file, path)
            self.active = True
        self.log("info", "record", f"stream recording started: {self.folder}")

    def stop(self):
        with self.lock:
            if not self.active:
                return
            self.active = False
            folder, wav, frames, rate = self.folder, self.wav, self.frames, self.rate
            seconds = 0 if rate is None else (self.next_sample - self.start_sample) / rate
            self.finishing = True
        if wav is not None:
            wav.close()
        threading.Thread(target=self._finish, args=(folder, frames, seconds), name="stream-recorder",
                         daemon=True).start()

    def update(self, message):
        enabled = message.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("set_stream_recording needs enabled: true or false")
        self.start() if enabled else self.stop()

    def status(self):
        with self.lock:
            seconds = 0.0
            if self.active and self.rate:
                seconds = (self.next_sample - self.start_sample) / self.rate
            return {"recording": self.active, "seconds": round(seconds, 1), "finishing": self.finishing,
                    "last": str(self.last) if self.last else None}

    # ---- media (called from the capture loop) ----
    def audio(self, pcm, first_sample, capture_us, rate):
        with self.lock:
            if not self.active:
                return
            if self.wav is None:
                self.rate, self.start_sample = rate, first_sample
                self.next_sample = first_sample
                self.t0_us = capture_us - first_sample * 1_000_000 / rate  # phone clock at sample 0
                self.wav = wave.open(str(self.folder / "audio.wav"), "wb")
                self.wav.setnchannels(1)
                self.wav.setsampwidth(2)
                self.wav.setframerate(rate)
            if first_sample > self.next_sample:  # dropped blocks: keep the timeline with silence
                self.wav.writeframes(bytes(2 * (first_sample - self.next_sample)))
                self.next_sample = first_sample
            if first_sample < self.next_sample:
                return
            self.wav.writeframes(pcm)
            self.next_sample += len(pcm) // 2
            if (self.next_sample - self.start_sample) / rate > MAX_SECONDS:
                threading.Thread(target=self.stop, daemon=True).start()

    def video(self, jpeg, capture_us, delay_s):
        with self.lock:
            if not self.active or self.t0_us is None:
                return
            at = (capture_us - self.t0_us) / 1e6 - self.start_sample / self.rate + delay_s
            if at < 0:
                return
            path = self.folder / "frames" / f"{len(self.frames):06d}.jpg"
            path.write_bytes(jpeg)
            self.frames.append((at, path))

    # ---- finish ----
    def _finish(self, folder, frames, seconds):
        try:
            if not frames or seconds <= 0:
                self.log("warn", "record", f"stream recording had no {'video' if seconds > 0 else 'audio'}: {folder}")
                return
            frames = sorted(frames)
            lines = ["ffconcat version 1.0"]
            for k, (at, path) in enumerate(frames):
                start = 0.0 if k == 0 else at  # the first frame also covers the time before it
                end = frames[k + 1][0] if k + 1 < len(frames) else max(seconds, at + 0.04)
                lines += [f"file '{path.as_posix()}'", f"duration {max(end - start, 0.001):.4f}"]
            lines.append(f"file '{frames[-1][1].as_posix()}'")
            (folder / "frames.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            out = folder / "lipsync-stream.mp4"
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                self.log("warn", "record", f"ffmpeg not found: kept audio.wav and frames/ in {folder}")
                self.last = folder
                return
            began = time.perf_counter()
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(folder / "frames.txt"),
                 "-i", str(folder / "audio.wav"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                 "-fps_mode", "cfr", "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                 "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)],
                capture_output=True, text=True)
            if result.returncode:
                self.log("error", "record", f"ffmpeg failed ({result.returncode}): {result.stderr[-400:]}")
                self.last = folder
                return
            shutil.rmtree(folder / "frames", ignore_errors=True)
            (folder / "frames.txt").unlink(missing_ok=True)
            self.last = out
            self.log("info", "record", f"saved {seconds:.1f} s stream recording ({len(frames)} frames) to {out} "
                                       f"in {time.perf_counter() - began:.1f} s")
        except Exception as exc:
            self.log("error", "record", f"stream recording failed: {exc}")
        finally:
            with self.lock:
                self.finishing = False
