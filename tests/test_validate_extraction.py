"""Recorded-media and benchmark contract tests; no model download or GPU needed."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_extraction", ROOT / "scripts" / "validate_extraction.py")
validation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validation
spec.loader.exec_module(validation)


class IdentityRuntime:
    device = "cpu"

    def __init__(self):
        self.first_frames = []

    def infer(self, audio, frames):
        self.first_frames.append(int(frames[0, 0, 0]))
        return audio.copy()


class StubSlicer:
    """External slicer contract; upstream crossfade is exercised in the smoke run."""
    def take(self, prediction, endpoint):
        return prediction[22720:30720].copy(), True, 0


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.linspace(-0.8, 0.8, 20 * validation.SR, dtype=np.float32)
        self.mouths = np.broadcast_to(
            (np.arange(20 * validation.FPS) % 256).astype(np.uint8)[:, None, None],
            (20 * validation.FPS, 96, 96),
        )

    def test_half_frame_hops_use_correct_frames(self):
        self.assertEqual(int(validation.mouth_window(self.mouths, 0)[0, 0, 0]), 0)
        self.assertEqual(int(validation.mouth_window(self.mouths, 8000)[0, 0, 0]), 13)
        self.assertEqual(int(validation.mouth_window(self.mouths, 16000)[0, 0, 0]), 25)
        self.assertEqual(int(validation.mouth_window(self.mouths, 24000)[0, 0, 0]), 38)

    def test_sample_aligned_output_and_cpu_cannot_pass_gate(self):
        runtime = IdentityRuntime()
        mixture, target, metrics = validation.benchmark(self.audio, self.mouths, runtime, StubSlicer(), None)
        np.testing.assert_array_equal(mixture, target)
        np.testing.assert_array_equal(mixture, self.audio[22720:318720])
        self.assertEqual(metrics["window_count"], 37)
        self.assertEqual(len(runtime.first_frames), 40)  # three warm-ups excluded
        self.assertEqual(metrics["throughput_status"], "not_validated_cpu")
        self.assertEqual(metrics["conclusion"], "validation_incomplete")

    def test_nan_output_is_rejected(self):
        runtime = IdentityRuntime()
        runtime.infer = lambda audio, frames: np.full(32000, np.nan, dtype=np.float32)
        with self.assertRaisesRegex(validation.ValidationError, "non-finite"):
            validation.benchmark(self.audio, self.mouths, runtime, StubSlicer(), None)

    def test_cuda_gate_requires_enough_windows_and_synchronized_timing(self):
        runtime = IdentityRuntime()
        runtime.device = "cuda"
        calls = []
        cuda = SimpleNamespace(synchronize=lambda: calls.append("sync"),
                               reset_peak_memory_stats=lambda: calls.append("reset"),
                               max_memory_allocated=lambda: 1234)
        _, _, metrics = validation.benchmark(self.audio[:40000], self.mouths, runtime,
                                              StubSlicer(), SimpleNamespace(cuda=cuda))
        self.assertEqual(metrics["throughput_status"], "insufficient_windows")
        self.assertGreaterEqual(calls.count("sync"), 1 + 2 * 4)
        self.assertEqual(metrics["peak_cuda_memory_bytes"], 1234)

    def test_cuda_gate_and_overruns_use_total_processing_time(self):
        runtime = IdentityRuntime()
        runtime.device = "cuda"
        cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                               max_memory_allocated=lambda: 0)
        # Each window has four clock readings. Model-only timing would wrongly
        # pass the slow case; the fast case still requires a listening review.
        for duration, status, overruns in ((.6, "throughput_insufficient", 37),
                                            (.05, "throughput_gate_passed", 0)):
            with self.subTest(status=status):
                ticks = iter(t for n in range(37) for t in (n, n + .01, n + .02, n + duration))
                with patch.object(validation.time, "perf_counter", side_effect=lambda: next(ticks)):
                    _, _, metrics = validation.benchmark(self.audio, self.mouths, runtime,
                                                          StubSlicer(), SimpleNamespace(cuda=cuda))
                self.assertEqual(metrics["throughput_status"], status)
                self.assertEqual(metrics["processing_overruns"], overruns)
                self.assertLess(metrics["inference"]["p95_ms"], 20)
                self.assertEqual(metrics["quality_status"], "pending_listening_review")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe required")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="lipsync-tests-")
        cls.directory = Path(cls.temporary.name)
        cls.ffmpeg, cls.ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        cls.clip = cls.directory / "clip with spaces.mkv"
        validation.command([
            cls.ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
            "testsrc2=size=128x96:rate=30:duration=20", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=20", "-c:v", "ffv1",
            "-c:a", "pcm_s16le", "-ac", "2", str(cls.clip)
        ])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_prepare_resamples_and_keeps_audio_video_duration(self):
        audio, mouths, info = validation.prepare_media(self.clip, (16, 16, 48), self.ffmpeg, self.ffprobe)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (320000,))
        self.assertEqual(mouths.dtype, np.uint8)
        self.assertEqual(mouths.shape, (500, 96, 96))
        self.assertEqual(info.start, 0)

    def test_real_stream_offset_is_not_reset_independently(self):
        clip = self.directory / "audio offset.mkv"
        validation.command([
            self.ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
            "nullsrc=size=96x96:rate=25:duration=3,geq=lum='if(lt(T,0.4),0,255)':cb=128:cr=128",
            "-itsoffset", "0.4", "-f", "lavfi", "-i",
            "aevalsrc='if(lt(t,0.15),0.5,0)':s=16000:d=2.6",
            "-c:v", "ffv1", "-c:a", "pcm_s16le", str(clip)
        ])
        audio, mouths, info = validation.prepare_media(clip, (0, 0, 96), self.ffmpeg, self.ffprobe)
        self.assertAlmostEqual(info.start, .4, places=3)
        self.assertGreater(float(np.mean(mouths[0])), 240)
        self.assertGreater(float(np.mean(audio[:1600])), .49)
        self.assertLess(float(np.max(np.abs(audio[4000:]))), .001)

    def test_rotated_phone_uses_upright_crop_coordinates(self):
        base, rotated = self.directory / "base.mp4", self.directory / "rotated.mp4"
        validation.command([
            self.ffmpeg, "-v", "error", "-nostdin", "-i", str(self.clip), "-t", "3",
            "-c:v", "mpeg4", "-c:a", "aac", str(base)
        ])
        validation.command([
            self.ffmpeg, "-v", "error", "-nostdin", "-display_rotation", "90",
            "-i", str(base), "-c", "copy", str(rotated)
        ])
        info = validation.probe(rotated, self.ffprobe)
        self.assertEqual((info.width, info.height), (96, 128))
        _, mouths, _ = validation.prepare_media(rotated, (80, 110, 16), self.ffmpeg, self.ffprobe)
        self.assertGreaterEqual(len(mouths), 50)
        with self.assertRaisesRegex(validation.ValidationError, "upright 96x128"):
            validation.prepare_media(rotated, (110, 0, 16), self.ffmpeg, self.ffprobe)

    def test_missing_audio_and_invalid_boxes_fail(self):
        with self.assertRaisesRegex(validation.ValidationError, "outside"):
            validation.prepare_media(self.clip, (-1, 0, 32), self.ffmpeg, self.ffprobe)
        silent = self.directory / "no audio.mkv"
        validation.command([self.ffmpeg, "-v", "error", "-nostdin", "-i", str(self.clip),
                            "-t", "3", "-an", "-c:v", "copy", str(silent)])
        with self.assertRaisesRegex(validation.ValidationError, "both a video stream and an audio"):
            validation.prepare_media(silent, (0, 0, 32), self.ffmpeg, self.ffprobe)

    def test_short_clip_and_cli_missing_file_fail_without_outputs(self):
        short = self.directory / "too short.mkv"
        validation.command([self.ffmpeg, "-v", "error", "-nostdin", "-i", str(self.clip),
                            "-t", "0.5", "-c", "copy", str(short)])
        with self.assertRaisesRegex(validation.ValidationError, "at least two seconds"):
            validation.prepare_media(short, (0, 0, 32), self.ffmpeg, self.ffprobe)
        output = self.directory / "missing input output"
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "validate_extraction.py"),
            "--falcon-dir", str(ROOT / "external" / "Falcon"),
            "--video", str(self.directory / "missing.mp4"), "--mouth-box", "0", "0", "32",
            "--device", "cpu", "--out-dir", str(output)
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input video does not exist", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_failed_inference_does_not_publish_an_output_directory(self):
        output = self.directory / "failed inference"
        runtime = IdentityRuntime()
        runtime.infer = lambda audio, frames: np.full(32000, np.nan, dtype=np.float32)
        args = argparse.Namespace(video=self.clip, falcon_dir=ROOT / "external" / "Falcon",
                                  mouth_box=[16, 16, 48], device="cpu", out_dir=output)
        with patch.object(validation, "load_engine", return_value=(runtime, StubSlicer(), None, None)):
            with self.assertRaisesRegex(validation.ValidationError, "non-finite"):
                validation.run(args)
        self.assertFalse(output.exists())

    def test_complete_run_publishes_aligned_outputs_and_provenance(self):
        checkpoint = self.directory / "test-checkpoint.pt"
        checkpoint.write_bytes(b"test fixture only")
        fake_torch = SimpleNamespace(version=SimpleNamespace(cuda=None))
        output = self.directory / "complete run"
        args = argparse.Namespace(video=self.clip, falcon_dir=ROOT / "external" / "Falcon",
                                  mouth_box=[16, 16, 48], device="cpu", out_dir=output)
        with patch.object(validation, "load_engine", return_value=(IdentityRuntime(), StubSlicer(), fake_torch, checkpoint)):
            metrics = validation.run(args)
        self.assertEqual({p.name for p in output.iterdir()},
                         {"mixture.wav", "target.wav", "mouth_preview.mp4", "metrics.json"})
        self.assertEqual(metrics["throughput_status"], "not_validated_cpu")
        self.assertEqual(json.loads((output / "metrics.json").read_text())["source"]["mouth_box"], [16, 16, 48])
        with wave.open(str(output / "mixture.wav")) as mixture, wave.open(str(output / "target.wav")) as target:
            self.assertEqual(mixture.getnframes(), target.getnframes())
            self.assertEqual(mixture.readframes(mixture.getnframes()), target.readframes(target.getnframes()))
        with self.assertRaisesRegex(validation.ValidationError, "already exists"):
            validation.run(args)


if __name__ == "__main__":
    unittest.main()
