"""Audio-only contracts: generated WAVs and injected models, no network or weights."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import soundfile as sf
    import torch
except ImportError:
    AUDIO_DEPENDENCIES = False
else:
    AUDIO_DEPENDENCIES = True

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("separate_audio", ROOT / "scripts" / "separate_audio.py")
separation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(separation)


class TestModel:
    calls = 0

    def separate_batch(self, batch):
        self.calls += 1
        return torch.stack((batch * .25, batch * .75), dim=-1)


@unittest.skipUnless(AUDIO_DEPENDENCIES, "Install the separate audio-only environment to run these tests")
class AudioOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lipsync-audio-tests-")
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_wav(self, name="input.wav", rate=8000, subtype="FLOAT", channels=1):
        signal = (.5 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)).astype(np.float32)
        if channels == 2:
            signal = np.stack((signal, np.zeros_like(signal)), axis=1)
        path = self.directory / name
        sf.write(str(path), signal, rate, subtype=subtype)
        return path, signal

    def test_float_wav_decodes_without_losing_sample_values(self):
        path, original = self.make_wav()
        audio, info = separation.prepare_audio(path, 8000)
        np.testing.assert_array_equal(audio, original)
        self.assertEqual(info["prepared_samples"], 4000)
        self.assertEqual(info["original_channels"], 1)
        self.assertEqual(audio.dtype, np.float32)

    def test_stereo_pcm_resamples_to_mono_at_model_rate(self):
        path, original = self.make_wav(rate=48000, subtype="PCM_16", channels=2)
        audio, info = separation.prepare_audio(path, 8000)
        self.assertEqual(audio.shape, (4000,))
        self.assertEqual(info["original_sample_rate"], 48000)
        self.assertEqual(info["original_channels"], 2)
        self.assertAlmostEqual(float(np.max(audio)), .25, places=2)
        # Confirm the 440 Hz tone survives resampling rather than being relabeled.
        frequency = np.argmax(np.abs(np.fft.rfft(audio))) * 8000 / len(audio)
        self.assertAlmostEqual(frequency, 440, places=1)

    def test_empty_short_corrupt_and_nonfinite_inputs_fail(self):
        for name, data, message in (
            ("empty.wav", np.array([], np.float32), "no audio samples"),
            ("short.wav", np.zeros(8, np.float32), "too short"),
            ("nan.wav", np.full(80, np.nan, np.float32), "non-finite"),
        ):
            with self.subTest(name=name):
                path = self.directory / name
                sf.write(str(path), data, 8000, subtype="FLOAT")
                with self.assertRaisesRegex(separation.SeparationError, message):
                    separation.prepare_audio(path, 8000)
        corrupt = self.directory / "corrupt.wav"
        corrupt.write_bytes(b"not a WAV")
        with self.assertRaisesRegex(separation.SeparationError, "Cannot read WAV"):
            separation.prepare_audio(corrupt, 8000)

    def test_device_selection_never_silently_falls_back_from_explicit_cuda(self):
        unavailable = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        available = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        self.assertEqual(separation.choose_device("auto", unavailable), "cpu")
        self.assertEqual(separation.choose_device("auto", available), "cuda")
        self.assertEqual(separation.choose_device("cpu", available), "cpu")
        with self.assertRaisesRegex(separation.SeparationError, "CUDA is unavailable"):
            separation.choose_device("cuda", unavailable)

    def test_one_warmup_and_one_measured_pass_preserve_length(self):
        _, audio = self.make_wav()
        model = TestModel()
        candidates, report = separation.measure_separation(model, audio, "cpu", torch)
        self.assertEqual(model.calls, 2)
        self.assertEqual(candidates.shape, (len(audio), 2))
        np.testing.assert_array_equal(candidates.sum(axis=1), audio)
        self.assertEqual(report["measured_passes"], 1)
        self.assertIsNone(report["peak_cuda_memory_bytes"])

    def test_cuda_timing_synchronizes_and_resets_peak_memory(self):
        _, audio = self.make_wav()
        events = []
        cuda = SimpleNamespace(synchronize=lambda: events.append("sync"),
                               reset_peak_memory_stats=lambda: events.append("reset"),
                               max_memory_allocated=lambda: 42)
        fake_torch = SimpleNamespace(cuda=cuda, inference_mode=torch.inference_mode,
                                     from_numpy=torch.from_numpy)
        _, report = separation.measure_separation(TestModel(), audio, "cuda", fake_torch)
        self.assertEqual(events, ["sync", "sync", "reset", "sync", "sync"])
        self.assertEqual(report["peak_cuda_memory_bytes"], 42)

    def test_nan_and_wrong_shape_outputs_are_rejected(self):
        _, audio = self.make_wav()
        for shape, value, message in (((1, len(audio), 2), float("nan"), "non-finite"),
                                      ((1, len(audio), 3), 0, "Expected model output")):
            model = SimpleNamespace(separate_batch=lambda batch: torch.full(shape, value))
            with self.subTest(message=message):
                with self.assertRaisesRegex(separation.SeparationError, message):
                    separation.measure_separation(model, audio, "cpu", torch)

    def test_exports_use_one_shared_gain_and_preserve_alignment(self):
        mixture = np.array([0, .5, -.5, 0], np.float32)
        candidates = np.stack((mixture * 4, mixture * .5), axis=1)
        output = self.directory / "output with spaces"
        metrics = {"sample_rate": 8000}
        separation.publish_outputs(output, mixture, candidates, metrics)
        self.assertAlmostEqual(metrics["export"]["shared_gain"], .495)
        self.assertFalse(metrics["export"]["candidate_indices_are_speaker_identities"])
        for name, expected in (("mixture.wav", mixture),
                               ("candidate_0.wav", candidates[:, 0]),
                               ("candidate_1.wav", candidates[:, 1])):
            saved, rate = sf.read(str(output / name), dtype="float32")
            self.assertEqual(rate, 8000)
            np.testing.assert_allclose(saved, expected * .495, atol=1/32768)
            self.assertLessEqual(float(np.max(np.abs(saved))), .9901)
        with self.assertRaisesRegex(separation.SeparationError, "already exists"):
            separation.publish_outputs(output, mixture, candidates, metrics)

    def test_failed_export_does_not_publish_partial_directory(self):
        output = self.directory / "failed export"
        with patch.object(sf, "write", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                separation.publish_outputs(output, np.zeros(100, np.float32),
                                           np.zeros((100, 2), np.float32), {"sample_rate": 8000})
        self.assertFalse(output.exists())
        self.assertEqual(list(self.directory.glob(".lipsync-audio-*")), [])

    def test_asset_downloads_use_pinned_revisions_and_hash_checks(self):
        model_dir = self.directory / "model"
        model_dir.mkdir()
        config = model_dir / "hyperparams.yaml"
        config.write_bytes(b"fixture, never loaded")
        sample, _ = self.make_wav()
        pins = {
            "model": {"repository": "model-repo", "revision": "model-revision",
                      "files": {config.name: separation.sha256(config)}},
            "sample": {"repository": "sample-repo", "revision": "sample-revision",
                       "filename": "test.wav", "sha256": separation.sha256(sample)},
        }
        with patch("huggingface_hub.snapshot_download", return_value=str(model_dir)) as snapshot:
            with patch("huggingface_hub.hf_hub_download", return_value=str(sample)) as download:
                directory, downloaded = separation.fetch_assets(pins, True)
                self.assertEqual((directory, downloaded), (model_dir, sample))
                self.assertEqual(snapshot.call_args.kwargs["revision"], "model-revision")
                self.assertEqual(download.call_args.kwargs["revision"], "sample-revision")
                config.write_bytes(b"changed")
                with self.assertRaisesRegex(separation.SeparationError, "SHA-256 mismatch"):
                    separation.fetch_assets(pins, True)

    def test_custom_wav_run_exports_four_files_with_real_device_metadata(self):
        path, _ = self.make_wav()
        output = self.directory / "complete"
        args = argparse.Namespace(input=path, device="cpu", out_dir=output)
        with patch.object(separation, "fetch_assets", return_value=(self.directory, None)) as fetch:
            with patch.object(separation, "load_model", return_value=TestModel()):
                metrics = separation.run(args)
        fetch.assert_called_once()
        self.assertFalse(fetch.call_args.kwargs["public_sample"])
        self.assertEqual({p.name for p in output.iterdir()},
                         {"mixture.wav", "candidate_0.wav", "candidate_1.wav", "metrics.json"})
        self.assertEqual(metrics["source_kind"], "local_wav")
        self.assertEqual(metrics["device"], "cpu")
        self.assertEqual(metrics["quality_status"], "pending_listening_review")
        self.assertEqual(metrics["streaming_suitability"], "not_evaluated")
        self.assertEqual(json.loads((output / "metrics.json").read_text())["source"]["filename"], path.name)

    def test_public_sample_run_and_failed_inference_publish_correctly(self):
        path, _ = self.make_wav()
        args = argparse.Namespace(input=None, device="cpu", out_dir=self.directory / "public")
        with patch.object(separation, "fetch_assets", return_value=(self.directory, path)) as fetch:
            with patch.object(separation, "load_model", return_value=TestModel()):
                metrics = separation.run(args)
        self.assertTrue(fetch.call_args.kwargs["public_sample"])
        self.assertEqual(metrics["source_kind"], "public_test_mixture")
        args.out_dir = self.directory / "failed inference"
        bad_model = SimpleNamespace(separate_batch=lambda batch: torch.full((1, batch.shape[1], 2), float("nan")))
        with patch.object(separation, "fetch_assets", return_value=(self.directory, path)):
            with patch.object(separation, "load_model", return_value=bad_model):
                with self.assertRaisesRegex(separation.SeparationError, "non-finite"):
                    separation.run(args)
        self.assertFalse(args.out_dir.exists())

    def test_cli_missing_file_is_actionable_and_does_not_download(self):
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "separate_audio.py"),
            "--input", str(self.directory / "missing.wav"), "--out-dir", str(self.directory / "missing-output")
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input WAV does not exist", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
