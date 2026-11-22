"""Falcon preparation contracts; generated fixtures are not speech-quality evidence."""
import argparse
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

from scripts import compare_falcon as comparison
from scripts import falcon_adapter as falcon


class IdentityRuntime:
    device = 'cpu'

    def infer(self, audio, frames):
        return audio.copy()


class StatefulSlicer:
    """Fail if a previous branch's state leaks into this branch."""
    instances = []

    def __init__(self):
        self.last = None
        self.instances.append(self)

    def take(self, prediction, endpoint):
        if self.last is not None:
            assert endpoint == self.last + 8000
        self.last = endpoint
        return prediction[22720:30720].copy(), True, 0


def fixture_audio(seconds=3):
    t = np.arange(seconds * falcon.SR) / falcon.SR
    return (.2*np.sin(2*np.pi*333*t) + .07*np.sin(2*np.pi*5900*t)).astype(np.float32)


def write_pcm(path, samples, rate):
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.round(np.clip(samples, -1, 32767/32768)*32768).astype('<i2').tobytes())


def export_separation(directory, audio, rate=8000, gain=.25):
    directory.mkdir()
    mixture = comparison.resample(audio, falcon.SR, rate)
    for name, samples in [('mixture', mixture), ('candidate_0', mixture*.7), ('candidate_1', mixture*.3)]:
        write_pcm(directory / f'{name}.wav', samples*gain, rate)
    (directory / 'metrics.json').write_text(json.dumps({'sample_rate': rate, 'export': {'shared_gain': gain}}))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.audio = fixture_audio()
        self.mouths = np.zeros((75, 96, 96), np.uint8)

    def test_branches_have_fresh_slicer_and_known_source_interval(self):
        StatefulSlicer.instances = []
        adapter = falcon.FalconAdapter(IdentityRuntime(), StatefulSlicer, None, None)
        for gain in (1, .2):
            baseline, output, report = adapter.extract(self.audio*gain, self.mouths)
            np.testing.assert_array_equal(baseline, output)
            self.assertEqual(report['output_sample_range'], [22720, 46720])
        self.assertEqual(len(StatefulSlicer.instances), 2)
        self.assertIsNot(StatefulSlicer.instances[0], StatefulSlicer.instances[1])

    def test_invalid_audio_mouths_prediction_and_slicer_are_rejected(self):
        bad_inputs = [self.audio.astype(np.float64), self.audio[::2], self.audio[:100],
                      np.full_like(self.audio, np.nan), self.audio[:, None]]
        adapter = falcon.FalconAdapter(IdentityRuntime(), StatefulSlicer, None, None)
        for audio in bad_inputs:
            with self.subTest(shape=audio.shape), self.assertRaises(falcon.ValidationError):
                adapter.extract(audio, self.mouths)
        for mouths in (self.mouths.astype(np.float32), self.mouths[:49], self.mouths[:, :, :80]):
            with self.assertRaises(falcon.ValidationError):
                adapter.extract(self.audio, mouths)
        for output in (np.zeros(12), np.full(32000, np.inf)):
            runtime = SimpleNamespace(device='cpu', infer=lambda *args: output)
            with self.assertRaises(falcon.ValidationError):
                falcon.FalconAdapter(runtime, StatefulSlicer, None, None).extract(self.audio, self.mouths)
        slicer = SimpleNamespace(take=lambda *args: (np.full(8000, np.nan), True, 0))
        with self.assertRaises(falcon.ValidationError):
            falcon.benchmark(self.audio, self.mouths, IdentityRuntime(), slicer, None)

    def test_explicit_device_and_source_pin_checked_before_loading(self):
        with self.assertRaisesRegex(falcon.ValidationError, 'explicit'):
            falcon.load_engine(Path('.'), 'auto', {})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'studio').mkdir()
            (directory/'studio'/'falcon_runtime.py').touch()
            with patch.object(falcon, 'command', return_value=b'wrong'):
                with self.assertRaisesRegex(falcon.ValidationError, 'Wrong Falcon commit'):
                    falcon.load_engine(directory, 'cpu', {'falcon_commit': 'pinned'})

    def test_dirty_source_checkpoint_hash_and_unavailable_cuda_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'studio').mkdir()
            (directory/'studio'/'falcon_runtime.py').touch()
            checkpoint = directory/'fixture.pt'
            checkpoint.write_bytes(b'not trusted weights')
            pins = {'falcon_commit': 'pinned', 'model_repository': 'fixture',
                    'model_revision': 'revision', 'checkpoint_filename': 'fixture.pt',
                    'checkpoint_sha256': 'incorrect'}
            with patch.object(falcon, 'command', side_effect=[b'pinned', b' M studio/file.py']):
                with self.assertRaisesRegex(falcon.ValidationError, 'local changes'):
                    falcon.load_engine(directory, 'cpu', pins)
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            hub = SimpleNamespace(hf_hub_download=lambda **kwargs: checkpoint)
            with patch.dict(sys.modules, {'torch': fake_torch, 'huggingface_hub': hub}):
                for device, error in [('cpu', 'SHA-256'), ('cuda', 'CUDA is unavailable')]:
                    with patch.object(falcon, 'command', side_effect=[b'pinned', b'']):
                        with self.assertRaisesRegex(falcon.ValidationError, error):
                            falcon.load_engine(directory, device, pins)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)/'separation'
        self.audio = fixture_audio()
        export_separation(self.directory, self.audio)

    def test_gain_restoration_and_resampling(self):
        candidates, control, report = comparison.load_candidates(self.directory, self.audio)
        expected = comparison.resample(comparison.resample(self.audio, 16000, 8000), 8000, 16000)
        np.testing.assert_allclose(control, expected, atol=1e-7)
        np.testing.assert_allclose(candidates[0], expected*.7, atol=.0003)
        np.testing.assert_allclose(candidates[1], expected*.3, atol=.0003)
        self.assertEqual(report['restored_export_gain'], .25)
        self.assertEqual(candidates[0].dtype, np.float32)
        self.assertEqual(len(report['files']), 4)
        # The round-trip control loses high-frequency content, unlike the native input.
        self.assertGreater(float(np.std(self.audio-control)), .04)

    def test_other_supported_rates_keep_exact_length(self):
        for rate in (16000, 44100, 48000):
            directory = Path(self.temp.name)/str(rate)
            export_separation(directory, self.audio, rate)
            candidates, control, _ = comparison.load_candidates(directory, self.audio)
            self.assertEqual(control.shape, self.audio.shape)
            self.assertTrue(all(candidate.shape == self.audio.shape for candidate in candidates))

    def test_wrong_recording_offset_or_gain_is_rejected(self):
        for source in (np.roll(self.audio, 100), -self.audio, self.audio[:-640]):
            with self.assertRaisesRegex(falcon.ValidationError, 'does not match'):
                comparison.load_candidates(self.directory, source)
        path = self.directory/'metrics.json'
        for gain in (None, 0, -1, True, float('nan'), float('inf'), .5):
            path.write_text(json.dumps({'sample_rate': 8000, 'export': {'shared_gain': gain}}))
            with self.assertRaises(falcon.ValidationError):
                comparison.load_candidates(self.directory, self.audio)

    def test_missing_files_wrong_candidate_rate_length_and_channels(self):
        path = self.directory/'candidate_0.wav'
        for rate, samples in ((16000, np.zeros(24000)), (8000, np.zeros(100))):
            write_pcm(path, samples, rate)
            with self.assertRaisesRegex(falcon.ValidationError, 'share'):
                comparison.load_candidates(self.directory, self.audio)
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(bytes(24000*4))
        with self.assertRaisesRegex(falcon.ValidationError, 'mono PCM16'):
            comparison.load_candidates(self.directory, self.audio)
        path.unlink()
        with self.assertRaisesRegex(falcon.ValidationError, 'four files|both candidates'):
            comparison.load_candidates(self.directory, self.audio)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg/ffprobe required')
class ComparisonMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='falcon-comparison-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root/'source with spaces.mkv'
        self.ffmpeg, self.ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
        falcon.command([self.ffmpeg, '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                        'testsrc2=size=128x96:rate=30:duration=3', '-f', 'lavfi', '-i',
                        'sine=frequency=333:sample_rate=48000:duration=3',
                        '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(self.video)])
        self.checkpoint = self.root/'checkpoint'
        self.checkpoint.write_bytes(b'test fixture, not weights')
        self.args = argparse.Namespace(video=self.video, mouth_box=[16, 16, 48], device='cpu',
                                       separation_dir=self.root/'separation', falcon_dir=self.root/'Falcon',
                                       out_dir=self.root/'comparison', prepare_only=False)
        audio, _, _ = comparison.media.prepare_media(self.video, (16, 16, 48), self.ffmpeg, self.ffprobe)
        export_separation(self.args.separation_dir, audio)
        self.adapter = falcon.FalconAdapter(IdentityRuntime(), StatefulSlicer,
                                           SimpleNamespace(version=SimpleNamespace(cuda=None)), self.checkpoint)

    def test_complete_run_and_preparation_cli(self):
        with patch.object(comparison.FalconAdapter, 'load', return_value=self.adapter) as load:
            report = comparison.run(self.args)
        load.assert_called_once()
        self.assertEqual(len(report['branches']), 4)
        self.assertEqual(report['output_sample_range'], [22720, 46720])
        self.assertEqual(report['quality_status'], 'pending_listening_review')
        for path in self.args.out_dir.glob('*.wav'):
            samples, rate = comparison.read_pcm(path)
            self.assertEqual(samples.shape, (24000,))
            self.assertEqual(rate, 16000)
            self.assertLessEqual(float(np.max(np.abs(samples))), .991)
        self.assertEqual(len(list(self.args.out_dir.glob('*.wav'))), 8)
        self.assertEqual(len(list((self.args.out_dir/'listening_previews').glob('*.wav'))), 8)
        for preview in report['listening_previews'].values():
            self.assertEqual(preview['status'], 'normalized')
            self.assertAlmostEqual(float(preview['result']['output_i']), -23, delta=.5)
        self.assertTrue((self.args.out_dir/'mouth_preview.mp4').is_file())
        with self.assertRaisesRegex(falcon.ValidationError, 'already exists'):
            comparison.run(self.args)
        output = self.root/'prepared'
        result = subprocess.run([sys.executable, str(Path(comparison.__file__)), '--video', str(self.video),
                                 '--mouth-box', '16', '16', '48', '--prepare-only', '--out-dir', str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        from scipy.io import wavfile
        rate, samples = wavfile.read(output/'input.wav')
        self.assertEqual((rate, samples.dtype, samples.shape), (8000, np.float32, (24000,)))

    def test_failure_leaves_no_published_run(self):
        with patch.object(comparison.FalconAdapter, 'load', return_value=self.adapter):
            for fail in ('inference', 'preview'):
                with self.subTest(failure=fail):
                    if fail == 'inference':
                        context = patch.object(self.adapter.runtime, 'infer', return_value=np.full(32000, np.nan))
                    else:
                        context = patch.object(comparison, 'listening_preview', side_effect=falcon.ValidationError('encoding failed'))
                    with context, self.assertRaises(falcon.ValidationError):
                        comparison.run(self.args)
                    self.assertFalse(self.args.out_dir.exists())
                    self.assertFalse(list(self.root.glob('.lipsync-comparison-*')))

    def test_clipped_upstream_output_is_not_published(self):
        baseline = np.zeros(24000, np.float32)
        with patch.object(comparison.FalconAdapter, 'load', return_value=self.adapter), patch.object(
                self.adapter, 'extract', return_value=(baseline, baseline, {'target_clipped_samples': 1})):
            with self.assertRaisesRegex(falcon.ValidationError, 'slicer clipped'):
                comparison.run(self.args)
        self.assertFalse(self.args.out_dir.exists())

    def test_one_export_gain_preserves_relative_analysis_levels(self):
        def extract(samples, mouths):
            baseline = samples[22720:46720].copy()
            return baseline, baseline*20, {'target_clipped_samples': 0,
                                          'output_sample_range': [22720, 46720]}
        with patch.object(comparison.FalconAdapter, 'load', return_value=self.adapter), patch.object(
                self.adapter, 'extract', side_effect=extract), patch.object(comparison, 'listening_preview', return_value={}):
            report = comparison.run(self.args)
        self.assertLess(report['export']['shared_gain'], 1)
        baseline, _ = comparison.read_pcm(self.args.out_dir/'original.wav')
        refined, _ = comparison.read_pcm(self.args.out_dir/'original_falcon.wav')
        np.testing.assert_allclose(baseline*20, refined, atol=.001)

    def test_silent_previews_are_marked_without_fabricated_loudness(self):
        path = self.root/'silent.wav'
        write_pcm(path, np.zeros(24000), 16000)
        result = comparison.listening_preview(path, self.root/'silent-preview.wav', True, self.ffmpeg)
        self.assertEqual(result['status'], 'silent')
        self.assertEqual(path.read_bytes(), (self.root/'silent-preview.wav').read_bytes())


if __name__ == '__main__':
    unittest.main()
