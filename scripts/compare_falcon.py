"""Compare Falcon on a mixture and anonymous SepFormer candidates, offline.

Use --prepare-only first to export the exact audio to separate. See
docs/falcon-comparison.md. No model dependencies are loaded for preparation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    from scripts import validate_extraction as media
    from scripts.falcon_adapter import FalconAdapter, ValidationError, SR, FPS, SIDE, command, sha256
except ModuleNotFoundError:
    import validate_extraction as media
    from falcon_adapter import FalconAdapter, ValidationError, SR, FPS, SIDE, command, sha256

ROOT = Path(__file__).resolve().parents[1]


def resample(samples, source_rate, target_rate):
    import numpy as np
    from scipy.signal import resample_poly
    divisor = math.gcd(source_rate, target_rate)
    return np.ascontiguousarray(resample_poly(samples, target_rate // divisor,
                                             source_rate // divisor), dtype=np.float32)


def read_pcm(path):
    import numpy as np
    with wave.open(str(path), 'rb') as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getcomptype() != 'NONE':
            raise ValidationError(f'{path.name} must be mono PCM16 WAV.')
        rate, count = handle.getframerate(), handle.getnframes()
        data = handle.readframes(count)
    if len(data) != count * 2 or not count:
        raise ValidationError(f'{path.name} is empty or truncated.')
    return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768, rate


def load_candidates(directory, audio):
    """Match the source at its declared rate; never infer offsets or identities."""
    import numpy as np
    names = ['mixture.wav', 'candidate_0.wav', 'candidate_1.wav', 'metrics.json']
    if not all((directory / name).is_file() for name in names):
        raise ValidationError('--separation-dir needs mixture.wav, both candidates, and metrics.json.')
    report = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    try:
        gain = report['export']['shared_gain']
        rate = report['sample_rate']
        if (isinstance(gain, bool) or not isinstance(gain, (int, float))
                or not math.isfinite(gain) or not 0 < gain <= 1
                or isinstance(rate, bool) or not isinstance(rate, int) or not 8000 <= rate <= 48000):
            raise ValueError()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError('Metrics need sample_rate (8000–48000) and finite export.shared_gain in (0, 1].') from exc
    mixture, actual_rate = read_pcm(directory / 'mixture.wav')
    expected = resample(audio, SR, rate)
    if (rate != actual_rate or expected.shape != mixture.shape
            or not np.allclose(expected.astype(np.float64) * gain, mixture, atol=2/32768, rtol=0)):
        raise ValidationError('Separation mixture does not match the prepared recording/timeline and shared gain. Use --prepare-only; no offset guessing is performed.')
    # Silence cannot verify which recording or timeline was used.
    if np.max(np.abs(expected)) * gain < 4/32768:
        raise ValidationError('Mixture is too quiet to verify its recording/timeline after export attenuation.')
    candidates = []
    for index in range(2):
        samples, candidate_rate = read_pcm(directory / f'candidate_{index}.wav')
        if candidate_rate != rate or samples.shape != mixture.shape:
            raise ValidationError('Candidates must share the mixture sample rate, count, and timeline.')
        restored = samples.astype(np.float64) / gain
        candidate = resample(restored, rate, SR)
        if len(candidate) < len(audio) or not np.isfinite(candidate).all():
            raise ValidationError('Restored candidate is non-finite or too short.')
        candidates.append(np.ascontiguousarray(candidate[:len(audio)]))
    control = resample(expected, rate, SR)[:len(audio)]
    return candidates, control, {
        'sample_rate': rate, 'restored_export_gain': gain,
        'mixture_max_error_after_gain': float(np.max(np.abs(expected.astype(np.float64)*gain-mixture))),
        'verification_tolerance': 2/32768,
        'files': {name: sha256(directory / name) for name in names},
        'metrics': report,
        'timeline_verification': 'samplewise mixture agreement; candidate alignment is an exporter contract',
    }


def mouth_preview(path, mouths, ffmpeg):
    command([ffmpeg, '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pixel_format', 'gray',
             '-video_size', f'{SIDE}x{SIDE}', '-framerate', str(FPS), '-i', 'pipe:0',
             '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(path)],
            input_bytes=mouths.tobytes())


def listening_preview(source, destination, silent, ffmpeg):
    """Two-pass EBU R128 previews; analysis files never undergo normalization."""
    if silent:
        shutil.copy2(source, destination)
        return {'status': 'silent', 'normalization': 'none'}
    base = [ffmpeg, '-v', 'info', '-nostdin', '-i', str(source)]
    result = subprocess.run(base + ['-af', 'loudnorm=I=-23:TP=-2:LRA=11:print_format=json',
                                    '-f', 'null', '-'], capture_output=True, check=False)
    if result.returncode:
        raise ValidationError('Listening loudness measurement failed: ' + result.stderr.decode(errors='replace')[-1500:])
    log = result.stderr.decode(errors='replace')
    measured = json.loads(log[log.rfind('{'):log.rfind('}')+1])
    values = [float(measured[key]) for key in ('input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset')]
    if not all(math.isfinite(value) for value in values):
        shutil.copy2(source, destination)
        return {'status': 'unmeasurable', 'normalization': 'none', 'measurement': measured}
    normalization = ('loudnorm=I=-23:TP=-2:LRA=11:linear=true:print_format=json:'
                     f'measured_I={values[0]}:measured_TP={values[1]}:measured_LRA={values[2]}:'
                     f'measured_thresh={values[3]}:offset={values[4]}')
    result = subprocess.run(base + ['-af', normalization, '-ar', str(SR), '-c:a', 'pcm_s16le',
                                    str(destination)], capture_output=True, check=False)
    if result.returncode:
        raise ValidationError('Listening preview export failed: ' + result.stderr.decode(errors='replace')[-1500:])
    log = result.stderr.decode(errors='replace')
    return {'status': 'normalized', 'target_lufs': -23, 'true_peak_limit_dbtp': -2,
            'measurement': measured, 'result': json.loads(log[log.rfind('{'):log.rfind('}')+1])}


def run(args):
    import numpy as np
    from scipy.io import wavfile
    output, video = args.out_dir.resolve(), args.video.resolve()
    if output.exists():
        raise ValidationError('Output directory already exists; choose a fresh directory.')
    if not video.is_file():
        raise ValidationError(f'Input video does not exist: {video}')
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise ValidationError('Install FFmpeg and put ffmpeg and ffprobe on PATH.')
    started = time.perf_counter()
    audio, mouths, info = media.prepare_media(video, tuple(args.mouth_box), ffmpeg, ffprobe)
    report = {
        'schema_version': 1, 'mode': 'offline_falcon_comparison',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {'filename': video.name, 'sha256': sha256(video), 'mouth_box': args.mouth_box,
                   'video_start_seconds': info.video_start, 'audio_start_seconds': info.audio_start,
                   'common_start_seconds': info.start, 'common_end_seconds': info.end,
                   'retained_end_seconds': info.start+len(audio)/SR,
                   'rotation_degrees': info.rotation, 'upright_dimensions': [info.width, info.height]},
        'sample_rate': SR, 'video_fps': FPS, 'mouth_frame_count': len(mouths),
        'preparation_ms': (time.perf_counter()-started)*1000,
        'quality_status': 'pending_listening_review', 'live_suitability': 'not_evaluated',
        'candidate_indices_are_speaker_identities': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.lipsync-comparison-', dir=output.parent) as temporary:
        stage = Path(temporary)
        mouth_preview(stage / 'mouth_preview.mp4', mouths, ffmpeg)
        if args.prepare_only:
            # Float32 avoids another PCM quantization before SepFormer inference.
            wavfile.write(stage / 'input.wav', 8000, resample(audio, SR, 8000))
            report.update(mode='falcon_comparison_input', separation_input_rate=8000,
                          input_sha256=sha256(stage / 'input.wav'))
        else:
            candidates, control, provenance = load_candidates(args.separation_dir.resolve(), audio)
            report['separation'] = provenance
            branches = {'original': audio, 'roundtrip': control,
                        'candidate_0': candidates[0], 'candidate_1': candidates[1]}
            pins = json.loads((ROOT / 'model-validation.lock.json').read_text(encoding='utf-8'))
            print(f'Loading pinned Falcon Small on {args.device}...', flush=True)
            started = time.perf_counter()
            adapter = FalconAdapter.load(args.falcon_dir.resolve(), args.device, pins)
            report.update(model_loading_ms=(time.perf_counter()-started)*1000, pins=pins,
                          checkpoint_sha256=sha256(adapter.checkpoint), device=adapter.runtime.device,
                          python=sys.version, platform=platform.platform(), packages=media.package_versions(),
                          cuda_version=adapter.torch.version.cuda,
                          gpu=adapter.torch.cuda.get_device_name() if args.device == 'cuda' else None,
                          nvidia_smi=media.nvidia_driver_information() if args.device == 'cuda' else None,
                          ffmpeg=command([ffmpeg, '-version']).decode().splitlines()[0],
                          resampling={'method': 'scipy.signal.resample_poly', 'falcon_rate': SR,
                                      'separator_rate': provenance['sample_rate'],
                                      'restores_missing_high_frequencies': False})
            tracks, measurements, interval = {}, {}, None
            for name, samples in branches.items():
                print(f'Comparing {name}...', flush=True)
                baseline, refined, metrics = adapter.extract(samples, mouths)
                if metrics['target_clipped_samples']:
                    raise ValidationError('Upstream Falcon slicer clipped output; rerun with quieter source input. No clipped comparison published.')
                if interval is not None and metrics['output_sample_range'] != interval:
                    raise ValidationError('Falcon branches returned different source intervals.')
                interval = metrics['output_sample_range']
                tracks[name], tracks[f'{name}_falcon'] = baseline, refined
                measurements[name] = metrics
            peak = max(float(np.max(np.abs(track))) for track in tracks.values())
            gain = min(1.0, .99 / peak) if peak else 1.0
            previews = stage / 'listening_previews'
            previews.mkdir()
            normalizations = {}
            for name, track in tracks.items():
                path = stage / f'{name}.wav'
                media.write_wav(path, track * gain)
                normalizations[name] = listening_preview(path, previews / path.name,
                                                        not np.any(track), ffmpeg)
            report.update(branches=measurements, output_sample_range=interval,
                          output_source_interval_seconds=[info.start+n/SR for n in interval],
                          export={'format': 'WAV PCM16', 'shared_gain': gain, 'unscaled_global_peak': peak,
                                  'upstream_slicer_clipped_samples': 0},
                          listening_previews=normalizations)
        report['output_hashes'] = {str(path.relative_to(stage)): sha256(path)
                                   for path in sorted(stage.rglob('*')) if path.is_file()}
        (stage / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        stage.rename(output)
    print(f'Saved {output}. Listening review and A3000 acceptance remain separate.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--mouth-box', nargs=3, type=int, metavar=('X', 'Y', 'SIZE'), required=True)
    parser.add_argument('--separation-dir', type=Path)
    parser.add_argument('--falcon-dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Export aligned 8 kHz input.wav for SepFormer, without loading Falcon')
    args = parser.parse_args(argv)
    if not args.prepare_only and (args.separation_dir is None or args.falcon_dir is None):
        parser.error('--separation-dir and --falcon-dir are required unless --prepare-only is used')
    try:
        run(args)
    except (ValidationError, OSError, ValueError, RuntimeError, ImportError, wave.Error) as exc:
        print(f'Comparison failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
