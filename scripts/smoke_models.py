"""Real-model generated-input smoke tests; never evidence of speech quality."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--separator', action='store_true')
    group.add_argument('--falcon', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    began = time.perf_counter()
    if args.separator:
        from processor.separation_models import load
        infer, info = load('auto', args.device)
        loaded_ms = (time.perf_counter()-began)*1000
        audio = (.05*np.sin(np.arange(12000)*2*np.pi*333/8000)).astype(np.float32)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        began = time.perf_counter()
        output = infer(audio)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter()-began)*1000
        assert output.shape == (12000, 2) and np.isfinite(output).all()
        report = {'model': info, 'loading_ms': loaded_ms, 'infer_ms': elapsed,
                  'hop_budget_ms': 250, 'single_pass_within_hop': elapsed < 250}
    else:
        from scripts.falcon_adapter import FalconAdapter
        pins = json.loads((ROOT/'model-validation.lock.json').read_text())
        adapter = FalconAdapter.load(ROOT/'external'/'Falcon', args.device, pins)
        loaded_ms = (time.perf_counter()-began)*1000
        audio = (.05*np.sin(np.arange(48000)*2*np.pi*333/16000)).astype(np.float32)
        mouths = np.full((75, 96, 96), 107, np.uint8)
        _, output, report = adapter.extract(audio, mouths)
        assert np.isfinite(output).all()
        report.update(loading_ms=loaded_ms, pins=pins)
    report.update(device=args.device, torch=torch.__version__, quality_status='not_evaluated_generated_input')
    destination = ROOT/'.tools'/('smoke-separator.json' if args.separator else 'smoke-falcon.json')
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
