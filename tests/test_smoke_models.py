"""The setup smoke test must use the selected model's rate, including 16 kHz."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import smoke_models


class SmokeModelTests(unittest.TestCase):
    def test_selected_separator_rate_controls_waveform_and_report(self):
        fake_torch = SimpleNamespace(__version__='fixture', cuda=SimpleNamespace(synchronize=lambda: None))
        for rate in (8000, 16000):
            received = []
            def infer(audio):
                received.append(audio.copy())
                return np.zeros((len(audio), 2), np.float32)
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as folder, \
                    patch.object(smoke_models, 'ROOT', Path(folder)), \
                    patch.dict(sys.modules, {'torch': fake_torch}), \
                    patch('processor.separation_models.load', return_value=(infer, {'sample_rate': rate})), \
                    patch.object(sys, 'argv', ['smoke_models.py', '--separator', '--device', 'cpu']), \
                    patch('builtins.print'):
                smoke_models.main()
                self.assertEqual(received[0].shape, (int(rate*1.5),))
                report = json.loads((Path(folder)/'.tools'/'smoke-separator.json').read_text())
                self.assertEqual(report['input_samples'], int(rate*1.5))
                self.assertEqual(report['input_duration_seconds'], 1.5)


if __name__ == '__main__':
    unittest.main()
