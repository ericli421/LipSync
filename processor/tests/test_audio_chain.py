"""Voice band gate: measured attenuation inside/outside the human voice band."""
import unittest

import numpy as np

from processor.audio_chain import AudioChain, ChainSettings, VoiceBand


def tone(freq, rate, seconds=1.0, amplitude=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def gain_db(band, freq, rate):
    x = tone(freq, rate)
    blocks = [band.process(block) for block in np.split(x, len(x) // (rate // 50))]  # 20 ms streaming
    y = np.concatenate(blocks)[rate // 2:]  # skip the filter's settling time
    return 20 * np.log10(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x[rate // 2:] ** 2)))


class VoiceBandTests(unittest.TestCase):
    def test_passes_voice_and_rejects_rumble_and_hiss(self):
        for rate in (48000, 44100):
            band = VoiceBand(rate)
            for freq in (200, 1000, 3000):
                self.assertGreater(gain_db(VoiceBand(rate), freq, rate), -1.0, (rate, freq))
            self.assertLess(gain_db(VoiceBand(rate), 30, rate), -30, rate)
            self.assertLess(gain_db(VoiceBand(rate), 16000, rate), -20, rate)
            del band

    def test_16k_capture_degrades_to_highpass(self):
        band = VoiceBand(16000)
        self.assertIsNone(band.high_hz)
        self.assertGreater(gain_db(VoiceBand(16000), 6000, 16000), -1.0)
        self.assertLess(gain_db(VoiceBand(16000), 30, 16000), -30)


class ChainTests(unittest.TestCase):
    def pcm(self, x):
        return (x * 32767).astype("<i2").tobytes()

    def test_bypass_is_bit_exact_and_length_preserved(self):
        settings = ChainSettings(voice_band=False)
        chain = AudioChain(48000, settings)
        block = self.pcm(tone(50, 48000, 0.02))
        self.assertEqual(chain.process(block), block)
        settings.update({"type": "set_layer", "layer": "voice_band", "enabled": True})
        filtered = chain.process(block)
        self.assertEqual(len(filtered), len(block))
        self.assertNotEqual(filtered, block)

    def test_band_changes_rebuild_filter_and_validate(self):
        settings = ChainSettings()
        chain = AudioChain(16000, settings)
        settings.update({"type": "set_band", "low_hz": 300, "high_hz": 3400})
        chain.process(bytes(640))
        self.assertEqual(chain.band.high_hz, 3400)
        for bad in ({"low_hz": 5000, "high_hz": 4000}, {"low_hz": 10, "high_hz": 8000},
                    {"low_hz": 1000, "high_hz": 1500}, {"low_hz": True, "high_hz": 8000}):
            with self.assertRaises(ValueError):
                settings.update({"type": "set_band", **bad})

    def test_status_reports_meters(self):
        chain = AudioChain(48000)
        chain.process(self.pcm(tone(1000, 48000, 0.02)))
        status = chain.status()
        self.assertEqual(status["type"], "pipeline")
        self.assertTrue(status["layers"]["voice_band"])
        self.assertGreater(status["in_db"], -10)
        self.assertLess(status["process_ms"], 5)


if __name__ == "__main__":
    unittest.main()
