"""Auto volume: quiet speech becomes comfortable, pauses are not pumped up, nothing clips."""
import unittest

import numpy as np

from processor.audio_chain import AudioChain, AutoVolume, ChainSettings, level_db

RATE = 48000
BLOCK = RATE // 50


def speech_like(seconds, level_db_rms, seed=0):
    """Syllable-like bursts (4 Hz) of band-limited noise at a given RMS level, with pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    carrier = np.convolve(rng.normal(size=len(t)), np.ones(8) / 8, mode="same")
    envelope = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2
    x = carrier * envelope
    active = envelope > 0.05
    return x * (10 ** (level_db_rms / 20) / np.sqrt(np.mean(x[active] ** 2)))


def run(volume, x):
    return np.concatenate([volume.process(x[i:i + BLOCK]) for i in range(0, len(x), BLOCK)])


class AutoVolumeTests(unittest.TestCase):
    def test_quiet_speech_is_raised_to_a_comfortable_level(self):
        x = speech_like(8, -40)
        y = run(AutoVolume(RATE), x)
        tail_in, tail_out = x[5 * RATE:], y[5 * RATE:]
        active = np.abs(tail_in) > 1e-4
        self.assertGreater(level_db(tail_out[active]), -24)
        self.assertLess(level_db(tail_out[active]), -16)

    def test_room_noise_in_a_pause_is_not_pumped_up(self):
        noise = np.random.default_rng(1).normal(0, 10 ** (-60 / 20), 6 * RATE)
        x = np.concatenate([speech_like(4, -40) + noise[:4 * RATE], noise[4 * RATE:]])
        volume = AutoVolume(RATE)
        y = run(volume, x)
        gain_in_pause = level_db(y[5 * RATE:]) - level_db(x[5 * RATE:])
        self.assertLess(gain_in_pause, 21)  # held at the speech gain, never driven to +30 dB

    def test_loud_speech_never_clips(self):
        y = run(AutoVolume(RATE), speech_like(4, -6) * 3)
        self.assertLessEqual(np.max(np.abs(y)), 0.891)

    def test_chain_finish_respects_the_toggle(self):
        settings = ChainSettings(auto_volume=False)
        chain = AudioChain(RATE, settings)
        block = (speech_like(0.02, -40) * 32767).astype("<i2").tobytes()
        self.assertEqual(chain.finish(block), block)
        settings.update({"type": "set_layer", "layer": "auto_volume", "enabled": True})
        louder = b""
        for _ in range(200):
            louder = chain.finish(block)
        self.assertGreater(level_db(np.frombuffer(louder, "<i2") / 32768), level_db(np.frombuffer(block, "<i2") / 32768) + 10)


if __name__ == "__main__":
    unittest.main()
