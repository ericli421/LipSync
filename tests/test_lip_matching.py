"""Lip matching with known synthetic associations; no camera, model, or network."""
from __future__ import annotations

import shutil
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
    import scipy
except ImportError:
    MATCH_DEPS = False
else:
    MATCH_DEPS = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
if MATCH_DEPS:
    import lip_matching as matching
    import match_lips as runner


@unittest.skipUnless(MATCH_DEPS, "Install the audio/lip-matching environment")
class LipMatchingTests(unittest.TestCase):
    def signals(self, seconds=20):
        from scipy.signal import butter, sosfiltfilt
        rng = np.random.default_rng(42)
        frames = seconds*25
        variation = sosfiltfilt(butter(2, [1, 5], fs=25, btype="bandpass", output="sos"),
                               rng.normal(size=(frames,2)), axis=0)
        mouths = .10 + .025*variation
        envelopes = .1*np.exp(mouths*6)
        return mouths, envelopes

    def test_detector_order_does_not_swap_track_ids(self):
        tracker = matching.FaceTracker()
        left, right = {"center":[.2,.4]}, {"center":[.8,.4]}
        self.assertEqual(tracker.assign([right,left]), [left,right])
        self.assertEqual(tracker.assign([left,right]), [left,right])
        self.assertEqual(tracker.assign([right]), [None,right])
        self.assertEqual(tracker.assign([]), [None,None])

    def test_distant_new_face_does_not_overwrite_a_track(self):
        tracker = matching.FaceTracker(max_distance=.15)
        tracker.assign([{"center":[.1,.1]}, {"center":[.9,.1]}])
        self.assertEqual(tracker.assign([{"center":[.5,.9]}]), [None,None])

    def test_known_associations_and_rolled_video_control(self):
        mouths, envelopes = self.signals()
        labels, statuses, records = matching.match_signals(mouths,envelopes)
        self.assertGreater(np.mean(labels[75:,0]==0), .90)
        self.assertGreater(np.mean(labels[75:,1]==1), .90)
        self.assertTrue(np.all(labels[:49]==-1))
        self.assertGreater(records[0]["scores"][0][0], .9)
        # Convert the known envelopes to amplitude-modulated audio, preserving
        # known source identities independently of the matcher.
        sample_times = np.arange(len(mouths)*320)/8000
        candidates = np.repeat(envelopes,320,axis=0)*np.column_stack((
            np.sin(2*np.pi*220*sample_times), np.sin(2*np.pi*330*sample_times)))
        _, _, _, report = matching.analyze(mouths,candidates,8000)
        self.assertEqual(report["negative_control"]["timing_check"], "advantage_observed")
        self.assertGreater(report["negative_control"]["mean_best_correlation_advantage"], .2)
        self.assertEqual(report["accuracy"], "not_measured_no_ground_truth")

    def test_small_audio_visual_lag_is_recovered(self):
        mouths, envelopes = self.signals()
        delayed = np.concatenate((np.repeat(envelopes[:1],3,axis=0),envelopes[:-3]))
        labels, _, records = matching.match_signals(mouths,delayed)
        self.assertGreater(np.mean(labels[75:,0]==0), .9)
        self.assertAlmostEqual(records[5]["audio_lag_seconds"][0][0], .12, places=2)

    def test_still_or_missing_mouth_is_uncertain(self):
        mouths, envelopes = self.signals()
        mouths[:,0] = .1
        mouths[:,1] = np.nan
        labels, _, _ = matching.match_signals(mouths,envelopes)
        self.assertTrue(np.all(labels==-1))

    def test_face_loss_mutes_instead_of_holding_forever(self):
        mouths, envelopes = self.signals()
        mouths[250:,0] = np.nan
        labels, _, _ = matching.match_signals(mouths,envelopes)
        self.assertTrue(np.all(labels[250:,0]==-1))

    def test_ambiguous_duplicate_sources_are_not_confidently_selected(self):
        mouths, envelopes = self.signals()
        envelopes[:,1] = envelopes[:,0]
        labels, _, _ = matching.match_signals(mouths,envelopes)
        self.assertTrue(np.all(labels==-1))

    def test_same_candidate_cannot_be_assigned_to_both_faces(self):
        mouths, envelopes = self.signals()
        mouths[:,1] = mouths[:,0]
        labels, _, _ = matching.match_signals(mouths,envelopes)
        self.assertFalse(np.any((labels[:,0]>=0)&(labels[:,0]==labels[:,1])))

    def test_selected_audio_alignment_and_uncertainty(self):
        frames=100
        choices=np.full((frames,2),-1,dtype=int)
        choices[25:75,0]=0
        choices[25:75,1]=1
        candidates=np.column_stack((np.full(frames*320,.2),np.full(frames*320,-.3)))
        selected=matching.selected_audio(candidates,choices,8000,25)
        self.assertEqual(selected.shape,candidates.shape)
        np.testing.assert_allclose(selected[30*320:70*320], candidates[30*320:70*320])
        self.assertTrue(np.all(selected[:20*320]==0))
        self.assertTrue(np.all(selected[80*320:]==0))
        self.assertLessEqual(float(np.max(np.abs(selected))), .300001)

    def test_timeline_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample-aligned"):
            matching.audio_envelopes(np.zeros((1000,2)),8000,5,25)


@unittest.skipUnless(MATCH_DEPS and shutil.which("ffmpeg") and shutil.which("ffprobe"), "SciPy and FFmpeg required")
class MediaIntegrationTests(unittest.TestCase):
    def test_common_origin_preserves_a_real_audio_offset_and_checks_roi(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            video=directory/"offset clip.mkv"
            runner.command([
                shutil.which("ffmpeg"),"-v","error","-nostdin","-f","lavfi","-i",
                "testsrc2=size=128x96:rate=30:duration=3","-itsoffset","0.4","-f","lavfi","-i",
                "sine=frequency=440:sample_rate=48000:duration=2.6","-c:v","ffv1","-c:a","pcm_s16le",str(video),
            ])
            stage=directory/"stage"
            stage.mkdir()
            prepared,audio,info=runner.prepare_video(video,stage,[0,0,128,96],25)
            samples,rate=sf.read(str(audio))
            self.assertAlmostEqual(info["common_start_seconds"],.4,places=3)
            self.assertEqual(rate,8000)
            self.assertEqual(len(samples),65*320)
            self.assertEqual(info["frames"],65)
            self.assertTrue(prepared.is_file())
            with self.assertRaisesRegex(runner.MatchError,"outside"):
                runner.prepare_video(video,stage,[0,0,500,500],25)

    def test_cached_audio_is_verified_before_reuse(self):
        import soundfile as sf
        import separate_audio
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            signal=np.sin(np.arange(8000)*.2).astype(np.float32)*.2
            original=directory/"input.wav"
            sf.write(str(original),signal,8000,subtype="FLOAT")
            cached=directory/"cached"
            candidates=np.column_stack((signal*.4,signal*.6))
            separate_audio.publish_outputs(cached,signal,candidates,{"sample_rate":8000})
            stage=directory/"stage"
            stage.mkdir()
            mixture,sources,rate=runner.get_separation(original,stage,cached,"cpu")
            self.assertEqual(sources.shape,(8000,2))
            self.assertEqual(rate,8000)
            sf.write(str(original),np.zeros(8000),8000,subtype="FLOAT")
            with self.assertRaisesRegex(runner.MatchError,"not aligned"):
                runner.get_separation(original,directory,cached,"cpu")


if __name__=="__main__":
    unittest.main()
