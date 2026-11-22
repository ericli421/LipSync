"""Pretrained active speaker detection (Light-ASD, CVPR 2023, MIT license).

Given the on-camera face and a sound, the network scores, per video frame, whether
that face is the one producing the sound. It learned this from labelled data, so
there is nothing to calibrate per person:

- "ava":     AVA-ActiveSpeaker (movies; several people, overlapping speech, noise).
- "talkset": the authors' fine-tune on TalkSet (VoxCeleb2 + LRS3, including clips
             where the audio belongs to someone other than the visible face), which is
             exactly our question: does *this* separated voice belong to *this* face?

We run it once per separated voice and boost the voice whose score is higher.

Inputs follow the reference implementation: 112x112 grayscale face crops at 25 fps
and 13 MFCCs at 100 frames/s from 16 kHz audio in int16 scale (4 audio frames per
video frame).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FPS = 25
RATE = 16000
CROP = 112
WEIGHTS_DIR = Path(__file__).resolve().parent / "light_asd"
WEIGHTS = {  # sha256 of the files in Junhua-Liao/Light-ASD @ ed38c23 (weight/)
    "ava": ("pretrain_AVA_CVPR.model", "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317"),
    "talkset": ("finetuning_TalkSet.model", "efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5"),
}


def face_crop(frame_bgr, box, scale=0.4):
    """112x112 grayscale crop around a face box (x0, y0, x1, y1) in pixels.

    Same geometry as Light-ASD's crop_video: a square that extends below the face
    detection to include the mouth and chin, padded with grey, then the centre half
    of a 224 resize.
    """
    import cv2
    x0, y0, x1, y1 = box
    bs = max(y1 - y0, x1 - x0) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pad = int(bs * (1 + 2 * scale)) + 1
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=110)
    my, mx = cy + pad, cx + pad
    face = padded[int(my - bs):int(my + bs * (1 + 2 * scale)), int(mx - bs * (1 + scale)):int(mx + bs * (1 + scale))]
    if face.size == 0:
        return None
    face = cv2.resize(face, (224, 224))
    return face[56:168, 56:168].copy()


def mfcc(audio, rate=RATE):
    """13 MFCCs at 100 frames/s; audio is float in [-1, 1] (converted to int16 scale)."""
    import python_speech_features
    return python_speech_features.mfcc(np.asarray(audio, np.float64) * 32768.0, rate,
                                       numcep=13, winlen=0.025, winstep=0.010)


class ActiveSpeaker:
    def __init__(self, weights="talkset", device="cpu"):
        import torch
        from processor.light_asd.model import ASD_Model
        name, digest = WEIGHTS[weights]
        path = WEIGHTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run scripts/setup-models.ps1 (Light-ASD weights)")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{path} does not match the pinned Light-ASD weights")
        self.torch, self.device = torch, device
        self.model = ASD_Model()
        self.fc = torch.nn.Linear(128, 2)   # lossAV.FC: "this face is speaking this sound"
        self.fcv = torch.nn.Linear(128, 2)  # lossV.FC: "this face is speaking", from the video alone
        state = torch.load(path, map_location="cpu", weights_only=True)
        own = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
        self.model.load_state_dict(own)
        self.fc.load_state_dict({"weight": state["lossAV.FC.weight"], "bias": state["lossAV.FC.bias"]})
        self.fcv.load_state_dict({"weight": state["lossV.FC.weight"], "bias": state["lossV.FC.bias"]})
        for module in (self.model, self.fc, self.fcv):
            module.to(device).eval()

    def speaking(self, faces):
        """Per-frame log-odds that the face is talking, from the face crops alone (T,)."""
        torch = self.torch
        with torch.no_grad():
            video = torch.as_tensor(np.asarray(faces), dtype=torch.float32, device=self.device)[None]
            v = self.model.forward_visual_frontend(video)[0]
            logits = self.fcv(v)
        return (logits[:, 1] - logits[:, 0]).cpu().numpy()

    def score(self, faces, audios):
        """faces: (T, 112, 112) uint8; audios: list of 16 kHz float arrays covering the same
        T / 25 s. Returns (len(audios), T) logits: > 0 means "this face is speaking this sound"."""
        torch = self.torch
        t = len(faces)
        features = []
        for audio in audios:
            m = mfcc(audio)
            need = 4 * t
            if len(m) < need:
                m = np.pad(m, ((0, need - len(m)), (0, 0)), mode="edge")
            features.append(m[:need])
        with torch.no_grad():
            video = torch.as_tensor(np.asarray(faces), dtype=torch.float32, device=self.device)
            video = video.unsqueeze(0).expand(len(audios), -1, -1, -1).contiguous()
            audio = torch.as_tensor(np.stack(features), dtype=torch.float32, device=self.device)
            v = self.model.forward_visual_frontend(video)
            a = self.model.forward_audio_frontend(audio)
            out = self.model.forward_audio_visual_backend(a, v)
            logits = self.fc(out)
            score = (logits[:, 1] - logits[:, 0]).reshape(len(audios), t)
        return score.cpu().numpy()


class AsdDecision:
    """Turn per-frame Light-ASD scores for the two voices into a stable pick.

    Each hop gives, for each separated voice, the mean score over the window's frames
    where the face is visible. Their difference (log-odds that voice 2 rather than
    voice 1 belongs to the face) is accumulated with a `tau_s` memory; the pick
    changes only when the accumulated evidence clearly favours the other voice.
    """
    def __init__(self, tau_s=1.0, enter=1.0, switch=2.0, clip=8.0, min_valid=0.5):
        self.tau_s, self.enter, self.switch, self.clip, self.min_valid = tau_s, enter, switch, clip, min_valid
        self.evidence = 0.0
        self.choice = -1
        self.switches = 0
        self.scores = [None, None]
        self.reason = "waiting for face"

    def update(self, scores, valid, dt):
        valid = np.asarray(valid, bool)
        if len(valid) == 0 or valid.mean() < self.min_valid:
            self.reason = "face not visible"
            return self.choice
        means = np.clip(np.asarray(scores)[:, valid].mean(axis=1), -self.clip, self.clip)
        self.scores = [round(float(v), 2) for v in means]
        delta = float(means[1] - means[0])
        alpha = 1 - np.exp(-dt / self.tau_s)
        self.evidence += alpha * (delta - self.evidence)
        previous = self.choice
        if self.choice == -1 and abs(self.evidence) > self.enter:
            self.choice = int(self.evidence > 0)
        elif self.choice == 0 and self.evidence > self.switch:
            self.choice = 1
        elif self.choice == 1 and self.evidence < -self.switch:
            self.choice = 0
        if previous != -1 and self.choice != previous:
            self.switches += 1
            self.reason = "switched"
        elif self.choice == -1:
            self.reason = "deciding"
        else:
            self.reason = "matched" if (delta > 0) == (self.choice == 1) else "holding (window disagrees)"
        return self.choice


def voice_activity(candidates, rate=RATE, fps=FPS, range_db=12.0, soft_db=4.0):
    """Per-frame activity (0..1) of each separated voice, on a shared loudness scale.

    A frame counts as active when the voice is within `range_db` of the loudest speech
    in the window. The scale is shared by both voices, so the other voice's leakage
    (10-20 dB down in SepFormer's output) reads as inactive.
    """
    spf = rate // fps
    frames = len(candidates) // spf
    power = np.mean(np.asarray(candidates[: frames * spf]).reshape(frames, spf, -1) ** 2, axis=1)
    db = 10 * np.log10(power + 1e-10)
    reference = np.percentile(db.max(axis=1), 95)
    return np.clip((db - (reference - range_db)) / soft_db + 0.5, 0.0, 1.0)


class TalkMatch:
    """Which voice talks when this mouth talks, and is quiet when it is quiet.

    `speaking` is Light-ASD's visual "this face is talking" log-odds per frame (learned
    from AVA-ActiveSpeaker, no calibration). Each voice gets an agreement score over
    the window: +1 when it is active while the mouth talks or silent while the mouth
    is still, -1 for the opposite. The difference feeds the same evidence/hysteresis
    rule as AsdDecision. Frames where the mouth is ambiguous count little.
    """
    def __init__(self, tau_s=1.5, enter=0.15, switch=0.3, min_valid=0.5):
        self.decision = AsdDecision(tau_s=tau_s, enter=enter, switch=switch, clip=1.0, min_valid=min_valid)

    def update(self, speaking, activity, valid, dt):
        talk = np.tanh(np.asarray(speaking) / 2.0)          # -1 still .. +1 talking
        agree = talk[None, :] * (2 * np.asarray(activity).T - 1)  # (2, T)
        return self.decision.update(agree, valid, dt)

    def __getattr__(self, name):
        return getattr(self.decision, name)


class FacePicker:
    """Both Light-ASD cues, combined: the pick for Focus.

    - sync:  Light-ASD's audio-visual score of the face with each voice (strong when
             both people talk at once: the matching voice scores higher);
    - talk:  the face's visual "is talking" score matched against each voice's
             activity (strong in turn-taking: the voice that is quiet while this mouth
             talks, and active while it is still, belongs to someone else).
    Per hop each voice gets  sync_weight * sync/4 + talk agreement  (both ~[-1, 1]);
    the difference is accumulated with hysteresis as in AsdDecision.
    """
    def __init__(self, sync_weight=1.0, tau_s=1.5, enter=0.2, switch=0.4, min_valid=0.5):
        self.sync_weight = sync_weight
        self.decision = AsdDecision(tau_s=tau_s, enter=enter, switch=switch, clip=2.0, min_valid=min_valid)

    def update(self, sync, speaking, activity, valid, dt):
        talk = np.tanh(np.asarray(speaking) / 2.0)
        agree = talk[None, :] * (2 * np.asarray(activity).T - 1)
        combined = agree + self.sync_weight * np.clip(np.asarray(sync), -4, 4) / 4
        return self.decision.update(combined, valid, dt)

    def __getattr__(self, name):
        return getattr(self.decision, name)


class TalkingFacePicker:
    """Live picker for StreamingSeparator (same interface as separation.LipPicker).

    The server hands it a 112x112 face crop per video frame (`add_face`); each hop it
    lays the crops on a 25 fps grid, asks Light-ASD whether the face is talking in each
    frame, and matches that with each separated voice's activity (FacePicker with the
    talk cue only: the settings that scored best on GRID conversations, see
    scripts/eval_lip_pick.py). No per-person calibration.
    """
    TOLERANCE_US = 120_000  # phone video arrives at 10-16 fps: hold a frame up to 120 ms

    def __init__(self, model, offset_ms=0.0, tau_s=3.0, enter=0.25, switch=0.5):
        import collections
        import threading
        self.model, self.offset_ms = model, offset_ms
        self.faces = collections.deque(maxlen=FPS * 10)
        self.lock = threading.Lock()
        self.rule = FacePicker(sync_weight=0.0, tau_s=tau_s, enter=enter, switch=switch)
        self.counts = collections.Counter()
        self.last_start_us = None
        self.ms = collections.deque(maxlen=40)  # detector time per hop

    # LipPicker-compatible state
    choice = property(lambda self: self.rule.decision.choice,
                      lambda self, v: setattr(self.rule.decision, "choice", v))
    reason = property(lambda self: self.rule.decision.reason,
                      lambda self, v: setattr(self.rule.decision, "reason", v))
    scores = property(lambda self: self.rule.decision.scores)
    evidence = property(lambda self: self.rule.decision.evidence)
    switches = property(lambda self: self.rule.decision.switches)

    def add(self, capture_us, openness):
        """Mouth openness is not used: the detector looks at the face itself."""

    def add_face(self, capture_us, crop):
        with self.lock:
            self.faces.append((capture_us, crop))

    def trace(self, start_us, frames):
        """Crops on a 25 fps grid from start_us, and which grid frames have a face."""
        with self.lock:
            faces = [(t + self.offset_ms * 1000, c) for t, c in self.faces]
        crops = np.full((frames, CROP, CROP), 110, np.uint8)
        valid = np.zeros(frames, bool)
        if not faces:
            return crops, valid
        times = np.array([t for t, _ in faces], float)
        grid = start_us + np.arange(frames) * (1e6 / FPS)
        idx = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
        left = np.clip(idx - 1, 0, len(times) - 1)
        idx = np.where(np.abs(times[left] - grid) < np.abs(times[idx] - grid), left, idx)
        for k, i in enumerate(idx):
            crop = faces[i][1]
            if crop is not None and abs(times[i] - grid[k]) <= self.TOLERANCE_US:
                crops[k], valid[k] = crop, True
        return crops, valid

    def pick(self, candidates, window_start_us, rate=RATE):
        import time
        frames = len(candidates) * FPS // rate
        dt = 0.25 if self.last_start_us is None else min(1.0, max(0.05, (window_start_us - self.last_start_us) / 1e6))
        self.last_start_us = window_start_us
        crops, valid = self.trace(window_start_us, frames)
        if valid.mean() < self.rule.decision.min_valid:
            self.reason = "face not visible"
        else:
            began = time.perf_counter()
            speaking = self.model.speaking(crops)
            self.ms.append((time.perf_counter() - began) * 1000)
            activity = voice_activity(candidates[: frames * (rate // FPS)], rate)
            self.rule.update(np.zeros((2, frames)), speaking, activity, valid, dt)
        self.counts[self.reason] += 1
        return self.choice
