"""Keep "Voice 1" and "Voice 2" on the same people with speaker embeddings.

Chunk-to-chunk stitching only works while a voice is present in the overlap.
After a pause the separator may put a returning talker in either output, so
Voice 1 switches person mid-recording. The tracker keeps a running embedding
(ECAPA-TDNN voiceprint, pretrained on VoxCeleb) per output slot:

- A chunk is judged only when a voice is active for enough of the window, and
  the voiceprint is computed from its active frames only (a returning talker's
  first, mostly silent window is ignored).
- While stitching is uncertain (after a pause) a clear voiceprint match reorders
  the chunk; while stitching is confident, the voiceprints must disagree strongly
  for several chunks in a row before they correct it.
- A slot's voiceprint learns only from audio that matches it clearly better than
  the other slot, so one person's voice never leaks into the other's voiceprint.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
import threading

import numpy as np
from scipy.signal import resample_poly

EMBED_RATE = 16000
PIN = {
    "repository": "speechbrain/spkrec-ecapa-voxceleb",
    "revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
    "files": {
        "hyperparams.yaml": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
        "embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
        "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
        "classifier.ckpt": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
        "label_encoder.txt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
    },
}


def load_embedder(device):
    """Return embed(audio float [n] at 16 kHz) -> unit vector, using the pinned ECAPA model."""
    import torch
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import FetchConfig, LocalStrategy
    from processor.separation_models import _separate_audio
    directory = Path(snapshot_download(repo_id=PIN["repository"], revision=PIN["revision"],
                                       allow_patterns=list(PIN["files"])))
    verify = _separate_audio().verify_file
    for name, expected in PIN["files"].items():
        verify(directory / name, expected)
    model = EncoderClassifier.from_hparams(
        source=str(directory), savedir=None, local_strategy=LocalStrategy.COPY,
        fetch_config=FetchConfig(allow_network=False),
        overrides={"pretrained_path": directory.as_posix()}, run_opts={"device": device})
    model.eval()
    lock = threading.Lock()

    def embed(audio):
        with lock, torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
            vector = model.encode_batch(batch)[0, 0].float().cpu().numpy()
        return vector / max(float(np.linalg.norm(vector)), 1e-9)

    embed(np.random.default_rng(0).normal(0, 0.05, EMBED_RATE).astype(np.float32))  # warm-up
    return embed


class IdentityTracker:
    """Decide per chunk whether the two outputs should be swapped to keep identities."""
    def __init__(self, embed, frame_s=0.02, active_ratio=0.316, floor_rms=1e-4, min_active_s=0.5, leak_corr=0.8,
                 margin_pair=0.2, margin_single=0.15, strong=0.3, confirm=2, new_speaker=0.2,
                 known=0.35, confident_stitch=0.5, learn=0.1, learn_margin=0.1, clear=0.4, trained=4):
        self.embed = embed
        self.leak_corr = leak_corr  # a quieter output whose loudness follows the louder one is leakage
        self.frame_s, self.active_ratio, self.floor_rms, self.min_active_s = frame_s, active_ratio, floor_rms, min_active_s
        self.margin_pair, self.margin_single, self.strong, self.confirm = margin_pair, margin_single, strong, confirm
        self.new_speaker, self.known, self.confident_stitch = new_speaker, known, confident_stitch
        self.learn, self.learn_margin = learn, learn_margin
        self.clear, self.trained = clear, trained  # act at once on a clear match once both voiceprints are trained
        self.centroids = [None, None]
        self.updates = [0, 0]
        self.swaps = 0
        self.disagree = 0  # consecutive strong disagreements with confident stitching
        self.last = {"decision": "waiting for speech"}

    def _active_audio(self, est, rate):
        """Per candidate: its active frames concatenated, or None if too little speech."""
        hop = max(1, int(self.frame_s * rate))
        frames = len(est) // hop
        if not frames:
            return [None, None]
        energy = np.sqrt(np.mean(est[: frames * hop].reshape(frames, hop, 2) ** 2, axis=1))
        # A voice counts only within 10 dB of the loudest frame of either output.
        threshold = np.maximum(self.floor_rms, self.active_ratio * energy.max())
        louder = int(np.argmax(energy.sum(axis=0)))
        follows = np.corrcoef(np.log(energy[:, 0] + 1e-9), np.log(energy[:, 1] + 1e-9))[0, 1]
        result = []
        for i in range(2):
            if i != louder and np.isfinite(follows) and follows > self.leak_corr:
                result.append(None)  # same person leaking into the other output, not a second talker
                continue
            active = energy[:, i] > threshold
            if active.sum() * hop < self.min_active_s * rate:
                result.append(None)
                continue
            result.append(est[: frames * hop, i].reshape(frames, hop)[active].ravel())
        return result

    def _vector(self, audio, rate):
        if rate != EMBED_RATE:
            common = gcd(rate, EMBED_RATE)
            audio = resample_poly(audio, EMBED_RATE // common, rate // common)
        return self.embed(audio)

    def assign(self, est, rate, stitch_confidence):
        """est: (n, 2) candidates already stitched. Returns (est, swapped)."""
        speech = self._active_audio(est, rate)
        if all(s is None for s in speech):
            self.last = {"decision": "not enough speech"}
            return est, False
        vectors = [None if s is None else self._vector(s, rate) for s in speech]
        sim = [[None if v is None or c is None else float(v @ c) for c in self.centroids] for v in vectors]
        swap = self._decide(vectors, sim, stitch_confidence)
        if swap:
            est, vectors = est[:, ::-1], vectors[::-1]
            self.swaps += 1
            self.disagree = 0
        self._learn(vectors)
        return est, swap

    def _evidence(self, vectors, sim):
        """(score, label): score > margin means the swapped order fits the voiceprints better."""
        if all(v is not None for v in vectors) and all(x is not None for row in sim for x in row):
            return (sim[0][1] + sim[1][0]) - (sim[0][0] + sim[1][1]), "pair"
        talkers = [i for i, v in enumerate(vectors) if v is not None]
        if len(talkers) != 1:
            return None, "voiceprints not ready"
        talker = talkers[0]
        own, other = sim[talker][talker], sim[talker][1 - talker]
        if own is not None and other is not None:
            return other - own, "single"
        if own is None and other is not None:  # talking in the empty slot
            return (1.0, f"known talker in empty slot ({other:.2f})") if other > self.known else (None, "new talker")
        if own is not None and other is None:  # the other slot has no voiceprint yet
            return (1.0, f"new talker, unlike this slot ({own:.2f})") if own < self.new_speaker else (None, "same talker")
        return None, "voiceprints not ready"

    def _decide(self, vectors, sim, stitch_confidence):
        score, label = self._evidence(vectors, sim)
        if stitch_confidence >= self.confident_stitch:
            # Never overrule confident stitching. Overruling it mid-sentence on separated,
            # reverberant speech (lower voiceprint similarity than clean clips) mixed
            # both people into each voice in the first voice-lock test.
            self.last = {"decision": "kept (stitching confident)"}
            return False
        if score is None:
            self.last = {"decision": f"kept ({label})"}
            return False
        margin = self.margin_pair if label == "pair" else self.margin_single
        self.last = {"decision": f"{label}: {'swap' if score > margin else 'keep'} ({score:+.2f})"}
        return score > margin

    def _learn(self, vectors):
        for slot in range(2):
            vector = vectors[slot]
            if vector is None:
                continue
            own, other = self.centroids[slot], self.centroids[1 - slot]
            other_sim = None if other is None else float(vector @ other)
            if own is None:
                # Never start a slot from a voice that sounds like the other slot.
                if other_sim is None or other_sim < self.known:
                    self.centroids[slot], self.updates[slot] = vector, 1
                continue
            own_sim = float(vector @ own)
            if own_sim > 0.25 and (other_sim is None or own_sim > other_sim + self.learn_margin):
                mixed = (1 - self.learn) * own + self.learn * vector
                self.centroids[slot] = mixed / max(float(np.linalg.norm(mixed)), 1e-9)
                self.updates[slot] += 1

    def status(self):
        return {"voiceprints": [c is not None for c in self.centroids], "identity_swaps": self.swaps,
                "updates": list(self.updates), **self.last}
