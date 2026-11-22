"""Score "which separated voice belongs to the face on camera" on GRID conversations.

No recordings of us and no calibration: test conversations are built from the public
GRID audiovisual corpus (CC BY 4.0, 25 fps frontal video with its own audio), where
the truth is known. The face on screen is talker A; talker B is a second, off-camera
voice. Two kinds of test:

- overlap: A and B talk at the same time (B at -5..+5 dB), the hardest case;
- turns:   A and B alternate; while B talks, A's face is silent (A's own pre-speech
           frames, played back and forth).

The candidates stand in for SepFormer's two outputs, including its leakage
(each holds the other talker 15 dB down), in random order. Each picker is replayed
hop by hop like the live server (2 s window every 0.25 s), with the phone's frame
rate simulated (25 / 15 / 10 fps, frames held in between).

  python scripts/eval_lip_pick.py --grid external/datasets/grid --cases 20
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RATE, FPS = 16000, 25
SPF = RATE // FPS          # audio samples per video frame
WINDOW, HOP = 2.0, 0.25
LEAK_DB = -15.0


# ---------------------------------------------------------------- data
def load_clip(path, landmarker_factory, cache):
    target = cache / f"{path.parent.name}_{path.stem}.npz"
    if target.exists():
        d = np.load(target)
        return {k: d[k] for k in d.files}
    import cv2
    import test2
    from processor.active_speaker import face_crop
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    capture = cv2.VideoCapture(str(path))
    landmarker = landmarker_factory()
    openness, crops, valid = [], [], []
    k = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        _, info = test2.process_frame(landmarker, frame.copy(), int(k * 1000 / FPS))
        h, w = frame.shape[:2]
        crop = None
        if info["face_found"]:
            result = landmarker_last(landmarker)
            if result is not None:
                xs, ys = result[:, 0] * w, result[:, 1] * h
                crop = face_crop(frame, (xs.min(), ys.min(), xs.max(), ys.max()))
        openness.append(info["openness"] if info["face_found"] else np.nan)
        valid.append(crop is not None)
        crops.append(crop if crop is not None else np.full((112, 112), 110, np.uint8))
        k += 1
    landmarker.close()
    n = min(k, len(audio) // SPF)
    clip = {"audio": audio[: n * SPF], "openness": np.array(openness[:n], float),
            "crops": np.array(crops[:n], np.uint8), "valid": np.array(valid[:n], bool)}
    np.savez_compressed(target, **clip)
    return clip


_last = {}


def landmarker_factory():
    """MediaPipe landmarker that remembers the last target face's landmarks (for the box)."""
    import test2
    lm = test2.create_landmarker()
    original = lm.detect_for_video

    def detect(image, ts):
        result = original(image, ts)
        faces = result.face_landmarks or []
        _last[id(lm)] = (np.array([(p.x, p.y) for p in max(faces, key=lambda f: _area(f))]) if faces else None)
        return result
    lm.detect_for_video = detect
    return lm


def _area(face):
    xs, ys = [p.x for p in face], [p.y for p in face]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def landmarker_last(lm):
    return _last.get(id(lm))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def silent_frames(clip):
    """Indices of frames before the talker starts (GRID clips begin with ~0.5-1 s of silence)."""
    energy = np.sqrt(np.mean(clip["audio"].reshape(-1, SPF) ** 2, axis=1))
    speaking = np.nonzero(energy > 0.1 * energy.max())[0]
    first = int(speaking[0]) if len(speaking) else len(energy)
    lead = list(range(0, max(0, first - 2)))
    return [i for i in lead if clip["valid"][i]] or [0]


def pingpong(indices, n):
    seq = indices + indices[-2:0:-1] if len(indices) > 2 else indices
    return [seq[i % len(seq)] for i in range(n)]


def build_case(kind, clips_a, clips_b, rng):
    """Returns face timeline (crops, openness, valid), source audios a and b."""
    crops, openness, valid, a, b = [], [], [], [], []

    def add_face(clip, idx):
        crops.append(clip["crops"][idx]); openness.append(clip["openness"][idx]); valid.append(clip["valid"][idx])

    if kind == "overlap":
        for clip in clips_a:
            for i in range(len(clip["valid"])):
                add_face(clip, i)
            a.append(clip["audio"])
        b_audio = np.concatenate([c["audio"] for c in clips_b])
        shift = int(rng.integers(0, RATE))
        b_audio = np.concatenate([np.zeros(shift, np.float32), b_audio])
        total = sum(len(x) for x in a)
        b = [np.pad(b_audio, (0, max(0, total - len(b_audio))))[:total]]
    else:  # turns: A, B, A, B ...
        for clip_a, clip_b in zip(clips_a, clips_b):
            for i in range(len(clip_a["valid"])):
                add_face(clip_a, i)
            a.append(clip_a["audio"]); b.append(np.zeros_like(clip_a["audio"]))
            still = pingpong(silent_frames(clip_a), len(clip_b["valid"]))
            for i in still:
                add_face(clip_a, i)
            a.append(np.zeros_like(clip_b["audio"])); b.append(clip_b["audio"])
    a, b = np.concatenate(a), np.concatenate(b)
    gain = 10 ** (rng.uniform(-5, 5) / 20) * rms(a[a != 0]) / max(rms(b[b != 0]), 1e-9)
    b = b * gain
    n = min(len(crops), len(a) // SPF)
    return (np.array(crops[:n]), np.array(openness[:n]), np.array(valid[:n]), a[: n * SPF], b[: n * SPF])


def phone_hold(n, fps):
    """For each 25 fps slot, the index of the latest frame the phone would have sent."""
    if fps >= FPS:
        return np.arange(n), np.arange(n)
    sent = np.unique(np.floor(np.arange(0, n / FPS, 1 / fps) * FPS).astype(int))
    sent = sent[sent < n]
    held = sent[np.searchsorted(sent, np.arange(n), side="right") - 1]
    return sent, held


# ---------------------------------------------------------------- pickers
def load_old_picker():
    """The per-window picker of the known-good version (tag good-focus-78db57a)."""
    source = subprocess.run(["git", "show", "good-focus-78db57a:processor/separation.py"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    path = Path(__file__).resolve().parent / "_old_separation.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_old_separation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path.unlink()
    return module.LipPicker


class OpennessRunner:
    """Feeds a LipPicker (old or current) exactly as the live server does."""
    def __init__(self, picker_cls):
        self.cls = picker_cls

    def run(self, case, sent, candidates):
        _, openness, valid, _, _ = case
        picker = self.cls()
        picks = []
        added = 0
        for end in hop_ends(len(valid)):
            start = end - int(WINDOW * FPS)
            while added < len(sent) and sent[added] < end:
                i = sent[added]
                picker.add(int(i * 1e6 / FPS), float(openness[i]) if valid[i] and np.isfinite(openness[i]) else None)
                added += 1
            window = candidates[start * SPF: end * SPF]
            picks.append(picker.pick(window, int(start * 1e6 / FPS), RATE))
        return np.array(picks)


class AsdRunner:
    """Light-ASD scores per separated voice; the decision rule lives in processor.active_speaker."""
    def __init__(self, model, decide_cls, **decide_args):
        self.model, self.decide_cls, self.args = model, decide_cls, decide_args

    def embeddings(self, case, held, candidates):
        import torch
        crops = case[0][held]
        m = self.model
        with torch.no_grad():
            v = m.model.forward_visual_frontend(torch.as_tensor(crops, dtype=torch.float32)[None])
            audios = []
            for k in range(2):
                from processor.active_speaker import mfcc
                f = mfcc(candidates[:, k])
                need = 4 * len(crops)
                f = np.pad(f, ((0, max(0, need - len(f))), (0, 0)), mode="edge")[:need]
                audios.append(m.model.forward_audio_frontend(torch.as_tensor(f, dtype=torch.float32)[None]))
        return v, audios

    def run(self, case, held, candidates, cached=None):
        import torch
        v, audios = cached if cached is not None else self.embeddings(case, held, candidates)
        valid = case[2][held]
        decide = self.decide_cls(**self.args)
        picks = []
        with torch.no_grad():
            for end in hop_ends(len(valid)):
                start = end - int(WINDOW * FPS)
                scores = []
                for a in audios:
                    out = self.model.model.forward_audio_visual_backend(a[:, start:end] , v[:, start:end])
                    logits = self.model.fc(out)
                    scores.append((logits[:, 1] - logits[:, 0]).numpy())
                picks.append(decide.update(np.array(scores), valid[start:end], HOP))
        return np.array(picks)


class TalkMatchRunner:
    """Light-ASD visual speaking detector matched against each voice's activity."""
    def __init__(self, model, **args):
        self.model, self.args = model, args

    def run(self, case, held, candidates):
        from processor.active_speaker import TalkMatch, voice_activity
        speaking = self.model.speaking(case[0][held])
        valid = case[2][held]
        decide = TalkMatch(**self.args)
        picks = []
        for end in hop_ends(len(valid)):
            start = end - int(WINDOW * FPS)
            activity = voice_activity(candidates[start * SPF: end * SPF])
            picks.append(decide.update(speaking[start:end], activity, valid[start:end], HOP))
        return np.array(picks)


class ComboRunner:
    """Computes Light-ASD once per case, then replays FacePicker for every setting in `grid`."""
    def __init__(self, model, grid):
        self.model, self.grid = model, grid

    def run_all(self, case, held, candidates):
        import torch
        from processor.active_speaker import FacePicker, mfcc, voice_activity
        m = self.model
        crops, valid = case[0][held], case[2][held]
        with torch.no_grad():
            v = m.model.forward_visual_frontend(torch.as_tensor(crops, dtype=torch.float32)[None])
            speaking = (lambda o: (o[:, 1] - o[:, 0]).numpy())(m.fcv(v[0]))
            audio = []
            for k in range(2):
                f = mfcc(candidates[:, k])
                need = 4 * len(crops)
                f = np.pad(f, ((0, max(0, need - len(f))), (0, 0)), mode="edge")[:need]
                audio.append(m.model.forward_audio_frontend(torch.as_tensor(f, dtype=torch.float32)[None]))
            hops = []
            for end in hop_ends(len(valid)):
                start = end - int(WINDOW * FPS)
                sync = []
                for a in audio:
                    o = m.fc(m.model.forward_audio_visual_backend(a[:, start:end], v[:, start:end]))
                    sync.append((o[:, 1] - o[:, 0]).numpy())
                hops.append((np.array(sync), speaking[start:end],
                             voice_activity(candidates[start * SPF: end * SPF]), valid[start:end]))
        results = {}
        for label, params in self.grid.items():
            picker = FacePicker(**params)
            results[label] = np.array([picker.update(*h, HOP) for h in hops])
        return results


def hop_ends(n):
    first = int(WINDOW * FPS)
    return list(range(first, n + 1, int(HOP * FPS)))


# ---------------------------------------------------------------- scoring
def score(picks, truth):
    picks = np.asarray(picks)
    correct = picks == truth
    first = int(np.argmax(correct)) if correct.any() else len(picks)
    switches = int(np.sum((picks[1:] != picks[:-1]) & (picks[1:] >= 0) & (picks[:-1] >= 0)))
    return {"correct": float(correct.mean()), "wrong": float(np.mean((picks >= 0) & ~correct)),
            "none": float(np.mean(picks < 0)), "first_correct_s": WINDOW + first * HOP,
            "switches": switches}


def summarize(rows):
    keys = ("correct", "wrong", "none", "first_correct_s", "switches")
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys} | {"cases": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=ROOT / "external" / "datasets" / "grid")
    parser.add_argument("--cache", type=Path, default=ROOT / "external" / "datasets" / "grid_cache")
    parser.add_argument("--cases", type=int, default=12, help="per kind")
    parser.add_argument("--fps", type=int, nargs="+", default=[25, 10])
    parser.add_argument("--pickers", nargs="*", default=["old", "evidence", "asd-talkset", "asd-ava"])
    parser.add_argument("--out", type=Path, default=ROOT / "validation-output" / "lip-pick-eval.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--combo", nargs="*", default=[], help="Light-ASD weights to grid-search FacePicker with")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    talkers = sorted(p for p in args.grid.iterdir() if p.is_dir())
    t0 = time.perf_counter()
    clips = {t.name: [load_clip(p, landmarker_factory, args.cache) for p in sorted(t.glob("*.mpg"))] for t in talkers}
    clips = {k: [c for c in v if c["valid"].mean() > 0.9] for k, v in clips.items()}
    print(f"clips ready in {time.perf_counter() - t0:.0f} s:", {k: len(v) for k, v in clips.items()}, flush=True)

    from processor.separation import LipPicker
    from processor.active_speaker import ActiveSpeaker, AsdDecision
    runners, models = {}, {}

    def parse(text):
        return {k: float(v) for k, v in (item.split("=") for item in text.split(",") if item)}
    for name in args.pickers:
        if name == "old":
            runners[name] = OpennessRunner(load_old_picker())
        elif name == "evidence":
            runners[name] = OpennessRunner(LipPicker)
        elif name.startswith("asd-"):
            weights, _, extra = name[4:].partition("@")
            runners[name] = AsdRunner(ActiveSpeaker(weights), AsdDecision, **parse(extra))
        elif name.startswith("talk-"):
            weights, _, extra = name[5:].partition("@")
            runners[name] = TalkMatchRunner(models.setdefault(weights, ActiveSpeaker(weights)), **parse(extra))

    if args.combo:
        grid = {}
        for w in (0.0, 0.5, 1.0, 2.0):
            for tau in (1.5, 3.0):
                for enter, switch in ((0.15, 0.3), (0.25, 0.5), (0.35, 0.7)):
                    grid[f"combo w={w} tau={tau} th={enter}/{switch}"] = dict(
                        sync_weight=w, tau_s=tau, enter=enter, switch=switch)
        for weights in args.combo:
            runners[f"combo-{weights}"] = ComboRunner(models.setdefault(weights, ActiveSpeaker(weights)), grid)
    rng = np.random.default_rng(args.seed)
    names = sorted(clips)
    results = {}
    for kind in ("overlap", "turns"):
        for c in range(args.cases):
            ta, tb = rng.choice(len(names), 2, replace=False)
            pa = [clips[names[ta]][i] for i in rng.choice(len(clips[names[ta]]), 3, replace=False)]
            pb = [clips[names[tb]][i] for i in rng.choice(len(clips[names[tb]]), 3, replace=False)]
            case = build_case(kind, pa, pb, rng)
            _, _, _, a, b = case
            leak = 10 ** (LEAK_DB / 20)
            truth = int(rng.integers(0, 2))
            pair = [a + leak * b, b + leak * a]
            candidates = np.stack([pair[0], pair[1]] if truth == 0 else [pair[1], pair[0]], axis=1)
            for fps in args.fps:
                sent, held = phone_hold(len(case[2]), fps)
                for name, runner in runners.items():
                    if isinstance(runner, ComboRunner):
                        for label, picks in runner.run_all(case, held, candidates).items():
                            results.setdefault((f"{name[6:]} {label}", kind, fps), []).append(score(picks, truth))
                        continue
                    if isinstance(runner, (AsdRunner, TalkMatchRunner)):
                        picks = runner.run(case, held, candidates)
                    else:
                        picks = runner.run(case, sent, candidates)
                    results.setdefault((name, kind, fps), []).append(score(picks, truth))
            print(f"{kind} case {c + 1}/{args.cases} ({names[ta]} on camera, {names[tb]} off) "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)

    table = {f"{n} | {k} | {f} fps": summarize(r) for (n, k, f), r in sorted(results.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2))
    print(f"\n{'picker | test | phone fps':40s} correct  wrong   none  first(s) switches")
    for key, s in table.items():
        print(f"{key:40s} {s['correct']:6.0%} {s['wrong']:6.0%} {s['none']:6.0%} {s['first_correct_s']:8.2f} {s['switches']:6.2f}")


if __name__ == "__main__":
    main()
