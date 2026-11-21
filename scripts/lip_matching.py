"""Offline, untrained lip-rhythm / separated-audio association.

Correlation scores are diagnostics, not calibrated speaker probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class MatchConfig:
    fps: int = 25
    window_seconds: float = 2.0
    step_seconds: float = .2
    max_lag_seconds: float = .24
    min_correlation: float = .30
    min_margin: float = .10
    min_valid_fraction: float = .8
    min_mouth_std: float = .004
    hold_seconds: float = 1.0
    switch_confirmation_windows: int = 2


class FaceTracker:
    """Keep two geometric track slots stable when detector output order changes."""
    def __init__(self, max_distance=.25):
        self.positions = [None, None]
        self.max_distance = max_distance

    def assign(self, detections):
        result = [None, None]
        centers = [np.asarray(item["center"]) for item in detections]
        occupied = [i for i, p in enumerate(self.positions) if p is not None]
        used = set()
        if occupied and centers:
            costs = np.array([[np.linalg.norm(self.positions[i] - center)
                               for center in centers] for i in occupied])
            rows, cols = linear_sum_assignment(costs)
            for row, col in zip(rows, cols):
                if costs[row, col] <= self.max_distance:
                    slot = occupied[row]
                    result[slot] = detections[col]
                    self.positions[slot] = centers[col]
                    used.add(int(col))
        # Initial ordering is left-to-right. Existing slots are not silently
        # overwritten by a distant new face or a camera cut.
        free = [i for i, p in enumerate(self.positions) if p is None]
        remaining = sorted((i for i in range(len(detections)) if i not in used),
                           key=lambda i: centers[i][0])
        for slot, index in zip(free, remaining):
            result[slot] = detections[index]
            self.positions[slot] = centers[index]
        return result


def audio_envelopes(candidates, sample_rate, frames, fps):
    if sample_rate % fps:
        raise ValueError("Audio sample rate must be divisible by analysis FPS.")
    hop = sample_rate // fps
    if candidates.shape != (frames * hop, 2):
        raise ValueError("Candidate audio must be sample-aligned with the video frame grid.")
    return np.sqrt(np.mean(candidates.reshape(frames, hop, 2).astype(np.float64) ** 2, axis=1))


def filtered_signal(signal, fps):
    valid = np.isfinite(signal)
    if np.count_nonzero(valid) < 20:
        return np.full(len(signal), np.nan)
    positions = np.arange(len(signal))
    filled = np.interp(positions, positions[valid], signal[valid])
    sos = butter(2, [.7, 7], fs=fps, btype="bandpass", output="sos")
    # This is explicitly an offline analysis: zero-phase filtering uses future
    # context. Missing-frame masks are still enforced in every matching window.
    return sosfiltfilt(sos, filled)


def correlation(a, b):
    a, b = a - np.mean(a), b - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-10 else float("nan")


def match_signals(mouths, envelopes, config=MatchConfig()):
    """Return per-frame source selections and auditable rolling correlation scores."""
    mouths, envelopes = np.asarray(mouths), np.asarray(envelopes)
    if mouths.shape != envelopes.shape or mouths.ndim != 2 or mouths.shape[1] != 2:
        raise ValueError("Expected two mouth traces and two envelopes, both shaped (frames, 2).")
    frames = len(mouths)
    window = round(config.window_seconds * config.fps)
    step = max(1, round(config.step_seconds * config.fps))
    max_lag = round(config.max_lag_seconds * config.fps)
    lips = np.column_stack([filtered_signal(mouths[:, i], config.fps) for i in range(2)])
    sounds = []
    for source in range(2):
        raw = envelopes[:, source]
        floor = max(float(np.percentile(raw, 10)), float(np.max(raw)) * .02, 1e-8)
        sounds.append(filtered_signal(np.log1p(raw / floor), config.fps))
    sounds = np.column_stack(sounds)
    selections = np.full((frames, 2), -1, dtype=int)
    statuses = np.full((frames, 2), "uncertain", dtype=object)
    last_source = [-1, -1]
    last_confident = [-100000, -100000]
    pending_source, pending_count = [-1, -1], [0, 0]
    records = []
    for end in range(window, frames + 1, step):
        start = end - window
        scores = np.full((2, 2), np.nan)
        lags = np.zeros((2, 2), dtype=int)
        proposals = []
        for face in range(2):
            valid = np.isfinite(mouths[start:end, face])
            enough = np.mean(valid) >= config.min_valid_fraction
            moving = enough and np.std(mouths[start:end, face][valid]) >= config.min_mouth_std
            if moving:
                for source in range(2):
                    active = np.mean(envelopes[start:end, source]) > max(1e-7, .03 * np.max(envelopes[:, source]))
                    if not active:
                        scores[face, source] = 0.0
                        continue
                    for lag in range(-max_lag, max_lag + 1):
                        indices = np.arange(start, end)
                        valid_indices = valid & (indices + lag >= 0) & (indices + lag < frames)
                        if np.mean(valid_indices) < config.min_valid_fraction:
                            continue
                        indices = indices[valid_indices]
                        score = correlation(lips[indices, face], sounds[indices + lag, source])
                        if np.isfinite(score) and (not np.isfinite(scores[face, source]) or score > scores[face, source]):
                            scores[face, source], lags[face, source] = score, lag
            finite = np.isfinite(scores[face])
            chosen, status = -1, "uncertain"
            if finite.all():
                best = int(np.argmax(scores[face]))
                if (scores[face, best] >= config.min_correlation
                        and scores[face, best] - scores[face, 1-best] >= config.min_margin):
                    chosen, status = best, "matched"
            visible_now = np.isfinite(mouths[end-1, face])
            if not visible_now:
                chosen, status = -1, "uncertain"
            if chosen >= 0 and last_source[face] >= 0 and chosen != last_source[face]:
                pending_count[face] = pending_count[face]+1 if pending_source[face] == chosen else 1
                pending_source[face] = chosen
                if pending_count[face] < config.switch_confirmation_windows:
                    chosen, status = -1, "uncertain"
            else:
                pending_source[face], pending_count[face] = -1, 0
            if chosen < 0 and visible_now and end-last_confident[face] <= config.hold_seconds * config.fps:
                chosen, status = last_source[face], "held"
            proposals.append([chosen, status])
        # A source cannot be assigned to both faces in the same decision window.
        claimed = set()
        order = sorted(range(2), key=lambda face: (
            proposals[face][1] == "matched",
            float(scores[face, proposals[face][0]]) if proposals[face][0] >= 0 and np.isfinite(scores[face, proposals[face][0]]) else -1,
        ), reverse=True)
        for face in order:
            chosen, status = proposals[face]
            if chosen >= 0 and chosen in claimed:
                chosen, status = -1, "uncertain"
            if chosen >= 0:
                claimed.add(chosen)
            if status == "matched":
                last_source[face], last_confident[face] = chosen, end
            selections[end-1:min(frames, end-1+step), face] = chosen
            statuses[end-1:min(frames, end-1+step), face] = status
            proposals[face] = [chosen, status]
        records.append({
            "time_seconds": (end-1) / config.fps,
            "scores": [[None if not np.isfinite(v) else round(float(v), 5) for v in row] for row in scores],
            "audio_lag_seconds": (lags / config.fps).tolist(),
            "selected_sources": [p[0] for p in proposals], "statuses": [p[1] for p in proposals],
        })
    # A coarse decision must not extrapolate through a subsequently lost face.
    lost = ~np.isfinite(mouths)
    selections[lost] = -1
    statuses[lost] = "face_lost"
    return selections, statuses, records


def selected_audio(candidates, selections, sample_rate, fps):
    hop = sample_rate // fps
    if candidates.shape != (len(selections) * hop, 2):
        raise ValueError("Selection and audio timelines do not align.")
    result = np.zeros_like(candidates)
    for face in range(2):
        weights = np.column_stack([(selections[:, face] == source).astype(float) for source in range(2)])
        weights = np.repeat(weights, hop, axis=0)
        # A 20 ms crossfade avoids clicks when selecting, switching, or muting.
        weights = uniform_filter1d(weights, max(1, round(.02 * sample_rate)), axis=0, mode="nearest")
        result[:, face] = np.sum(candidates * weights, axis=1)
    return result


def analyze(mouths, candidates, sample_rate, config=MatchConfig()):
    envelopes = audio_envelopes(candidates, sample_rate, len(mouths), config.fps)
    selections, statuses, records = match_signals(mouths, envelopes, config)
    shifted = np.full_like(mouths, np.nan)
    shifted[config.fps:] = mouths[:-config.fps]
    control_selections, _, control_records = match_signals(shifted, envelopes, config)
    differences, by_face = [], [[], []]
    for actual, control in zip(records, control_records):
        for face in range(2):
            a, b = actual["scores"][face], control["scores"][face]
            if all(v is not None for v in a+b):
                differences.append(max(a)-max(b))
                by_face[face].append(max(a)-max(b))
    improvement = float(np.mean(differences)) if differences else None
    summary = {
        "algorithm": "Offline mouth-opening / speech-envelope correlation; no pretrained lip-reading model",
        "config": asdict(config), "scores_are_probabilities": False,
        "uses_future_context": True, "accuracy": "not_measured_no_ground_truth",
        "faces": [{"face_id": face,
                   "detection_fraction": float(np.mean(np.isfinite(mouths[:, face]))),
                   "matched_fraction": float(np.mean(statuses[:, face] == "matched")),
                   "held_fraction": float(np.mean(statuses[:, face] == "held")),
                   "uncertain_fraction": float(np.mean(selections[:, face] < 0)),
                   "shift_control_mean_advantage": float(np.mean(by_face[face])) if by_face[face] else None,
                   "shift_control": "advantage_observed" if by_face[face] and np.mean(by_face[face]) > .05 else "inconclusive",
                   "selected_seconds_by_source": [float(np.count_nonzero(selections[:, face] == i) / config.fps) for i in range(2)]}
                  for face in range(2)],
        "negative_control": {
            "video_shift_seconds": 1.0,
            "mean_best_correlation_advantage": improvement,
            "timing_check": "advantage_observed" if improvement is not None and improvement > .05 else "inconclusive",
            "shifted_selected_fraction": float(np.mean(control_selections >= 0)),
            "note": "A diagnostic comparison, not accuracy or a statistical significance test.",
        },
    }
    return selections, statuses, records, summary
