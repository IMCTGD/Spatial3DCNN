"""Longitudinal degeneration and cognitive-score trend analyses."""
from __future__ import annotations

import numpy as np


def is_non_decreasing(values, atol=0.0):
    x = np.asarray(values, dtype=float).reshape(-1)
    return bool(x.size > 0 and np.all(np.diff(x) >= -atol))


def final_increase(values, atol=0.0):
    x = np.asarray(values, dtype=float).reshape(-1)
    return bool(x.size >= 2 and x[-1] > x[0] + atol)


def new_regions_over_time(region_lists):
    seen, result = set(), []
    for regions in region_lists:
        current = set(int(r) for r in regions)
        result.append(sorted(current - seen)); seen.update(current)
    return result


def continuous_growth_statistics(region_trajectories, atol=0.0):
    """Summarize continuously increasing and final-increasing trajectories."""
    trajectories = {int(r): np.asarray(v, dtype=float) for r, v in region_trajectories.items()}
    return {
        "continuous": [r for r, v in trajectories.items() if is_non_decreasing(v, atol)],
        "final_increase": [r for r, v in trajectories.items() if final_increase(v, atol)],
    }


def total_score_by_mmse(total_scores, mmse, normalize=True):
    scores, cognitive = np.asarray(total_scores, float), np.asarray(mmse, float)
    if scores.shape != cognitive.shape:
        raise ValueError("total_scores and mmse must have the same shape")
    if normalize and scores.size:
        span = scores.max() - scores.min()
        scores = (scores - scores.min()) / span if span > 1e-12 else np.zeros_like(scores)
    groups = {}
    for score, value in zip(scores, cognitive):
        groups.setdefault(float(value), []).append(float(score))
    return {key: float(np.mean(value)) for key, value in groups.items()}
