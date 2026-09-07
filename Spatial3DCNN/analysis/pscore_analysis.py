"""P-score and brain-region aggregation utilities.

The cube-to-region weighting follows the legacy ``score.py`` implementation:
each cube contributes its score in proportion to the fraction of a template
region covered by that cube.
"""
from __future__ import annotations

import numpy as np


def normalize_binary_scores(probabilities, eps=1e-12):
    """Normalize two-class logits/probabilities row-wise to sum to one."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.shape[-1] != 2:
        raise ValueError("probabilities must have two class columns")
    return p / np.maximum(p.sum(axis=-1, keepdims=True), eps)


def combine_cube_scores(weights, probabilities):
    """Apply learned cube weights to AD-vs-HC probabilities."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    p = normalize_binary_scores(probabilities)
    if p.shape[0] != w.size:
        raise ValueError("weights and probabilities must contain the same cubes")
    return p.T @ w


def cube_to_region_scores(cube_scores, cube_region_voxels, region_voxel_counts):
    """Map cube scores to regions using voxel-overlap/template-size weights."""
    scores = np.asarray(cube_scores, dtype=float).reshape(-1)
    overlap = np.asarray(cube_region_voxels, dtype=float)
    if overlap.ndim != 2 or overlap.shape[0] != scores.size:
        raise ValueError("cube_region_voxels must be [n_cubes, n_regions]")
    denom = np.maximum(np.asarray(region_voxel_counts, dtype=float).reshape(-1), 1e-12)
    if overlap.shape[1] != denom.size:
        raise ValueError("region dimensions do not match")
    return (overlap * scores[:, None] / denom[None, :]).sum(axis=0)


def region_scores_from_cubes(cube_scores, cube_region_voxels, region_voxel_counts,
                             normalize=False):
    result = cube_to_region_scores(cube_scores, cube_region_voxels, region_voxel_counts)
    return normalize_region_scores_minmax(result) if normalize else result


def normalize_region_scores_minmax(scores, eps=1e-12):
    x = np.asarray(scores, dtype=float)
    span = x.max() - x.min() if x.size else 0.0
    return np.zeros_like(x) if span <= eps else (x - x.min()) / span


def region_summary_mean_std(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    return np.mean(x, axis=0), np.std(x, axis=0)


def degeneration_regions_from_threshold(scores, mean=None, std=None, z=2.0):
    """Return region indices whose score exceeds mean + z*std."""
    x = np.asarray(scores, dtype=float)
    mu = np.mean(x, axis=0) if mean is None else np.asarray(mean, dtype=float)
    sd = np.std(x, axis=0) if std is None else np.asarray(std, dtype=float)
    return np.flatnonzero(x > mu + z * sd).tolist()


def region_frequency(region_lists):
    """Count how often each region occurs across samples/time points."""
    counts = {}
    for regions in region_lists:
        for region in set(regions):
            key = int(region)
            counts[key] = counts.get(key, 0) + 1
    return counts


def total_brain_score(region_scores, weights=None):
    x = np.asarray(region_scores, dtype=float)
    return float(np.sum(x)) if weights is None else float(np.sum(x * np.asarray(weights)))
