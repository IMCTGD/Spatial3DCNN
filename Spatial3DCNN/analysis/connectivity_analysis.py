"""Spatial and spatiotemporal connectivity of degenerated regions."""
from __future__ import annotations

import numpy as np


def build_region_adjacency(mask, connectivity=6):
    """Build region adjacency from a 3-D integer atlas mask."""
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26")
    atlas = np.asarray(mask)
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
               for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    if connectivity == 6:
        offsets = [o for o in offsets if sum(abs(v) for v in o) == 1]
    elif connectivity == 18:
        offsets = [o for o in offsets if sum(abs(v) for v in o) <= 2]
    adjacency = {}
    for axis, off in enumerate(offsets):
        src = tuple(slice(max(0, -d), atlas.shape[q] - max(0, d)) for q, d in enumerate(off))
        dst = tuple(slice(max(0, d), atlas.shape[q] - max(0, -d)) for q, d in enumerate(off))
        a, b = atlas[src], atlas[dst]
        pairs = np.unique(np.stack([a, b], axis=-1).reshape(-1, 2), axis=0)
        for left, right in pairs:
            left, right = int(left), int(right)
            if left > 0 and right > 0 and left != right:
                adjacency.setdefault(left, set()).add(right)
                adjacency.setdefault(right, set()).add(left)
    return adjacency


def connected_components(regions, adjacency):
    remaining = set(int(r) for r in regions)
    components = []
    while remaining:
        root = remaining.pop(); stack = [root]; component = [root]
        while stack:
            for nxt in adjacency.get(stack.pop(), ()):
                if nxt in remaining:
                    remaining.remove(nxt); stack.append(nxt); component.append(nxt)
        components.append(sorted(component))
    return sorted(components, key=lambda c: (c[0], len(c)))


def spatial_components(regions, adjacency):
    return connected_components(regions, adjacency)


def spatiotemporal_components(time_region_lists, adjacency):
    """Connect regions spatially or when present in adjacent time points."""
    components = []
    previous = set()
    for regions in time_region_lists:
        current = set(int(r) for r in regions)
        components.extend(connected_components(current, adjacency))
        # Temporal persistence is represented as the union of consecutive sets.
        if previous and current:
            components.extend(connected_components(previous | current, adjacency))
        previous = current
    return components


def component_counts(region_lists, adjacency):
    comps = [spatial_components(regions, adjacency) for regions in region_lists]
    return [len(c) for c in comps], comps
