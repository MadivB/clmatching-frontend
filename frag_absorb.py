"""Small-fragment absorption v0.1 (adopted 2026-07-29, Billy's N-squeeze).

Final chain stage: every label with <= fa_max_hits hits is absorbed into the
label of its nearest non-small hit, provided that hit lies within
fa_radius_cm. Fragments beyond the radius keep their own label.

Why: the label count is dominated by tiny leftover-DBSCAN fragments whose
energies are negligible; absorbing them into their nearest neighbor retires
~140 labels/event at noise-level purity cost while RAISING completeness
(nearest neighbors are overwhelmingly the same interaction).

Validated on 126 events on top of the full frontier chain:
  N 1570 -> 1429 | P 0.9903 -> 0.9901 | C 0.3855 -> 0.3875 | P_track -0.0002
  (radius 5 cm starts pulling fragments across real gaps: P -0.0006 - keep 3)

API: absorb_fragments(labels, split_index, xyz, config) -> (labels, stats).
Config keys (defaults): fa_max_hits 10, fa_radius_cm 3.0,
fa_targets 'any' ('blob' restricts absorption targets to blob labels).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["absorb_fragments"]


def absorb_fragments(labels_global, split_index, xyz, config):
    g = lambda k, d: getattr(config, k, d) if config is not None else d
    max_hits = int(g("fa_max_hits", 10))
    radius = float(g("fa_radius_cm", 3.0))
    targets = str(g("fa_targets", "any"))

    labels = np.asarray(labels_global).copy()
    xyz = np.asarray(xyz, np.float64)
    si = int(split_index)

    u, cnt = np.unique(labels[labels >= 0], return_counts=True)
    small = [int(l) for l, c in zip(u, cnt) if c <= max_hits]
    if not small:
        return labels, {"n_absorbed": 0}
    small_set = np.isin(labels, small)
    if targets == "blob":
        big_mask = (labels >= si) & ~small_set
    else:
        big_mask = (labels >= 0) & ~small_set
    if not big_mask.any():
        return labels, {"n_absorbed": 0}
    tree = cKDTree(xyz[big_mask])
    big_lab = labels[big_mask]

    n_abs = 0
    merges = []
    for l in sorted(small):
        idx = np.flatnonzero(labels == l)
        d, j = tree.query(xyz[idx], k=1)
        k = int(np.argmin(d))
        if d[k] <= radius:
            tgt = int(big_lab[j[k]])
            labels[idx] = tgt
            merges.append((int(l), tgt))
            n_abs += 1
    return labels, {"n_absorbed": n_abs, "merges": merges}
