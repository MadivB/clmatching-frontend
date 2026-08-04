"""Targeted big-blob refinement v0.1 (adopted design, 2026-07-29).

Runs as the LAST chain stage, after vertex pinpointing. For every blob label
(id >= split_index) with at least br_min_hits hits, re-cluster its hits with a
fine DBSCAN(br_eps, br_min_samples). If that finds >= 2 sub-clusters, the
LARGEST sub-cluster and ALL noise points keep the parent label and every other
sub-cluster gets a fresh label.

Why: the purity budget showed 62.6% of all foreign energy lives in large mixed
shower blobs glued together by the coarse leftover DBSCAN (eps 4). Refining
only large blobs spends new labels exactly where the mixing is, and keeping
noise points on the parent label makes energy demotion structurally impossible
- every hit stays labeled, nothing is hidden from the light matcher.

Validated on 126 events vs the adopted B3 chain (T=500, eps=2.5):
  P 0.9879 -> 0.9902 | C 0.3916 -> 0.3853 (blob-side fragmentation only)
  P_track 0.9930 flat | foreign energy -18% | N/event 1493 -> 1570 (cap 1600)

API: refine_blobs(labels, split_index, xyz, config) -> (labels, stats).
Config keys (defaults): br_min_hits 500, br_eps 2.5, br_min_samples 3.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

__all__ = ["refine_blobs"]


def refine_blobs(labels_global, split_index, xyz, config):
    g = lambda k, d: getattr(config, k, d) if config is not None else d
    min_hits = int(g("br_min_hits", 500))
    eps = float(g("br_eps", 2.5))
    min_samples = int(g("br_min_samples", 3))

    labels = np.asarray(labels_global).copy()
    xyz = np.asarray(xyz, np.float64)
    si = int(split_index)

    nxt = int(labels.max()) + 1
    refined = []
    u_lab, cnt = np.unique(labels[labels >= si], return_counts=True)
    for l, c in zip(u_lab, cnt):
        if c < min_hits:
            continue
        idx = np.flatnonzero(labels == l)
        sub = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz[idx])
        u, ucnt = np.unique(sub[sub >= 0], return_counts=True)
        if len(u) < 2:
            continue
        keep = u[ucnt.argmax()]
        new_ids = []
        for uu in u:
            if uu == keep:
                continue
            labels[idx[sub == uu]] = nxt
            new_ids.append(nxt)
            nxt += 1
        # noise points (sub == -1) keep the parent label: zero demotion
        refined.append({"label": int(l), "n_hits": int(c),
                        "n_fragments": int(len(u)), "new_labels": new_ids})
    return labels, {"refined": refined,
                    "n_new_labels": sum(len(r["new_labels"]) for r in refined)}
