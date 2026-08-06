"""Spatial link score J_ij = P(same interaction | pair geometry) v0.

First coded piece of the iterative matching stage (MATCHING_STAGE_DESIGN.md
section 2.2): a calibrated pairwise link between clusters, fitted on the 126
truth events (files 1-8 train, 9-10 test). Because a common t0 shifts both
clusters' drift coordinates together, pair geometry at a shared slot is
t-independent -> J is precomputed once per event; only per-cluster
containment depends on t.

Calibration v0 (test files 9-10): AUC 0.649 vs 0.556 distance-only; bins
honest (J 0.80-0.90 -> 87.4% actual, 0.90-0.95 -> 92.5%, >=0.95 -> 94.6%).
Base rate of candidate pairs (min hit distance <= 12 cm) is 86.7% - the
link's job is tie-breaking on the hard minority, not standalone
classification. Feature signs tell the session's doppelganger story:
contact and collinearity build trust (+); a very line-like partner and
track-track pairs erode it (-) - nearby clean tracks are often OTHER
interactions passing through.

API:
  pair_features(A, B) -> feature vector (A, B from cluster_summary)
  cluster_summary(idx, xyz, e, is_track, rng) -> summary dict
  LinkModel(path).score(feats) / .score_pairs(list) -> J in [0, 1]
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["cluster_summary", "pair_features", "LinkModel"]

SUB = 300          # per-cluster hit subsample used at fit time (rng(0))


def cluster_summary(idx, xyz, e, is_track, rng=None):
    """Per-cluster geometry summary matching the calibration harvest."""
    rng = rng or np.random.default_rng(0)
    pts = xyz[idx]
    c = pts.mean(0)
    q = pts - c
    try:
        _, sv, vt = np.linalg.svd(q, full_matrices=False)
        lin = float(sv[0] ** 2 / max(float((sv ** 2).sum()), 1e-12))
        udir = vt[0]
    except np.linalg.LinAlgError:
        lin, udir = 0.0, np.array([1.0, 0.0, 0.0])
    s = idx if len(idx) <= SUB else rng.choice(idx, SUB, replace=False)
    return dict(cent=c, dir=udir, lin=lin, n=len(idx),
                e=float(e[idx].sum()), sub=xyz[s], track=bool(is_track))


def pair_features(A, B):
    """Feature vector for one cluster pair (order-independent)."""
    d, _ = cKDTree(B['sub']).query(A['sub'], k=1)
    d_min = float(d.min())
    n3 = int((d <= 3.0).sum())
    return np.array([
        d_min,
        np.log1p(n3),
        float(abs(A['dir'] @ B['dir'])),
        min(A['lin'], B['lin']),
        max(A['lin'], B['lin']),
        np.log10(max(min(A['e'], B['e']), 0.1)),
        np.log10(max(max(A['e'], B['e']), 0.1)),
        np.log10(min(A['n'], B['n'])),
        float(A['track'] and B['track']),
        float(np.linalg.norm(A['cent'] - B['cent'])),
    ])


class LinkModel:
    def __init__(self, path='link_model_v0.json'):
        m = json.load(open(path))
        self.mean = np.asarray(m['mean'])
        self.scale = np.asarray(m['scale'])
        self.coef = np.asarray(m['coef'])
        self.intercept = float(m['intercept'])
        self.features = list(m['features'])

    def score(self, feats):
        z = (np.asarray(feats) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept))))

    def score_pairs(self, feat_rows):
        Z = (np.asarray(feat_rows) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(Z @ self.coef + self.intercept)))
