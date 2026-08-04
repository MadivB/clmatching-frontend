"""MIP peel v0.1 (Billy's design: 'fit one line, the rest is another blob').

Surgery for dE/dx-doubling-flagged track windows. For each flagged
(track label, TPC) window:

  1. Robust single-line fit (mini-RANSAC + one refit round): hits within
     mp_inlier_cm of the best line are the MIP core - they KEEP the track
     label.
  2. The remainder is grouped into connected components with a coarse
     DBSCAN (mp_rem_eps). ONLY components that are themselves MIP-like
     (>= mp_comp_min_hits hits, >= mp_comp_min_len_cm long, linearity >=
     mp_comp_min_lin) AND that axially OVERLAP the kept core (fraction of
     component hits projecting inside the core's span >= mp_comp_overlap;
     the dE/dx doubling that raised the flag happens exactly in the overlap)
     are peeled to fresh BLOB labels. Diffuse halo stays on the track, and -
     critically - so do line-like continuations BEYOND the core's ends:
     those are the same particle curving out of the straight-line cylinder,
     and peeling them breaks real tracks (the v0.2 failure mode).

v0.1 lesson (126 ev): peeling the whole remainder is purity-monotone but
costs -0.017 completeness and +45 labels/ev (95% of peels were own-halo).
The line-like component gate keeps the crossers and drops the halo peels.

No cross-label adjudication: charge cannot decide whether the peeled line
is another interaction, so it is demoted to blob semantics and the light
matcher decides.

API: peel_flagged(labels, split_index, label_info, xyz, e, tpc, config)
     -> (labels, stats)
Config keys (defaults): mp_iters 40, mp_inlier_cm 1.2, mp_min_kept 15,
mp_min_kept_len_cm 8.0, mp_rem_eps 4.0, mp_rem_min_samples 3,
mp_comp_min_hits 15, mp_comp_min_len_cm 8.0, mp_comp_min_lin 0.85,
mp_comp_overlap 0.6.
"""
from __future__ import annotations

import os
from importlib import util as _util
from types import SimpleNamespace

import numpy as np
from sklearn.cluster import DBSCAN

__all__ = ["peel_flagged"]


def _line_fit(pts):
    c = pts.mean(0)
    q = pts - c
    try:
        _, _, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    u = vt[0]
    return c, u


def _robust_line_inliers(pts, iters, r_in, rng):
    """Best single-line inlier mask via mini-RANSAC + one refit round."""
    n = len(pts)
    best_mask = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        u = d / nd
        v = pts - pts[i]
        perp = np.linalg.norm(v - np.outer(v @ u, u), axis=1)
        m = perp <= r_in
        if best_mask is None or m.sum() > best_mask.sum():
            best_mask = m
    if best_mask is None or best_mask.sum() < 3:
        return None
    f = _line_fit(pts[best_mask])
    if f is None:
        return None
    v = pts - f[0]
    perp = np.linalg.norm(v - np.outer(v @ f[1], f[1]), axis=1)
    m = perp <= r_in
    if m.sum() < 3:
        return None
    f = _line_fit(pts[m])
    if f is None:
        return None
    v = pts - f[0]
    perp = np.linalg.norm(v - np.outer(v @ f[1], f[1]), axis=1)
    return perp <= r_in, f


def peel_flagged(labels_global, split_index, label_info, xyz, e, tpc, config):
    g = lambda k, d: getattr(config, k, d) if config is not None else d
    iters = int(g("mp_iters", 40))
    r_in = float(g("mp_inlier_cm", 1.2))
    min_kept = int(g("mp_min_kept", 15))
    min_kept_len = float(g("mp_min_kept_len_cm", 8.0))
    rem_eps = float(g("mp_rem_eps", 4.0))
    rem_min = int(g("mp_rem_min_samples", 3))
    c_hits = int(g("mp_comp_min_hits", 15))
    c_len = float(g("mp_comp_min_len_cm", 8.0))
    c_lin = float(g("mp_comp_min_lin", 0.85))
    c_ov = float(g("mp_comp_overlap", 0.6))

    spec = _util.spec_from_file_location(
        "mp_sp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "segment_split.py"))
    sp = _util.module_from_spec(spec)
    spec.loader.exec_module(sp)

    labels = np.asarray(labels_global).copy()
    xyz = np.asarray(xyz, np.float64)
    e = np.asarray(e, np.float64)
    tpc = np.asarray(tpc)
    si = int(split_index)
    rng = np.random.default_rng(0)

    flags = sp.two_track_flag(labels, xyz, e, tpc, SimpleNamespace())
    if label_info is not None:
        flags = {lab: tps for lab, tps in flags.items()
                 if lab < si and str(label_info.get(lab, {}).get(
                     "type", "track")).lower() == "track"}
    else:
        flags = {lab: tps for lab, tps in flags.items() if lab < si}

    def comp_geom(pts):
        c = pts.mean(0)
        q = pts - c
        try:
            _, sv, vt = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        t = q @ vt[0]
        lin = float(sv[0] ** 2 / max(float((sv ** 2).sum()), 1e-12))
        return float(t.max() - t.min()), lin

    nxt = int(labels.max()) + 1
    peels = []
    for lab, tpcs in sorted(flags.items()):
        for t_ in sorted(np.atleast_1d(list(tpcs))):
            idx = np.flatnonzero((labels == lab) & (tpc == t_))
            if len(idx) < min_kept + c_hits:
                continue
            res = _robust_line_inliers(xyz[idx], iters, r_in, rng)
            if res is None:
                continue
            core, f = res
            kept = idx[core]
            rem = idx[~core]
            if len(kept) < min_kept or len(rem) < c_hits:
                continue
            t_kept = (xyz[kept] - f[0]) @ f[1]
            t_lo, t_hi = float(t_kept.min()), float(t_kept.max())
            if t_hi - t_lo < min_kept_len:
                continue
            sub = DBSCAN(eps=rem_eps, min_samples=rem_min).fit_predict(xyz[rem])
            peeled_comps, skipped_comps = [], []
            for uu in np.unique(sub[sub >= 0]):
                ci = rem[sub == uu]
                L, lin = comp_geom(xyz[ci])
                t_c = (xyz[ci] - f[0]) @ f[1]
                ov = float(np.mean((t_c >= t_lo) & (t_c <= t_hi)))
                feat = {"n": int(len(ci)), "len_cm": L, "lin": lin,
                        "e": float(e[ci].sum()), "ov": ov}
                if len(ci) >= c_hits and L >= c_len and lin >= c_lin and ov >= c_ov:
                    labels[ci] = nxt
                    feat["new_label"] = nxt
                    peeled_comps.append(feat)
                    nxt += 1
                else:
                    # diffuse halo (deltas, tails) stays on the track
                    skipped_comps.append(feat)
            if peeled_comps:
                peels.append({"label": int(lab), "tpc": int(t_),
                              "n_kept": int(len(kept)),
                              "peeled": peeled_comps,
                              "skipped": skipped_comps,
                              "new_labels": [p["new_label"]
                                             for p in peeled_comps]})
    return labels, {"peels": peels,
                    "n_new_labels": sum(len(p["new_labels"]) for p in peels)}
