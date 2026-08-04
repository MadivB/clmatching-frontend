"""Flag-gated two-track splitter S2 v0.2 (PROTOTYPE - Billy's X-crossing
design + CROSSER-CONTINUITY gate). NOT yet part of the adopted chain.

v0.1 lesson (126-event audit): pure two-line geometry splits at only 8-18%
precision - two-line structure is common WITHIN one interaction (delta rays,
vertex prongs, kinks). The reco-accessible discriminator: a real crosser
CONTINUES beyond the flagged window as a different track label; a delta ray
does not.

For every (track label, TPC) window marked by the dE/dx-doubling two-track
flag, fit a TWO-LINE model via 2-line RANSAC, then require:

  - each child >= tts_min_hits hits and >= tts_min_len_cm axial length,
  - rms(2-line) <= tts_rms_ratio x rms(1-line), rms(1-line) <= tts_rms1_max
    (two fused MIPs are moderately thin), rms(2-line) <= tts_rms2_max
    (children must be MIP-thin),
  - CROSSER CONTINUITY: exactly one child's line, extended beyond the window
    by up to tts_cont_reach_cm, collects >= tts_cont_min_hits hits of a
    single OTHER track label within tts_cont_radius_cm of the line, locally
    collinear (|dot| >= tts_cont_dot).

That child is DONATED to the continuing label (removes the foreign energy
from the fused track AND heals the crosser - zero new labels). The other
child keeps the parent label. No continuation, or two continuations -> no
split (purity-first).

API: split_two_track(labels, split_index, label_info, xyz, e, tpc, config)
     -> (labels, stats)
Config keys (defaults): tts_min_hits 15, tts_min_len_cm 8.0,
tts_rms_ratio 0.6, tts_rms1_max 3.0, tts_rms2_max 0.8, tts_ransac_iters 60,
tts_inlier_cm 1.2, tts_cont_radius_cm 1.5, tts_cont_reach_cm 40.0,
tts_cont_min_hits 10, tts_cont_dot 0.95.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

__all__ = ["split_two_track"]


def _line_fit(pts):
    c = pts.mean(0)
    q = pts - c
    try:
        _, sv, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    u = vt[0]
    t = q @ u
    resid = np.linalg.norm(q - np.outer(t, u), axis=1)
    return c, u, t, resid


def _two_line_ransac(pts, iters, inlier_cm, rng):
    """Best 2-line partition of pts. Returns (assign, rms2, lines) or None."""
    n = len(pts)
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        u1 = d / nd
        r1 = np.linalg.norm((pts - pts[i]) - np.outer((pts - pts[i]) @ u1, u1), axis=1)
        in1 = r1 <= inlier_cm
        if in1.sum() < 3 or (~in1).sum() < 3:
            continue
        f1 = _line_fit(pts[in1])
        f2 = _line_fit(pts[~in1])
        if f1 is None or f2 is None:
            continue
        # refine assignment: nearest line
        d1 = np.linalg.norm((pts - f1[0]) - np.outer((pts - f1[0]) @ f1[1], f1[1]), axis=1)
        d2 = np.linalg.norm((pts - f2[0]) - np.outer((pts - f2[0]) @ f2[1], f2[1]), axis=1)
        assign = d2 < d1
        if assign.sum() < 3 or (~assign).sum() < 3:
            continue
        g1 = _line_fit(pts[~assign])
        g2 = _line_fit(pts[assign])
        if g1 is None or g2 is None:
            continue
        rms2 = float(np.sqrt((np.concatenate([g1[3], g2[3]]) ** 2).mean()))
        if best is None or rms2 < best[1]:
            best = (assign, rms2, (g1, g2))
    return best


def split_two_track(labels_global, split_index, label_info, xyz, e, tpc, config):
    g = lambda k, d: getattr(config, k, d) if config is not None else d
    min_hits = int(g("tts_min_hits", 15))
    min_len = float(g("tts_min_len_cm", 8.0))
    rms_ratio = float(g("tts_rms_ratio", 0.6))
    rms1_max = float(g("tts_rms1_max", 3.0))
    rms2_max = float(g("tts_rms2_max", 0.8))
    iters = int(g("tts_ransac_iters", 60))
    inlier = float(g("tts_inlier_cm", 1.2))
    c_rad = float(g("tts_cont_radius_cm", 1.5))
    c_reach = float(g("tts_cont_reach_cm", 40.0))
    c_min = int(g("tts_cont_min_hits", 10))
    c_dot = float(g("tts_cont_dot", 0.95))

    from importlib import util as _u
    import os as _os
    spec = _u.spec_from_file_location(
        "tts_sp", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                "segment_split.py"))
    sp = _u.module_from_spec(spec)
    spec.loader.exec_module(sp)

    labels = np.asarray(labels_global).copy()
    xyz = np.asarray(xyz, np.float64)
    tpc = np.asarray(tpc)
    si = int(split_index)
    rng = np.random.default_rng(0)          # deterministic

    flags = sp.two_track_flag(labels, xyz, np.asarray(e, np.float64), tpc,
                              SimpleNamespace())
    if label_info is not None:
        flags = {lab: tps for lab, tps in flags.items()
                 if lab < si and str(label_info.get(lab, {}).get(
                     "type", "track")).lower() == "track"}
    else:
        flags = {lab: tps for lab, tps in flags.items() if lab < si}

    def continuation(lab, child_idx, line):
        """Strongest other-track-label continuation of `line` beyond the
        child's own axial span. Returns (label, n_hits, dot) or None."""
        c, u = line[0], line[1]
        t_child = (xyz[child_idx] - c) @ u
        t_lo, t_hi = float(t_child.min()), float(t_child.max())
        v = xyz - c
        t_all = v @ u
        perp = np.linalg.norm(v - np.outer(t_all, u), axis=1)
        m = ((labels >= 0) & (labels != lab) & (labels < si)
             & (perp <= c_rad)
             & (((t_all > t_hi + 1.0) & (t_all <= t_hi + c_reach))
                | ((t_all < t_lo - 1.0) & (t_all >= t_lo - c_reach))))
        if m.sum() < c_min:
            return None
        cand, cnt = np.unique(labels[m], return_counts=True)
        kbest = int(cand[cnt.argmax()])
        if cnt.max() < c_min:
            return None
        ch = np.flatnonzero(m & (labels == kbest))
        fc = _line_fit(xyz[ch])
        if fc is None:
            return None
        dot = abs(float(fc[1] @ u))
        if dot < c_dot:
            return None
        return kbest, int(cnt.max()), dot

    splits = []
    for lab, tpcs in sorted(flags.items()):
        for t_ in sorted(np.atleast_1d(list(tpcs))):
            idx = np.flatnonzero((labels == lab) & (tpc == t_))
            if len(idx) < 2 * min_hits:
                continue
            pts = xyz[idx]
            f0 = _line_fit(pts)
            if f0 is None:
                continue
            rms1 = float(np.sqrt((f0[3] ** 2).mean()))
            if rms1 > rms1_max:
                continue                       # shower-y region, not two MIPs
            best = _two_line_ransac(pts, iters, inlier, rng)
            if best is None:
                continue
            assign, rms2, (g1, g2) = best
            if rms2 > rms_ratio * rms1 or rms2 > rms2_max:
                continue
            n2 = int(assign.sum())
            n1 = len(pts) - n2
            if min(n1, n2) < min_hits:
                continue
            L1 = float(g1[2].max() - g1[2].min())
            L2 = float(g2[2].max() - g2[2].min())
            if min(L1, L2) < min_len:
                continue
            # crosser-continuity gate: exactly one child continues as a
            # DIFFERENT track label beyond the window
            cont1 = continuation(lab, idx[~assign], g1)
            cont2 = continuation(lab, idx[assign], g2)
            if (cont1 is None) == (cont2 is None):
                continue                       # none or both: ambiguous, skip
            if cont2 is not None:
                child, donee = idx[assign], cont2
            else:
                child, donee = idx[~assign], cont1
            labels[child] = donee[0]
            splits.append({"label": int(lab), "tpc": int(t_),
                           "donee": int(donee[0]), "n_child": int(len(child)),
                           "cont_hits": donee[1], "cont_dot": donee[2],
                           "rms1": rms1, "rms2": rms2})
    return labels, {"splits": splits}
