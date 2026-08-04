"""Segment splitter v0.1 - un-fuse per-TPC RANSAC segments that captured TWO
parallel/overlapping tracks in one line fit (the dominant remaining purity
loss: ~22.8 GeV of foreign energy trapped inside single segments over the
126-event release set).

Method (pure numpy, O(n log n) per segment, few ms/event):
  1. For each track segment with >= ss_min_hits hits: project hits onto the
     plane transverse to the fitted axis; take the principal transverse axis;
     1D Otsu split of that coordinate.
  2. Propose a split only if the two modes are >= ss_min_sep_cm apart
     (measured: pure segments' blind Otsu separation never exceeds ~1.2 cm,
     fused two-track segments sit at 4.9-7.4 cm - a clean dead zone).
  3. Refine: ss_refit_iters rounds of {fit a line to each side, reassign every
     hit to the nearer line} - recovers hits the 1D cut misassigns where the
     tracks wobble.
  4. Commit only if, after refinement, each side has >= ss_min_side_hits hits,
     >= ss_min_side_frac of the parent, longitudinal extent >= ss_min_side_elong
     of the parent's (kills delta-ray blobs: elongated cores only), and the
     lines remain >= ss_min_sep_cm apart (median cross-line distance).
  5. Recurse once on each child (ss_max_depth).

Wire-in (recluster.py, params key "segment_split_enable": true):
    sp = _file_load("clu_sp", HERE/"segment_split.py")
    toolbox._build_tpc_segments_toolbox = sp.wrap_builder(
        toolbox._build_tpc_segments_toolbox, SimpleNamespace(**params))
Config keys (defaults): ss_min_sep_cm 2.0, ss_min_hits 60, ss_min_side_hits 20,
ss_min_side_frac 0.12, ss_min_side_elong 0.4, ss_refit_iters 2, ss_max_depth 2.
"""
from __future__ import annotations

import numpy as np

__all__ = ["split_segments", "wrap_builder"]


def _line(pts):
    c = pts.mean(0)
    q = pts - c
    try:
        _, sv, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    u = vt[0]
    t = q @ u
    lin = float(sv[0] ** 2 / max(float((sv ** 2).sum()), 1e-12))
    return c, u, t, lin


def _dist_to_line(pts, c, u):
    q = pts - c
    return np.linalg.norm(q - np.outer(q @ u, u), axis=1)


def _otsu(s):
    ss = np.sort(s)
    n = len(ss)
    csum = np.cumsum(ss)
    idx = np.arange(1, n)
    m1 = csum[:-1] / idx
    m2 = (csum[-1] - csum[:-1]) / (n - idx)
    w = idx * (n - idx) * (m1 - m2) ** 2
    k = int(np.argmax(w))
    thr = 0.5 * (ss[k] + ss[k + 1])
    lo, hi = ss[: k + 1], ss[k + 1:]
    return thr, float(hi.mean() - lo.mean())


def _try_split(pts, cfg):
    """Return boolean mask (side A) or None."""
    fr = _line(pts)
    if fr is None:
        return None
    c, u, t, _ = fr
    resid = (pts - c) - np.outer(t, u)
    try:
        _, _, vt2 = np.linalg.svd(resid, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    s = resid @ vt2[0]
    thr, sep = _otsu(s)
    if sep < cfg["min_sep"]:
        return None
    side = s <= thr
    # refine: per-side line fits + nearest-line reassignment
    for _ in range(cfg["refit_iters"]):
        if side.sum() < 3 or (~side).sum() < 3:
            return None
        fa, fb = _line(pts[side]), _line(pts[~side])
        if fa is None or fb is None:
            return None
        da = _dist_to_line(pts, fa[0], fa[1])
        db = _dist_to_line(pts, fb[0], fb[1])
        side = da <= db
    nA, nB = int(side.sum()), int((~side).sum())
    n = len(pts)
    if min(nA, nB) < cfg["min_side_hits"] or min(nA, nB) < cfg["min_side_frac"] * n:
        return None
    # elongation guard (delta-ray blobs are compact along the parent axis)
    span = t.max() - t.min()
    if span <= 1e-6:
        return None
    for m in (side, ~side):
        tm = t[m]
        if (tm.max() - tm.min()) < cfg["min_side_elong"] * span:
            return None
    # final acceptance: the fused topology is TWO PARALLEL, individually-TIGHT
    # lines well separated - curved single tracks and wide blobs fail these
    fa, fb = _line(pts[side]), _line(pts[~side])
    if fa[3] < cfg["min_child_lin"] or fb[3] < cfg["min_child_lin"]:
        return None
    if abs(float(fa[1] @ fb[1])) < cfg["min_parallel"]:
        return None
    own_a = float(np.median(_dist_to_line(pts[side], fa[0], fa[1])))
    own_b = float(np.median(_dist_to_line(pts[~side], fb[0], fb[1])))
    if max(own_a, own_b) > cfg["max_child_rms"]:
        return None
    sep_ab = float(np.median(_dist_to_line(pts[side], fb[0], fb[1])))
    sep_ba = float(np.median(_dist_to_line(pts[~side], fa[0], fa[1])))
    # adaptive: the gap must exceed BOTH the absolute floor and the childrens'
    # own transverse spread (two tracks are distinct only if separated by more
    # than their widths)
    need = max(cfg["min_sep"], cfg["sep_scale"] * (own_a + own_b))
    if min(sep_ab, sep_ba) < need:
        return None
    return side


def _axial_components(xyz, hits, gap_cm, min_hits):
    """Decompose a hit set into axially CONTIGUOUS runs along its own principal
    axis (break at longitudinal gaps > gap_cm). A transverse split side can be
    two disconnected islands; passing such a child on inflates its endpoints and
    fools the stitcher into long-distance joins (f4 ev8 label 32 bug)."""
    hits = np.asarray(hits, int)
    if len(hits) < max(min_hits, 3):
        return []
    fr = _line(xyz[hits])
    if fr is None:
        return [hits] if len(hits) >= min_hits else []
    _, _, t, _ = fr
    order = np.argsort(t)
    breaks = np.flatnonzero(np.diff(t[order]) > gap_cm)
    return [hits[part] for part in np.split(order, breaks + 1)
            if len(part) >= min_hits]


def _seg_from(parent, hits, xyz):
    pts = xyz[hits]
    fr = _line(pts)
    c, u, t, lin = fr
    ep = np.stack([c + t.min() * u, c + t.max() * u])
    out = dict(parent)
    out["hits"] = np.asarray(hits, dtype=int)
    out["endpoints"] = ep
    out["direction"] = u
    out["linearity"] = lin
    return out


def split_segments(segments, xyz, config=None):
    g = lambda k, d: float(getattr(config, k, d)) if config is not None else d
    cfg = dict(min_sep=g("ss_min_sep_cm", 2.0),
               min_side_hits=int(g("ss_min_side_hits", 20)),
               min_side_frac=g("ss_min_side_frac", 0.12),
               min_side_elong=g("ss_min_side_elong", 0.4),
               refit_iters=int(g("ss_refit_iters", 2)),
               min_child_lin=g("ss_min_child_lin", 0.92),
               min_parallel=g("ss_min_parallel", 0.985),
               max_child_rms=g("ss_max_child_rms", 1e9),
               sep_scale=g("ss_sep_scale", 1.5))
    min_hits = int(g("ss_min_hits", 60))
    max_depth = int(g("ss_max_depth", 2))
    axial_gap = g("ss_axial_gap_cm", 8.0)
    out = []
    n_split = 0

    def process(seg, depth):
        nonlocal n_split
        hits = np.asarray(seg["hits"], int)
        # no source gate: fused pairs often arrive as 'rescue_cluster' segments
        # (overlapping tracks fail clean RANSAC and get promoted from leftovers)
        if depth >= max_depth or len(hits) < min_hits:
            out.append(seg)
            return
        side = _try_split(xyz[hits], cfg)
        if side is None:
            out.append(seg)
            return
        n_split += 1
        # children must be axially CONTIGUOUS: decompose each transverse side
        # into longitudinal runs; sub-threshold fragments fall back to the
        # leftover pool (handled by leftover DBSCAN downstream)
        for side_hits in (hits[side], hits[~side]):
            for comp in _axial_components(xyz, side_hits, axial_gap,
                                          cfg["min_side_hits"]):
                process(_seg_from(seg, comp, xyz), depth + 1)

    for seg in segments:
        process(seg, 0)
    return out, n_split


def wrap_builder(orig_builder, config):
    """Wrap _build_tpc_segments_toolbox: split fused segments right after building."""
    def builder(x, y, z, *args, **kwargs):
        segments, dbg = orig_builder(x, y, z, *args, **kwargs)
        xyz = np.column_stack([np.asarray(x, np.float64), np.asarray(y, np.float64),
                               np.asarray(z, np.float64)])
        segments, n_split = split_segments(segments, xyz, config)
        if isinstance(dbg, dict):
            dbg = dict(dbg)
            dbg["n_segment_splits"] = int(n_split)
        return segments, dbg
    return builder


def two_track_flag(labels, xyz, energy, tpc, config=None):
    """Two-track-suspect flag v0.1 (dE/dx + width; Billy's design 2026-07-23).

    For each label, per-TPC piece with >= 40 hits and >= 15 cm axial span:
    linear charge density = E / length, transverse width = median distance to
    the piece's own PCA line. Suspect iff width >= tt_wmed_cm (1.2 cm) AND
    density >= tt_dens_mevcm (2.0 MeV/cm). Measured on 126 events: catches 77%
    of truth-fused segments (73% of those the geometric splitter cannot split)
    at a 4.7% false-flag rate on pure segments. Pure metadata - no clustering
    change; the downstream matcher should treat flagged charge cautiously.
    Returns {label: [suspect tpc ids]}.
    """
    g = lambda k, d: float(getattr(config, k, d)) if config is not None else d
    wmin = g("tt_wmed_cm", 1.2)
    dmin = g("tt_dens_mevcm", 2.0)
    out = {}
    labels = np.asarray(labels)
    tpc = np.asarray(tpc)
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        if m.sum() < 40:
            continue
        for t in np.unique(tpc[m]):
            mt = np.flatnonzero(m & (tpc == t))
            if len(mt) < 40:
                continue
            fr = _line(xyz[mt])
            if fr is None:
                continue
            c, u, tt_, _lin = fr
            length = float(tt_.max() - tt_.min())
            if length < 15.0:
                continue
            dens = float(energy[mt].sum()) / length
            resid = (xyz[mt] - c) - np.outer(tt_, u)
            wmed = float(np.median(np.linalg.norm(resid, axis=1)))
            if wmed >= wmin and dens >= dmin:
                out.setdefault(int(lab), []).append(int(t))
    return out
