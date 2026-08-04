"""Endpoint rejoin pass v0.1 (Billy's design) - runs AFTER stitching, before
vertex pinpointing. For every backbone track endpoint it looks for:

  1. a SEGMENT-LIKE DBSCAN remnant nearby (Billy's calibrated variance cut:
     >=15 hits and perpendicular RMS <= er_rms_a + er_rms_b * L) whose end-line
     continues the track (anti-aligned outward directions, small transverse
     offset) -> connect;
  2. another TRACK collinear with it across a gap -> connect, requiring
     BRIDGE-HIT evidence when the gap lies inside active volume (a gap crossing
     a module dead-region legitimately has no hits, so there the directional
     discipline alone decides).

All connections use the stitcher-grade gates that won the directional
campaign: dot(u_out_a, u_out_b) <= er_dot_max, transverse component of the gap
vector <= er_trans_tol. Union-find merging, keep-smallest label (tracks are
numbered below blobs, so a healed track keeps its track label and type).

API: rejoin_endpoints(labels, split_index, label_info, xyz, tpc, config)
     -> (labels, stats). label_info mutated in place (merged_into bookkeeping).
Config keys (defaults): er_min_track_len_cm 10, er_local_cm 10,
er_remnant_min_hits 15, er_rms_a 0.20, er_rms_b 0.1653, er_search_cm 40,
er_gap_max_cm 80, er_dot_max -0.97, er_trans_tol_cm 5.0,
er_bridge_radius_cm 2.5, er_bridge_active_frac 0.35, er_bridge_occupancy 0.25.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["rejoin_endpoints"]


def _line_frame(pts):
    c = pts.mean(0)
    q = pts - c
    try:
        _, sv, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    u = vt[0]
    t = q @ u
    resid = q - np.outer(t, u)
    rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    return c, u, t, rms


def _end_anchor_dir(pts, side, local_cm):
    """(anchor, outward unit dir) for one end (side=+1 high-t, -1 low-t)."""
    fr = _line_frame(pts)
    if fr is None:
        return None
    c, u, t, _ = fr
    order = np.argsort(t)
    idx = order[-3:] if side > 0 else order[:3]
    anchor = pts[idx].mean(0)
    d = np.linalg.norm(pts - anchor, axis=1)
    sel = pts[d <= local_cm]
    if len(sel) < 3:
        sel = pts[np.argsort(d)[:6]]
    fr2 = _line_frame(sel)
    if fr2 is None:
        return None
    c2, u2 = fr2[0], fr2[1]
    if u2 @ (anchor - c2) < 0:
        u2 = -u2
    return anchor, u2


def _trans_lon(pa, ua, pb, ub):
    d = pb - pa
    u = ua - ub
    n = np.linalg.norm(u)
    if n < 1e-12:
        return float(np.linalg.norm(d)), 0.0
    u = u / n
    lon = float(abs(d @ u))
    return float(np.linalg.norm(d - (d @ u) * u)), lon


def rejoin_endpoints(labels_global, split_index, label_info, xyz, tpc, config):
    g = lambda k, d: getattr(config, k, d) if config is not None else d
    min_len = float(g("er_min_track_len_cm", 10.0))
    local_cm = float(g("er_local_cm", 10.0))
    rem_min_hits = int(g("er_remnant_min_hits", 15))
    rms_a = float(g("er_rms_a", 0.20))
    rms_b = float(g("er_rms_b", 0.1653))
    search = float(g("er_search_cm", 40.0))
    gap_max = float(g("er_gap_max_cm", 80.0))
    dot_max = float(g("er_dot_max", -0.97))
    trans_tol = float(g("er_trans_tol_cm", 5.0))
    br_rad = float(g("er_bridge_radius_cm", 2.5))
    br_af = float(g("er_bridge_active_frac", 0.35))
    br_occ = float(g("er_bridge_occupancy", 0.25))

    labels = np.asarray(labels_global).copy()
    xyz = np.asarray(xyz, np.float64)
    si = int(split_index)

    # charge boxes for active-volume test
    boxes = []
    for t_ in np.unique(np.asarray(tpc)):
        h = xyz[np.asarray(tpc) == t_]
        boxes.append((h.min(0), h.max(0)))

    def in_active(p):
        return any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in boxes)

    # track ends
    track_labels = [int(l) for l in range(si)
                    if str(label_info.get(int(l), {}).get("type", "track")).lower() == "track"
                    and np.any(labels == l)]
    ends = []          # (label, anchor, u_out, n_hits, length, hits_per_cm)
    pts_of = {}
    for l in track_labels:
        pts = xyz[labels == l]
        fr = _line_frame(pts)
        if fr is None:
            continue
        L = float(fr[2].max() - fr[2].min())
        if L < min_len:
            continue
        pts_of[l] = pts
        dens = len(pts) / max(L, 1e-6)
        for side in (-1, +1):
            ad = _end_anchor_dir(pts, side, local_cm)
            if ad is not None:
                ends.append((l, ad[0], ad[1], len(pts), L, dens))

    # segment-like remnants (Billy's fitted variance cut + sanity caps:
    # the fitted rms(L) cut is unbounded at large L, so a module-wide shower
    # region could qualify - cap absolute rms and remnant size)
    rms_abs = float(g("er_rms_abs_max_cm", 3.0))
    remnants = []      # (label, anchor_lo, dir_lo, anchor_hi, dir_hi, L, n_hits)
    u_lab, cnt = np.unique(labels[labels >= 0], return_counts=True)
    for l, c in zip(u_lab, cnt):
        if l < si or c < rem_min_hits:
            continue
        pts = xyz[labels == l]
        fr = _line_frame(pts)
        if fr is None:
            continue
        L = float(fr[2].max() - fr[2].min())
        if fr[3] > min(rms_a + rms_b * L, rms_abs):
            continue
        lo = _end_anchor_dir(pts, -1, local_cm)
        hi = _end_anchor_dir(pts, +1, local_cm)
        if lo is None or hi is None:
            continue
        remnants.append((int(l), lo[0], lo[1], hi[0], hi[1], L, int(c)))

    pairs: list[dict[str, Any]] = []

    br_min_gap = float(g("er_bridge_min_gap_cm", 4.0))

    def bridge_ok(pa, pb):
        """Bridge-hit evidence for the corridor pa->pb (waived in dead regions
        and for tiny gaps, where the corridor window is degenerate and the
        directional gates alone are decisive)."""
        d = pb - pa
        L = float(np.linalg.norm(d))
        if L <= br_min_gap:
            return True, 1.0, 0
        u = d / L
        # active fraction of the corridor
        s = np.linspace(0.05, 0.95, 19)
        act = float(np.mean([in_active(pa + si_ * L * u) for si_ in s]))
        if act <= br_af:
            return True, act, -1              # mostly dead region: waived
        # hits inside the corridor (any label or noise)
        lo = np.minimum(pa, pb) - br_rad
        hi = np.maximum(pa, pb) + br_rad
        cand = np.flatnonzero(np.all((xyz >= lo) & (xyz <= hi), axis=1))
        if len(cand):
            v = xyz[cand] - pa
            proj = v @ u
            perp = np.linalg.norm(v - np.outer(proj, u), axis=1)
            nb = int(np.sum((perp <= br_rad) & (proj > 1.0) & (proj < L - 1.0)))
        else:
            nb = 0
        need = br_occ * act * L * 3.0          # ~3 hits/cm nominal MIP occupancy
        return nb >= need, act, nb

    max_ratio = float(g("er_remnant_max_hits_ratio", 0.8))

    # 1) track end <-> remnant end  (endpoint keys carried for exclusivity)
    if ends and remnants:
        r_anchors = np.array([r[1] for r in remnants] + [r[3] for r in remnants])
        r_meta = [(k, 0) for k in range(len(remnants))] + [(k, 1) for k in range(len(remnants))]
        tree = cKDTree(r_anchors)
        for ei, (l, pa, ua, n, L, dens) in enumerate(ends):
            for j in tree.query_ball_point(pa, search):
                k, end_i = r_meta[j]
                rl = remnants[k][0]
                if rl == l:
                    continue
                if remnants[k][6] > max_ratio * n:     # remnant must BE a remnant
                    continue
                pb = remnants[k][1] if end_i == 0 else remnants[k][3]
                ub = remnants[k][2] if end_i == 0 else remnants[k][4]
                dot = float(ua @ ub)
                if dot > dot_max:
                    continue
                trans, lon = _trans_lon(pa, ua, pb, ub)
                if trans > trans_tol:
                    continue
                okb, act, nb = bridge_ok(pa, pb)
                if not okb:
                    continue
                pairs.append(dict(kind="remnant", labels=[int(l), int(rl)],
                                  gap_cm=float(np.linalg.norm(pb - pa)),
                                  dot=dot, trans_cm=trans,
                                  active_frac=act, n_bridge=nb,
                                  end_a=("t", ei), end_b=("r", k, end_i)))

    # 2) track end <-> track end (collinear across a gap)
    if ends:
        e_anchors = np.array([e_[1] for e_ in ends])
        tree = cKDTree(e_anchors)
        done = set()
        for i, (l, pa, ua, n, L, dens) in enumerate(ends):
            for j in tree.query_ball_point(pa, gap_max):
                if j <= i:
                    continue
                l2, pb, ub = ends[j][0], ends[j][1], ends[j][2]
                if l2 == l or (min(l, l2), max(l, l2)) in done:
                    continue
                dot = float(ua @ ub)
                if dot > dot_max:
                    continue
                trans, lon = _trans_lon(pa, ua, pb, ub)
                if trans > trans_tol:
                    continue
                okb, act, nb = bridge_ok(pa, pb)
                if not okb:
                    continue
                done.add((min(l, l2), max(l, l2)))
                pairs.append(dict(kind="collinear", labels=[int(l), int(l2)],
                                  gap_cm=float(np.linalg.norm(pb - pa)),
                                  dot=dot, trans_cm=trans,
                                  active_frac=act, n_bridge=nb,
                                  end_a=("t", i), end_b=("t", j)))

    # ENDPOINT-EXCLUSIVE best-first acceptance (Billy's two-endpoint principle):
    # each endpoint is consumed by its first (best) join; a consumed junction
    # is interior and never joins again. Chains remain possible through the
    # partner's far end.
    for p in pairs:
        p["score"] = (p["trans_cm"] / max(trans_tol, 1e-6)
                      + (1.0 + p["dot"]) / max(1.0 + dot_max, 1e-6)
                      + p["gap_cm"] / max(gap_max, 1e-6))
    used_ends = set()
    accepted = []
    for p in sorted(pairs, key=lambda q: q["score"]):
        if p["end_a"] in used_ends or p["end_b"] in used_ends:
            continue
        used_ends.add(p["end_a"])
        used_ends.add(p["end_b"])
        accepted.append(p)
    pairs = accepted

    # merge (union-find, keep smallest)
    all_labs = set()
    for p in pairs:
        all_labs.update(p["labels"])
    parent = {l: l for l in all_labs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in pairs:
        a, b = p["labels"]
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for l in all_labs:
        groups.setdefault(find(l), []).append(l)
    groups = sorted(sorted(gr) for gr in groups.values() if len(gr) >= 2)
    for gr in groups:
        keep = int(gr[0])
        for lab in gr[1:]:
            labels[labels == lab] = keep
            if int(lab) in label_info:
                label_info[int(lab)]["n_hits"] = 0
                label_info[int(lab)]["merged_into"] = keep
        if keep in label_info:
            label_info[keep]["rejoined"] = True
            label_info[keep]["rejoined_from"] = [int(v) for v in gr[1:]]
    return labels, {"pairs": pairs, "groups": groups}
