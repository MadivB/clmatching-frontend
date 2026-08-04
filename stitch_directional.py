"""Directional cross-TPC stitcher v0.1 (Billy's design, 2026-07-22).

Replaces the production angle gate (sign-agnostic |cos|) with DIRECTIONALITY:
each segment end gets a local direction from the hits within sd_end_window_cm
in z of that endpoint (robust to scattering along the track), oriented OUTWARD
through the endpoint. A join is only allowed when the two outward vectors
ANTI-ALIGN: dot(u_i, u_j) <= sd_dot_max (default -0.97). True continuations
point at each other (dot ~ -1); the overlap topology that fuses collinear
through-goers pairs same-side ends with dot ~ +1 and is now rejected.

Validated on 126 events (cached RANSAC segments, vs production matcher):
  edges              : baseline 21010 correct / 59 wrong
  dot-0.97 W8 ep40   : 21211 correct / 45 wrong, stitched foreign -23%
  dot-0.97 W8 ep60   : 21587 correct / 58 wrong (heals longer-gap breaks)
Full chain (+ vertex_pinpoint v0.3), vs baseline chain:
  ep40: purity +0.0004 (files 2-10), completeness  +0.0001   (purity option)
  ep60: purity -0.0001 (files 2-10), completeness  +0.0041   (balanced option)

Config keys (fallback defaults): sd_dot_max -0.97, sd_end_window_cm 8.0,
sd_dist_tol 5.0, sd_ep_tol 40.0, sd_allow_same_tpc False, sd_same_tpc_min_gap 2.0.
Wire-in: recluster.py patches the loaded toolbox when params has
"stitch_directional_enable": true ->
    toolbox._match_segments_across_tpcs_toolbox = make_matcher(cfg, toolbox._fit_line_metrics)
"""
from __future__ import annotations

import numpy as np

__all__ = ["make_matcher"]
_WE, _WA, _WQ = 0.45, 0.35, 0.15


def _seg_seg_dist(p1, q1, p2, q2):
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c = d1 @ r
    b = d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0.0:
        t = 0.0
        s = np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _extend_to_box(p1, p2, bounds):
    """Extend segment [p1,p2] along its line to the bounding box (both ways)."""
    d = p2 - p1
    n = np.linalg.norm(d)
    if n < 1e-9:
        return p1.copy(), p2.copy()
    d = d / n
    lo = np.array([bounds[0], bounds[2], bounds[4]])
    hi = np.array([bounds[1], bounds[3], bounds[5]])
    tmin, tmax = -1e18, 1e18
    for k in range(3):
        if abs(d[k]) < 1e-12:
            continue
        t1, t2 = (lo[k] - p1[k]) / d[k], (hi[k] - p1[k]) / d[k]
        t1, t2 = min(t1, t2), max(t1, t2)
        tmin, tmax = max(tmin, t1), min(tmax, t2)
    if tmin > tmax:
        return p1.copy(), p2.copy()
    return p1 + tmin * d, p1 + tmax * d


def _best_endpoint_pair(ea, eb):
    best = (1e18, 0, 0)
    for i in range(2):
        for j in range(2):
            d = float(np.linalg.norm(ea[i] - eb[j]))
            if d < best[0]:
                best = (d, i, j)
    return best


def _local_out_dir(pts, p_end, W, mode="z"):
    if mode == "euclid":
        sel = pts[np.linalg.norm(pts - p_end, axis=1) <= W]
    else:
        sel = pts[np.abs(pts[:, 2] - p_end[2]) <= W]
    if len(sel) < 3:
        d = np.linalg.norm(pts - p_end, axis=1)
        sel = pts[np.argsort(d)[:6]]
    c = sel.mean(0)
    q = sel - c
    try:
        _, _, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    u = vt[0]
    n = np.linalg.norm(u)
    if n < 1e-12:
        return None
    u = u / n
    return -u if u @ (p_end - c) < 0 else u


def make_matcher(config, fit_line_metrics):
    """Return a drop-in replacement for _match_segments_across_tpcs_toolbox."""
    g = lambda k, d: getattr(config, k, d)
    dot_max = float(g("sd_dot_max", -0.97))
    W = float(g("sd_end_window_cm", 8.0))
    wmode = str(g("sd_end_window_mode", "z"))   # 'z' slab | 'euclid' ball
    dist_tol = float(g("sd_dist_tol", 5.0))
    ep_tol = float(g("sd_ep_tol", 40.0))
    allow_same = bool(g("sd_allow_same_tpc", False))
    same_min_gap = float(g("sd_same_tpc_min_gap", 2.0))
    trans_tol = g("sd_trans_tol", None)      # cm; None = no transverse gate
    trans_tol = None if trans_tol is None else float(trans_tol)

    def matcher(segments, x, y, z, **_kw):
        xyz = np.column_stack([np.asarray(x, np.float64), np.asarray(y, np.float64),
                               np.asarray(z, np.float64)])
        if len(segments) == 0:
            return [], {"candidate_edges": [], "accepted_edges": [],
                        "n_candidate_edges": 0, "n_accepted_edges": 0}
        bounds = (float(xyz[:, 0].min()), float(xyz[:, 0].max()),
                  float(xyz[:, 1].min()), float(xyz[:, 1].max()),
                  float(xyz[:, 2].min()), float(xyz[:, 2].max()))
        ext = [_extend_to_box(np.asarray(s["endpoints"][0], float),
                              np.asarray(s["endpoints"][1], float), bounds)
               for s in segments]
        pts_of = [xyz[np.asarray(s["hits"], int)] for s in segments]
        dcache: dict = {}

        def odir(i, e_i):
            key = (i, e_i)
            if key not in dcache:
                dcache[key] = _local_out_dir(
                    pts_of[i], np.asarray(segments[i]["endpoints"][e_i], float),
                    W, wmode)
            return dcache[key]

        best_by_side, kept = {}, []
        for i in range(len(segments)):
            si = segments[i]
            for j in range(i + 1, len(segments)):
                sj = segments[j]
                same = int(si["tpc"]) == int(sj["tpc"])
                if same and not allow_same:
                    continue
                d_seg = _seg_seg_dist(ext[i][0], ext[i][1], ext[j][0], ext[j][1])
                if d_seg > dist_tol:
                    continue
                ep_dist, e_i, e_j = _best_endpoint_pair(
                    np.asarray(si["endpoints"], float), np.asarray(sj["endpoints"], float))
                if ep_dist > ep_tol or (same and ep_dist < same_min_gap):
                    continue
                ui, uj = odir(i, e_i), odir(j, e_j)
                if ui is None or uj is None:
                    continue
                dot = float(ui @ uj)
                if dot > dot_max:
                    continue
                if trans_tol is not None:
                    # endpoint-gap component perpendicular to the mean
                    # continuation direction: genuine joins measure <~0.4 cm,
                    # wrong joins several cm (side-by-side displacement)
                    pa = np.asarray(si["endpoints"][e_i], float)
                    pb = np.asarray(sj["endpoints"][e_j], float)
                    dv = pb - pa
                    um = ui - uj
                    nm = np.linalg.norm(um)
                    if nm > 1e-12:
                        um = um / nm
                        if np.linalg.norm(dv - (dv @ um) * um) > trans_tol:
                            continue
                score = (d_seg / dist_tol + _WE * (ep_dist / ep_tol)
                         + _WA * ((1.0 + dot) / max(1.0 + dot_max, 1e-6))
                         - _WQ * min(float(si["linearity"]), float(sj["linearity"])))
                rec = (score, i, j, e_i, e_j, dot, d_seg, ep_dist)
                for side in ((i, e_i), (j, e_j)):
                    if side not in best_by_side or score < best_by_side[side][0]:
                        best_by_side[side] = rec
                kept.append(rec)
        accepted, seen = [], set()
        for rec in kept:
            _, i, j, e_i, e_j, dot, d_seg, ep_dist = rec
            if best_by_side.get((i, e_i)) is not rec or best_by_side.get((j, e_j)) is not rec:
                continue
            if (i, j) in seen:
                continue
            seen.add((i, j))
            accepted.append({"i": int(i), "j": int(j), "end_i": int(e_i), "end_j": int(e_j),
                             "dot": float(dot), "segment_dist": float(d_seg),
                             "endpoint_dist": float(ep_dist),
                             "tpc_i": int(segments[i]["tpc"]), "tpc_j": int(segments[j]["tpc"]),
                             "score": float(rec[0]),
                             "source_i": str(segments[i]["source"]),
                             "source_j": str(segments[j]["source"])})
        parent = np.arange(len(segments))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for edge in accepted:
            ra, rb = find(edge["i"]), find(edge["j"])
            if ra != rb:
                parent[rb] = ra
        groups: dict = {}
        for i in range(len(segments)):
            groups.setdefault(find(i), []).append(int(i))
        global_tracks = []
        for members in groups.values():
            all_hits = np.unique(np.concatenate(
                [np.asarray(segments[m]["hits"], int) for m in members]))
            metrics = fit_line_metrics(xyz[all_hits])
            global_tracks.append({
                "segments": [segments[m] for m in members],
                "segment_indices": list(members),
                "hit_indices": np.asarray(all_hits, dtype=int),
                "point": np.asarray(metrics["point"], np.float64),
                "direction": np.asarray(metrics["direction"], np.float64),
                "endpoints": np.asarray(metrics["endpoints"], np.float64)})
        debug = {"candidate_edges": [], "accepted_edges": accepted,
                 "n_candidate_edges": int(len(kept)),
                 "n_accepted_edges": int(len(accepted))}
        return global_tracks, debug

    return matcher
