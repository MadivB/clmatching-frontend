"""Vertex pinpointing v0.3 (Billy's endpoint-DBSCAN + local-extension design,
boundary-aware margins + BACK-WALL VETO) - defaults validated on all 126 events.

Algorithm:
  1. DBSCAN over track END-POINT anchors only (anchor = mean of the 3 axially
     most-extreme hits per end), eps = 5 cm, min_samples = 2.
  2. Inside each endpoint cluster, each participating track end gets a LOCAL
     straight-line fit using only that track's hits within 10 cm of the anchor
     (never the whole track), oriented outward and extended eps cm beyond the
     anchor (catches vertices in walls / dead material just outside the charge).
  3. For each track pair, take the end combination with the smallest closest
     approach (doca) of the extended local segments; the candidate vertex is the
     midpoint of the closest-approach points.
  4. Boundary-aware acceptance: vertex depth inside the charge volume (distance
     to the nearest face of the deepest per-TPC hit bounding box containing it;
     0 if in a gap/wall) >= 3 cm  =>  interior, accept if doca < 1.5 cm;
     otherwise boundary, accept if doca < 0.75 cm. Rationale (measured): real
     interior vertices are messy and need up to ~2.5 cm slack, while ALL
     observed fake vertices (unrelated tracks meeting by coincidence) pinpoint
     within ~2 cm of a charge boundary - wall, module gap, or cathode.
  5. BACK-WALL VETO (vp_backwall_cm = 10): any pair whose pinpointed vertex is
     within 10 cm of the downstream z-face of the charge (or beyond it) is NOT
     merged. Beam-correlated through-goers exit there, so exit-point
     coincidences pile up on that face - 6 of the 11 wrong merges observed at
     scale were this pattern, all with near-perfect doca (0.09-0.44 cm) that no
     margin can reject.
  6. Pairs merge individually (partial merging inside an endpoint cluster).
     No angle window: collinear accepts measured ~44:1 correct (they double as
     broken-track rejoins).

Validation v0.4 (126 events; chain = directional stitch dot<=-0.97/W8/ep40 +
stage 3.5 OFF via vertex_min_samples=9999 + THIS pass, blob-inclusive):
  file 1 (13 ev)     : purity 0.9887  completeness 0.3875
  files 2-10 (113 ev): purity 0.9875  completeness 0.3801
  ALL 126            : purity 0.9876  completeness 0.3809
  (previous default - stage 3.5 ON, track-only vp: 0.9867 / 0.3780)
  4395 correct / 192 wrong pairs (95.8% precision); energy in wrongly-merged
  smaller partners 6135 MeV (~0.13% of track energy); 569 ms/event.
  Regression checks PASS: f7ev9 parallel muons separate, f8ev9 fake-V corner
  separate, f2ev7 exit stub separate. Track-only v0.3 numbers (99.5% pair
  precision) remain available via vp_include_clusters=false.

API mirrors vertex_merge.merge_vertex_tracks -> (labels, stats); mutates
label_info in place; split_index unchanged. Pass per-hit `tpc` for full
boundary awareness (module gaps + cathodes); without it only the global
envelope is used. Config keys (fallback defaults): vp_eps_cm 5.0, vp_local_cm
10.0, vp_margin_interior_cm 1.5, vp_margin_boundary_cm 0.75,
vp_boundary_depth_cm 3.0, vp_min_samples 2.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

__all__ = ["pinpoint_vertex_merge", "find_vertex_pairs"]


def _seg_seg_dist(p1, q1, p2, q2):
    """Closest distance between 3D segments [p1,q1], [p2,q2] + closest points."""
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
    cp1, cp2 = p1 + s * d1, p2 + t * d2
    return float(np.linalg.norm(cp1 - cp2)), cp1, cp2


def _end_anchors(pts):
    """Two endpoint anchors: mean of the 3 axially most-extreme hits (global PCA)."""
    c = pts.mean(0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    t = q @ vt[0]
    o = np.argsort(t)
    return pts[o[:3]].mean(0), pts[o[-3:]].mean(0)


def _local_end_line(pts, anchor, local_cm):
    """Line through the hits within local_cm of the anchor, oriented OUTWARD."""
    d = np.linalg.norm(pts - anchor, axis=1)
    sel = pts[d <= local_cm]
    if len(sel) < 3:
        sel = pts[np.argsort(d)[:6]]
    if len(sel) < 2:
        return None
    c = sel.mean(0)
    q = sel - c
    try:
        _, _, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    u = vt[0]
    if u @ (anchor - c) < 0:
        u = -u
    return c, u


def _charge_boxes(xyz, tpc):
    """Axis-aligned hit bounding boxes: per TPC if tpc given, else one envelope."""
    if tpc is None:
        return [(xyz.min(0), xyz.max(0))]
    tpc = np.asarray(tpc)
    return [(xyz[tpc == t].min(0), xyz[tpc == t].max(0)) for t in np.unique(tpc)]


def _depth_in_charge(v, boxes):
    """Depth of point v inside the charge volume: max over containing boxes of the
    distance to that box's nearest face; 0 if inside none (gap / wall)."""
    best = 0.0
    for lo, hi in boxes:
        if np.all(v >= lo) and np.all(v <= hi):
            best = max(best, float(min((v - lo).min(), (hi - v).min())))
    return best


def find_vertex_pairs(x, y, z, labels, track_labels, *, tpc=None,
                      eps_cm=5.0, min_samples=2, local_cm=10.0,
                      margin_interior_cm=1.5, margin_boundary_cm=0.75,
                      boundary_depth_cm=3.0, backwall_cm=10.0,
                      blob_labels=None, margin_boundary_blob_cm=None):
    """Accepted merge pairs [{labels, doca_cm, angle_deg, end_dist_cm, vertex,
    vertex_depth_cm, boundary}, ...]. Per pair the smallest-doca end combination
    is judged; boundary vertices get the tighter margin. BACK-WALL VETO: any
    pair whose vertex lies within backwall_cm of the detector's downstream
    z-face (or beyond it) is rejected outright - through-going beam-correlated
    tracks exit there, so exit-point coincidences pile up on that face.
    Pass backwall_cm=None to disable."""
    xyz = np.column_stack([np.asarray(x, np.float64), np.asarray(y, np.float64),
                           np.asarray(z, np.float64)])
    labels = np.asarray(labels)
    boxes = _charge_boxes(xyz, tpc)
    pts_of = {int(l): xyz[labels == l] for l in track_labels}
    anchors, meta = [], []
    for l in track_labels:
        if len(pts_of[l]) < 5:
            continue
        a0, a1 = _end_anchors(pts_of[l])
        anchors += [a0, a1]
        meta += [(int(l), 0), (int(l), 1)]
    if not anchors:
        return []
    A = np.asarray(anchors)
    cl = DBSCAN(eps=eps_cm, min_samples=min_samples).fit_predict(A)
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for cid in np.unique(cl[cl >= 0]):
        mem = np.flatnonzero(cl == cid)
        fits = {k: _local_end_line(pts_of[meta[k][0]], A[k], local_cm) for k in mem}
        mm = [k for k in mem if fits[k] is not None]
        for ii in range(len(mm)):
            for jj in range(ii + 1, len(mm)):
                ka, kb = mm[ii], mm[jj]
                la, lb = meta[ka][0], meta[kb][0]
                if la == lb:
                    continue
                ca, ua = fits[ka]
                cb, ub = fits[kb]
                pa, pb = A[ka], A[kb]
                doca, cp1, cp2 = _seg_seg_dist(pa - local_cm * ua, pa + eps_cm * ua,
                                               pb - local_cm * ub, pb + eps_cm * ub)
                key = (min(la, lb), max(la, lb))
                if key in best and doca >= best[key]["doca_cm"]:
                    continue
                v = 0.5 * (cp1 + cp2)
                best[key] = dict(labels=[la, lb], doca_cm=float(doca),
                                 angle_deg=float(np.degrees(np.arccos(
                                     np.clip(abs(ua @ ub), -1.0, 1.0)))),
                                 end_dist_cm=float(np.linalg.norm(pa - pb)),
                                 vertex=v.tolist())
    z_back = float(xyz[:, 2].max())
    out = []
    for rec in best.values():
        depth = _depth_in_charge(np.asarray(rec["vertex"]), boxes)
        rec["vertex_depth_cm"] = float(depth)
        rec["boundary"] = bool(depth < boundary_depth_cm)
        rec["backwall"] = bool(backwall_cm is not None
                               and rec["vertex"][2] > z_back - backwall_cm)
        if rec["backwall"]:
            continue
        margin = margin_boundary_cm if rec["boundary"] else margin_interior_cm
        # blob end-directions are weakly defined; at boundary vertices (where
        # wall-coincidences concentrate) blob-involved pairs get a tighter margin
        rec["blob_involved"] = bool(blob_labels and (
            rec["labels"][0] in blob_labels or rec["labels"][1] in blob_labels))
        if (rec["boundary"] and rec["blob_involved"]
                and margin_boundary_blob_cm is not None):
            margin = float(margin_boundary_blob_cm)
        if rec["doca_cm"] < margin:
            out.append(rec)
    return out


def _attach_blobs_to_vertices(labels, split_index, xyz, pairs, parent, find,
                              config, blob_pool):
    """Phase 2 of two-phase mode: blobs ATTACH to established (track-track)
    vertices when an end-line converges into the vertex point. Blobs never
    create vertices and never attach to each other."""
    attach_r = float(getattr(config, "vp_attach_anchor_cm", 8.0))
    local_cm = float(getattr(config, "vp_local_cm", 10.0))
    ext_cm = float(getattr(config, "vp_eps_cm", 5.0))
    m_int = float(getattr(config, "vp_margin_interior_cm", 1.5))
    m_bnd = float(getattr(config, "vp_margin_boundary_cm", 0.75))
    verts = [(np.asarray(p["vertex"]), bool(p["boundary"]), int(p["labels"][0]))
             for p in pairs]
    attaches = []
    for bl in sorted(blob_pool):
        pts = xyz[labels == bl]
        if len(pts) < 5:
            continue
        best = None
        for pa in _end_anchors(pts):
            fit = _local_end_line(pts, pa, local_cm)
            if fit is None:
                continue
            _, u = fit
            s0, s1 = pa - local_cm * u, pa + ext_cm * u
            for v, bnd, owner in verts:
                if np.linalg.norm(v - pa) > attach_r:
                    continue
                # distance from the vertex point to the blob's end segment
                d = s1 - s0
                tt_ = np.clip(((v - s0) @ d) / max(d @ d, 1e-12), 0.0, 1.0)
                dist = float(np.linalg.norm(s0 + tt_ * d - v))
                margin = m_bnd if bnd else m_int
                if dist < margin and (best is None or dist < best[0]):
                    best = (dist, owner, v)
        if best is not None:
            attaches.append(dict(blob=int(bl), owner=int(find(best[1])),
                                 dist_cm=float(best[0]),
                                 vertex=np.asarray(best[2]).tolist()))
    return attaches


def pinpoint_vertex_merge(*, labels_global, split_index, label_info, x, y, z,
                          config, tpc=None):
    """Drop-in counterpart of vertex_merge.merge_vertex_tracks -> (labels, stats).

    v0.4: with vp_include_clusters (default True), leftover-DBSCAN blob labels
    (>= split_index) with at least vp_cluster_min_hits hits participate in the
    pinpointing alongside backbone tracks - an interaction's blobs and stubs
    get pinned to their vertex if their local end-lines genuinely intersect
    there. Track labels are always < split_index, so keep-smallest merging
    preserves the track label and its type for mixed merges."""
    labels = np.asarray(labels_global).copy()
    track_labels = [int(l) for l in range(int(split_index))
                    if str(label_info.get(int(l), {}).get("type", "track")).lower() == "track"
                    and np.any(labels == l)]
    mode = str(getattr(config, "vp_mode", "legacy"))
    blob_labels = set()
    amorphous = set()
    if mode != "two_phase" and bool(getattr(config, "vp_include_clusters", True)):
        cl_min = int(getattr(config, "vp_cluster_min_hits", 15))
        u, cnt = np.unique(labels[labels >= 0], return_counts=True)
        blob_labels = {int(l) for l, c in zip(u, cnt)
                       if int(l) >= int(split_index) and c >= cl_min}
        # Billy's peel: classify DBSCAN remnants; SEGMENT-LIKE ones are full
        # citizens, amorphous ones are handled per vp_blob_policy:
        #   'all'                - every blob participates everywhere (legacy v0.4)
        #   'tracklike'          - amorphous blobs excluded from merging entirely
        #   'tracklike_boundary' - amorphous blobs allowed at interior vertices
        #                          only (rejected at boundary vertices)
        policy = str(getattr(config, "vp_blob_policy", "all"))
        if policy != "all" and blob_labels:
            min_lin = float(getattr(config, "vp_blob_min_lin", 0.85))
            min_len = float(getattr(config, "vp_blob_min_len_cm", 6.0))
            xyz_ = np.column_stack([np.asarray(x, np.float64),
                                    np.asarray(y, np.float64),
                                    np.asarray(z, np.float64)])
            for l in blob_labels:
                pts = xyz_[labels == l]
                c = pts.mean(0)
                q = pts - c
                try:
                    _, sv, vt = np.linalg.svd(q, full_matrices=False)
                except np.linalg.LinAlgError:
                    amorphous.add(l)
                    continue
                t = q @ vt[0]
                lin = float(sv[0] ** 2 / max(float((sv ** 2).sum()), 1e-12))
                if lin < min_lin or float(t.max() - t.min()) < min_len:
                    amorphous.add(l)
            if policy == "tracklike":
                blob_labels = blob_labels - amorphous
                amorphous = set()
        track_labels = track_labels + sorted(blob_labels)
    g = lambda k, d: float(getattr(config, k, d))
    pairs = find_vertex_pairs(
        x, y, z, labels, track_labels, tpc=tpc,
        eps_cm=g("vp_eps_cm", 5.0),
        min_samples=int(getattr(config, "vp_min_samples", 2)),
        local_cm=g("vp_local_cm", 10.0),
        margin_interior_cm=g("vp_margin_interior_cm", 1.5),
        margin_boundary_cm=g("vp_margin_boundary_cm", 0.75),
        boundary_depth_cm=g("vp_boundary_depth_cm", 3.0),
        backwall_cm=(None if getattr(config, "vp_backwall_cm", 10.0) is None
                     else float(getattr(config, "vp_backwall_cm", 10.0))),
        blob_labels=(amorphous if amorphous else blob_labels),
        margin_boundary_blob_cm=(float(getattr(config, "vp_amorph_boundary_margin_cm", 0.0))
                                 if amorphous else
                                 getattr(config, "vp_margin_boundary_blob_cm", None)))
    parent = {l: l for l in track_labels}

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
    attaches = []
    if mode == "two_phase" and pairs and bool(getattr(config, "vp_include_clusters", True)):
        # phase 2: blobs attach to ESTABLISHED track-track vertices only
        cl_min = int(getattr(config, "vp_cluster_min_hits", 15))
        u_l, cnt_l = np.unique(labels[labels >= 0], return_counts=True)
        blob_pool = {int(l) for l, c in zip(u_l, cnt_l)
                     if int(l) >= int(split_index) and c >= cl_min}
        xyz_ = np.column_stack([np.asarray(x, np.float64), np.asarray(y, np.float64),
                                np.asarray(z, np.float64)])
        attaches = _attach_blobs_to_vertices(labels, split_index, xyz_, pairs,
                                             parent, find, config, blob_pool)
        for at in attaches:
            parent[at["blob"]] = find(at["owner"])
            track_labels.append(at["blob"])
    raw = {}
    for l in track_labels:
        raw.setdefault(find(l), []).append(l)
    groups = sorted(sorted(g) for g in raw.values() if len(g) >= 2)
    stats = {"groups": groups, "pairs": pairs, "attaches": attaches,
             "labels_premerge": np.asarray(labels_global).copy()}
    for grp in groups:
        keep = int(grp[0])
        for lab in grp[1:]:
            labels[labels == lab] = keep
            if int(lab) in label_info:
                label_info[int(lab)]["n_hits"] = 0
                label_info[int(lab)]["merged_into"] = keep
        if keep in label_info:
            label_info[keep]["vertex_merged"] = True
            label_info[keep]["merged_from"] = [int(l) for l in grp[1:]]
    return labels, stats
