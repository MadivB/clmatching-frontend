"""Per-TRACK purity analysis vs MC truth + standalone chain runner.

Three modes (all CPU-only, no GPU anywhere):

  # per-track purity catalog (shipped labels by default; --rerun re-clusters)
  python track_purity.py scan pt/flow0000001.pt [pt/*.pt] [--rerun] [--top 20]
                         [--csv tracks.csv] [--broken-csv broken.csv]

  # stage-by-stage standalone run of the full front-end chain on one event
  python track_purity.py stages pt/flow0000001.pt --event 0 [--vm]

  # 3D diagnosis HTML for suspicious cluster(s): own hits vs foreign hits
  python track_purity.py html pt/flow0000001.pt --event 0 --labels 42[,57]

Truth definition = recluster.py's: truth_t0 grouped with a 3-tick gap (one group
per interaction; no per-particle MC ids exist in the dumps).  Purity of a cluster
= energy fraction of its truth-matched hits belonging to its dominant group.
"Foreign" energy = truth-matched energy from any OTHER group ("things from other
places") — the poison for charge-light matching.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from recluster import run_clustering, truth_groups  # reuse the release harness

DEFAULTS = os.path.join(HERE, "clustering_v055", "clustering_defaults.json")


# ---------------------------------------------------------------- loading

def load_events(fp):
    blob = torch.load(fp, weights_only=False)
    if "events" in blob:
        return list(blob["events"].items())
    return [(blob["meta"]["event"], blob)]


def get_labels_and_types(ev, rerun, params):
    """Return (labels, type_of: label->str, backbone: label->bool)."""
    if rerun:
        labels, split_index, linfo, _dbg = run_clustering(ev, params)
        type_of = {int(l): str(i.get("type", "?")) for l, i in (linfo or {}).items()}
        backbone = {int(l): int(l) < split_index for l in np.unique(labels[labels >= 0])}
        return labels, type_of, backbone
    labels = ev["labels"].numpy().astype(int)
    cl = ev.get("clusters", {}) or {}
    type_of = {int(l): str(d.get("type", "?")) for l, d in cl.items()}
    backbone = {int(l): bool(d.get("backbone", False)) for l, d in cl.items()}
    return labels, type_of, backbone


def event_arrays(ev):
    e = np.clip(ev["energy"].numpy().astype(np.float64), 0, None)
    tg = truth_groups(ev["truth_t0"].numpy().astype(np.float64))
    tpc = ev["tpc"].numpy().astype(int)
    xyz = np.stack([ev["x"].numpy(), ev["y"].numpy(), ev["z"].numpy()], 1).astype(np.float64)
    tt = ev["truth_t0"].numpy().astype(np.float64)
    return xyz, e, tg, tpc, tt


# ---------------------------------------------------------------- geometry

def pca_line(pts):
    """(centroid, direction, tmin, tmax, linearity) of a point cloud."""
    c = pts.mean(0)
    q = pts - c
    try:
        _, s, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return c, np.array([1.0, 0, 0]), 0.0, 0.0, 0.0
    d = vt[0]
    t = q @ d
    lin = float(s[0] ** 2 / max(float((s ** 2).sum()), 1e-12))
    return c, d, float(t.min()), float(t.max()), lin


# ---------------------------------------------------------------- scan mode

def analyze_event(fp, ev_id, ev, rerun, params):
    """Per-cluster purity rows + broken-track candidate rows for one event."""
    labels, type_of, backbone = get_labels_and_types(ev, rerun, params)
    xyz, e, tg, tpc, tt = event_arrays(ev)

    ok = (labels >= 0) & (tg >= 0)
    ulab = np.unique(labels[labels >= 0])
    li = np.searchsorted(ulab, labels[ok])
    ngr = int(tg.max()) + 1 if (tg >= 0).any() else 0
    M = np.zeros((len(ulab), ngr))
    np.add.at(M, (li, tg[ok]), e[ok])

    # mean truth_t0 per group (for "other places" = other interaction times)
    t0_of_group = np.full(ngr, np.nan)
    for g in range(ngr):
        m = tg == g
        if m.any():
            t0_of_group[g] = np.nanmean(tt[m])

    rows, geo = [], {}
    for k, lab in enumerate(ulab):
        m = labels == lab
        row = M[k]
        e_truth = float(row.sum())
        e_tot = float(e[m].sum())
        if e_truth <= 0:
            continue
        dom = int(row.argmax())
        e_dom = float(row[dom])
        e_for = e_truth - e_dom
        pts = xyz[m]
        c, d, t0_, t1_, lin = pca_line(pts) if m.sum() >= 3 else (pts.mean(0), None, 0, 0, 0)
        length = t1_ - t0_
        geo[int(lab)] = (c, d, t0_, t1_, lin, dom)
        # classification hint for foreign energy
        hint = ""
        if e_for > 0:
            per_tpc_for = {}
            fm = m & ok & (tg != dom)
            for t in np.unique(tpc[fm]):
                per_tpc_for[int(t)] = float(e[fm & (tpc == t)].sum())
            if per_tpc_for:
                worst_tpc, worst_e = max(per_tpc_for.items(), key=lambda kv: kv[1])
                own_in_tpc = float(e[m & ok & (tg == dom) & (tpc == worst_tpc)].sum())
                if worst_e > 0.5 * e_for and worst_e > 4 * max(own_in_tpc, 1e-9):
                    hint = f"bad-stitch?(tpc{worst_tpc})"
                elif d is not None:
                    perp_f = np.linalg.norm((xyz[fm] - c) - np.outer((xyz[fm] - c) @ d, d), axis=1)
                    perp_o = np.linalg.norm((xyz[m & ok & (tg == dom)] - c)
                                            - np.outer((xyz[m & ok & (tg == dom)] - c) @ d, d), axis=1)
                    if len(perp_f) and np.median(perp_f) > 1.5 * max(np.quantile(perp_o, 0.9), 1e-9):
                        hint = "attached-blob?"
                    else:
                        hint = "mixed-inline?"
        top_foreign = ""
        if e_for > 0:
            order = np.argsort(row)[::-1]
            fg = [g for g in order if g != dom and row[g] > 0][:2]
            top_foreign = "; ".join(
                f"g{g}:{row[g]:.1f}MeV(t0 {t0_of_group[g]:.0f})" for g in fg)
        rows.append(dict(
            file=os.path.basename(fp), event=ev_id, label=int(lab),
            type=type_of.get(int(lab), "?"), backbone=backbone.get(int(lab), False),
            n_hits=int(m.sum()), e_mev=e_tot, length_cm=length,
            n_tpcs=len(np.unique(tpc[m])), linearity=lin,
            purity=e_dom / e_truth, foreign_mev=e_for,
            dom_group=dom, dom_t0=t0_of_group[dom],
            top_foreign=top_foreign, hint=hint))

    # broken-track candidates: same dominant group, collinear, facing endpoints
    broken = []
    tracks = [r for r in rows if r["type"] == "track" and r["length_cm"] > 15
              and geo[r["label"]][1] is not None]
    by_group = {}
    for r in tracks:
        by_group.setdefault(r["dom_group"], []).append(r)
    for g, rs in by_group.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                ca, da, ta0, ta1, _, _ = geo[a["label"]]
                cb, db, tb0, tb1, _, _ = geo[b["label"]]
                ang = np.degrees(np.arccos(min(1.0, abs(float(da @ db)))))
                if ang > 15:
                    continue
                ea = [ca + ta0 * da, ca + ta1 * da]
                eb = [cb + tb0 * db, cb + tb1 * db]
                gap = min(float(np.linalg.norm(p - q)) for p in ea for q in eb)
                if gap > 80:
                    continue
                pb = min(eb, key=lambda q: min(float(np.linalg.norm(p - q)) for p in ea))
                off = float(np.linalg.norm((pb - ca) - ((pb - ca) @ da) * da))
                if off > 8:
                    continue
                broken.append(dict(
                    file=os.path.basename(fp), event=ev_id, group=g,
                    label_a=a["label"], label_b=b["label"],
                    gap_cm=gap, angle_deg=float(ang), offset_cm=off,
                    e_a=a["e_mev"], e_b=b["e_mev"],
                    len_a=a["length_cm"], len_b=b["length_cm"],
                    cross_tpc=a["n_tpcs"] + b["n_tpcs"] > 1
                    and bool(set(np.unique(tpc[labels == a["label"]]))
                             != set(np.unique(tpc[labels == b["label"]])))))
    return rows, broken


def mode_scan(args):
    params = json.load(open(args.params))
    all_rows, all_broken = [], []
    for fp in args.pt_files:
        events = load_events(fp)
        if args.max_events:
            events = events[: args.max_events]
        for ev_id, ev in events:
            rows, broken = analyze_event(fp, ev_id, ev, args.rerun, params)
            all_rows += rows
            all_broken += broken
            print(f"{os.path.basename(fp)} ev {ev_id}: {len(rows)} clusters analyzed, "
                  f"{sum(1 for r in rows if r['type'] == 'track')} tracks, "
                  f"{len(broken)} broken-track candidates")

    tracks = [r for r in all_rows if r["type"] == "track"]
    e_tr = sum(r["e_mev"] for r in tracks)
    e_for = sum(r["foreign_mev"] for r in tracks)
    print("\n================ TRACK PURITY SUMMARY ================")
    print(f"track clusters: {len(tracks)} | track energy {e_tr:.0f} MeV | "
          f"foreign (other-interaction) energy {e_for:.0f} MeV "
          f"({100 * e_for / max(e_tr, 1e-9):.2f}%)")
    for lmin in (0, 30, 100):
        sel = [r for r in tracks if r["length_cm"] >= lmin]
        if not sel:
            continue
        w = sum(r["e_mev"] for r in sel)
        f = sum(r["foreign_mev"] for r in sel)
        nbad = sum(1 for r in sel if r["purity"] < 0.9)
        print(f"  length >= {lmin:3d} cm: {len(sel):5d} tracks | "
              f"E {w:9.0f} MeV | foreign {100 * f / max(w, 1e-9):5.2f}% | "
              f"purity<0.9: {nbad}")

    worst = sorted(tracks, key=lambda r: -r["foreign_mev"])[: args.top]
    print(f"\n---- worst {len(worst)} tracks by foreign energy ----")
    hdr = (" file            ev lab   type  E[MeV] len[cm] tpcs purity foreign "
           " hint              top foreign groups")
    print(hdr)
    for r in worst:
        print(f" {r['file'][:15]:15s} {r['event']:2d} {r['label']:4d} {r['type'][:6]:6s}"
              f" {r['e_mev']:7.1f} {r['length_cm']:7.1f} {r['n_tpcs']:4d}"
              f" {r['purity']:6.3f} {r['foreign_mev']:7.1f}  {r['hint']:17s}"
              f" {r['top_foreign']}")

    if all_broken:
        worstb = sorted(all_broken, key=lambda r: -(r["e_a"] + r["e_b"]))[: args.top]
        print(f"\n---- {len(all_broken)} broken-track candidates "
              f"(collinear same-truth track pairs; top {len(worstb)}) ----")
        for r in worstb:
            print(f" {r['file'][:15]:15s} ev {r['event']:2d} labels "
                  f"{r['label_a']}+{r['label_b']}: gap {r['gap_cm']:5.1f} cm, "
                  f"angle {r['angle_deg']:4.1f} deg, offset {r['offset_cm']:4.1f} cm, "
                  f"E {r['e_a']:.0f}+{r['e_b']:.0f} MeV, len {r['len_a']:.0f}+{r['len_b']:.0f} cm"
                  f"{', cross-TPC' if r['cross_tpc'] else ', same-TPC'}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {args.csv} ({len(all_rows)} clusters)")
    if args.broken_csv and all_broken:
        with open(args.broken_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_broken[0].keys()))
            w.writeheader()
            w.writerows(all_broken)
        print(f"wrote {args.broken_csv} ({len(all_broken)} pairs)")


# ---------------------------------------------------------------- stages mode

def _summ(x, depth=0, key=""):
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{key}: dict({len(x)})")
        for k, v in list(x.items())[:12]:
            _summ(v, depth + 1, str(k))
        if len(x) > 12:
            print(f"{pad}  ... {len(x) - 12} more keys")
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{key}: {type(x).__name__}[{len(x)}]"
              + (f" first={type(x[0]).__name__}" if len(x) else ""))
    elif isinstance(x, np.ndarray):
        print(f"{pad}{key}: ndarray{x.shape} {x.dtype}")
    else:
        print(f"{pad}{key}: {x!r}"[:140])


def mode_stages(args):
    params = json.load(open(args.params))
    if args.vm:
        params["vertex_merge_enable"] = True
    events = dict(load_events(args.pt_files[0]))
    ev = events[args.event]
    xyz, e, tg, tpc, tt = event_arrays(ev)
    print(f"event {args.event}: {len(e)} hits, {int(tg.max()) + 1} truth interactions, "
          f"vertex_merge_enable={params.get('vertex_merge_enable')}")
    labels, split_index, linfo, dbg = run_clustering(ev, params)
    print(f"\nfinal: {len(np.unique(labels[labels >= 0]))} clusters "
          f"(split_index={split_index} backbone), {int((labels < 0).sum())} noise hits")
    types = {}
    for l, i in (linfo or {}).items():
        types[i.get("type", "?")] = types.get(i.get("type", "?"), 0) + 1
    print("label types:", types)
    print("\n---- debug info by stage ----")
    _summ(dbg or {}, key="debug")
    # per-stage headline numbers if present
    for k in ("segments", "matches", "vertex", "absorption", "vertex_merge"):
        if dbg and k in dbg:
            v = dbg[k]
            if isinstance(v, dict):
                print(f"\n[{k}] " + ", ".join(f"{kk}={str(vv)[:60]}" for kk, vv in list(v.items())[:8]
                                              if not isinstance(vv, (list, dict, np.ndarray))))


# ---------------------------------------------------------------- html mode

def mode_html(args):
    import plotly.graph_objects as go
    params = json.load(open(args.params))
    events = dict(load_events(args.pt_files[0]))
    ev = events[args.event]
    labels, type_of, _bb = get_labels_and_types(ev, args.rerun, params)
    xyz, e, tg, tpc, tt = event_arrays(ev)
    labs = [int(s) for s in args.labels.split(",")]
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#bcbd22"]
    fig = go.Figure()
    tpcs = set()
    title = []
    for i, lab in enumerate(labs):
        m = labels == lab
        tpcs |= set(np.unique(tpc[m]).tolist())
        ok = m & (tg >= 0)
        row = np.bincount(tg[ok], weights=e[ok]) if ok.any() else np.array([0.0])
        dom = int(row.argmax())
        own = m & (tg == dom)
        foreign = m & (tg >= 0) & (tg != dom)
        naked = m & (tg < 0)
        pur = row[dom] / max(row.sum(), 1e-9)
        title.append(f"label {lab} ({type_of.get(lab, '?')}, {e[m].sum():.0f} MeV, "
                     f"purity {pur:.3f}, foreign {row.sum() - row[dom]:.1f} MeV)")
        fig.add_trace(go.Scatter3d(x=xyz[own, 0], y=xyz[own, 1], z=xyz[own, 2],
            mode="markers", marker=dict(size=2.4, color=palette[i % len(palette)]),
            name=f"lab {lab} own (truth g{dom})",
            text=[f"lab {lab} g{g} t0={t:.0f}" for g, t in zip(tg[own], tt[own])],
            hoverinfo="text"))
        if naked.any():
            fig.add_trace(go.Scatter3d(x=xyz[naked, 0], y=xyz[naked, 1], z=xyz[naked, 2],
                mode="markers", marker=dict(size=2.0, color="rgba(120,120,120,0.7)"),
                name=f"lab {lab} no-truth", hoverinfo="skip"))
        if foreign.any():
            fig.add_trace(go.Scatter3d(x=xyz[foreign, 0], y=xyz[foreign, 1], z=xyz[foreign, 2],
                mode="markers", marker=dict(size=5.0, color="red", symbol="diamond",
                                            line=dict(width=0.5, color="black")),
                name=f"lab {lab} FOREIGN",
                text=[f"lab {lab} g{g} t0={t:.0f}" for g, t in zip(tg[foreign], tt[foreign])],
                hoverinfo="text"))
    ctx = np.isin(tpc, sorted(tpcs)) & ~np.isin(labels, labs)
    fig.add_trace(go.Scatter3d(x=xyz[ctx, 0], y=xyz[ctx, 1], z=xyz[ctx, 2], mode="markers",
        marker=dict(size=1.3, color="rgba(160,175,195,0.25)"),
        name=f"context (TPCs {sorted(tpcs)})", hoverinfo="skip"))
    fig.update_layout(title="  |  ".join(title), legend=dict(orientation="h", y=-0.02))
    fig.update_scenes(aspectmode="data", xaxis_title="x [cm]",
                      yaxis_title="y [cm]", zaxis_title="z [cm]")
    out = os.path.join(HERE, "plots", "html",
                       f"badtrack_{os.path.basename(args.pt_files[0]).split('.')[0]}"
                       f"_ev{args.event}_lab{'_'.join(str(l) for l in labs)}"
                       f"{'_rerun' if args.rerun else ''}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, default_width="100%", default_height="94vh")
    print("saved:", out)
    for t in title:
        print(" ", t)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("pt_files", nargs="+")
        p.add_argument("--params", default=DEFAULTS)
        p.add_argument("--rerun", action="store_true",
                       help="re-run the clustering instead of using shipped labels")

    ps = sub.add_parser("scan", help="per-track purity catalog")
    common(ps)
    ps.add_argument("--max-events", type=int, default=None)
    ps.add_argument("--top", type=int, default=20)
    ps.add_argument("--csv", default=None)
    ps.add_argument("--broken-csv", default=None)

    pg = sub.add_parser("stages", help="standalone stage-by-stage run, one event")
    common(pg)
    pg.add_argument("--event", type=int, required=True)
    pg.add_argument("--vm", action="store_true", help="enable the vertex (pinpoint) merge")

    ph = sub.add_parser("html", help="3D diagnosis HTML for given labels")
    common(ph)
    ph.add_argument("--event", type=int, required=True)
    ph.add_argument("--labels", required=True, help="comma-separated cluster labels")

    args = ap.parse_args()
    {"scan": mode_scan, "stages": mode_stages, "html": mode_html}[args.mode](args)


if __name__ == "__main__":
    main()
