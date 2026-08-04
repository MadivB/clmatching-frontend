"""Standalone v0.5.5 front-end clustering: re-run + evaluate on a .pt event.

CPU-only, no NERSC dependencies: file-loads the clustering toolbox and the
vertex merge directly (no package imports, no torch model, no light data
needed).  This is THE tuning loop for improving the clustering:

    python recluster.py pt/flow0000001.pt --max-events 1        # quick check
    python recluster.py pt/flow0000001.pt --params my.json      # tuned params
    python recluster.py pt/*.pt --params my.json                # all 126 events

It prints purity / completeness / backbone stats vs MC truth and compares
against the labels shipped inside the .pt (the NERSC v0.5.5 result).
Edit clustering_v055/clustering_defaults.json (or a copy) and iterate.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.abspath(os.path.dirname(__file__))
CLU = os.path.join(HERE, "clustering_v055")


def _file_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_clustering(ev, params):
    """v0.5.5 = toolbox RANSAC clustering + v0.2.5 vertex merge."""
    toolbox = _file_load("clu_toolbox",
                         os.path.join(CLU, "global_track_clustering_toolbox_v11_2.py"))
    if params.get("fast_ransac_enable", False):
        fr = _file_load("clu_fr", os.path.join(HERE, "fast_ransac.py"))
        fr.enable(toolbox, SimpleNamespace(**params))
    if params.get("segment_split_enable", False):
        sp = _file_load("clu_sp", os.path.join(HERE, "segment_split.py"))
        toolbox._build_tpc_segments_toolbox = sp.wrap_builder(
            toolbox._build_tpc_segments_toolbox, SimpleNamespace(**params))
    if params.get("stitch_directional_enable", False):
        sd = _file_load("clu_sd", os.path.join(HERE, "stitch_directional.py"))
        toolbox._match_segments_across_tpcs_toolbox = sd.make_matcher(
            SimpleNamespace(**params), toolbox._fit_line_metrics)
    sig = set(inspect.signature(toolbox.build_global_labels_toolbox).parameters)
    kw = {k: v for k, v in params.items() if k in sig}
    kw["return_label_info"] = True
    kw["return_debug_info"] = True
    labels, split_index, label_info, debug = toolbox.build_global_labels_toolbox(
        ev["x"].numpy(), ev["y"].numpy(), ev["z"].numpy(),
        ev["io_group"].numpy(), **kw)
    if params.get("vertex_merge_enable", True):
        vm = _file_load("clu_vm", os.path.join(CLU, "vertex_merge.py"))
        cfg = SimpleNamespace(**params)
        # merge_vertex_tracks returns (labels, stats); it mutates label_info
        # in place and never changes split_index
        labels, vm_stats = vm.merge_vertex_tracks(
            labels_global=labels, split_index=split_index, label_info=label_info,
            x=ev["x"].numpy(), y=ev["y"].numpy(), z=ev["z"].numpy(), config=cfg)
        debug = dict(debug or {})
        debug["vertex_merge"] = vm_stats
    if params.get("endpoint_rejoin_enable", False):
        err = _file_load("clu_er", os.path.join(HERE, "endpoint_rejoin.py"))
        cfg = SimpleNamespace(**params)
        tpc = (ev["tpc"].numpy() if "tpc" in ev
               else (ev["io_group"].numpy().astype(int) - 1) // 2)
        xyz_ = np.stack([ev["x"].numpy(), ev["y"].numpy(), ev["z"].numpy()], 1)
        labels, er_stats = err.rejoin_endpoints(
            labels, int(split_index), label_info, xyz_.astype(np.float64),
            tpc, cfg)
        debug = dict(debug or {})
        debug["endpoint_rejoin"] = er_stats
    if params.get("vertex_pinpoint_enable", False):
        vpp = _file_load("clu_vp", os.path.join(HERE, "vertex_pinpoint.py"))
        cfg = SimpleNamespace(**params)
        tpc = (ev["tpc"].numpy() if "tpc" in ev
               else (ev["io_group"].numpy().astype(int) - 1) // 2)
        labels, vp_stats = vpp.pinpoint_vertex_merge(
            labels_global=labels, split_index=split_index, label_info=label_info,
            x=ev["x"].numpy(), y=ev["y"].numpy(), z=ev["z"].numpy(),
            config=cfg, tpc=tpc)
        debug = dict(debug or {})
        debug["vertex_pinpoint"] = vp_stats
    if params.get("blob_refine_enable", False):
        br = _file_load("clu_br", os.path.join(HERE, "blob_refine.py"))
        xyz_ = np.stack([ev["x"].numpy(), ev["y"].numpy(), ev["z"].numpy()], 1)
        labels, br_stats = br.refine_blobs(labels, int(split_index),
                                           xyz_.astype(np.float64),
                                           SimpleNamespace(**params))
        debug = dict(debug or {})
        debug["blob_refine"] = br_stats
    if params.get("two_track_split_enable", False):
        tts = _file_load("clu_tts", os.path.join(HERE, "two_track_split.py"))
        tpc = (ev["tpc"].numpy() if "tpc" in ev
               else (ev["io_group"].numpy().astype(int) - 1) // 2)
        xyz_ = np.stack([ev["x"].numpy(), ev["y"].numpy(), ev["z"].numpy()], 1)
        labels, tts_stats = tts.split_two_track(
            labels, int(split_index), label_info, xyz_.astype(np.float64),
            ev["energy"].numpy().astype(np.float64), tpc,
            SimpleNamespace(**params))
        debug = dict(debug or {})
        debug["two_track_split"] = tts_stats
    if params.get("frag_absorb_enable", False):
        fa = _file_load("clu_fa", os.path.join(HERE, "frag_absorb.py"))
        xyz_ = np.stack([ev["x"].numpy(), ev["y"].numpy(), ev["z"].numpy()], 1)
        labels, fa_stats = fa.absorb_fragments(labels, int(split_index),
                                               xyz_.astype(np.float64),
                                               SimpleNamespace(**params))
        debug = dict(debug or {})
        debug["frag_absorb"] = fa_stats
    if params.get("two_track_flag_enable", False):
        sp2 = _file_load("clu_sp2", os.path.join(HERE, "segment_split.py"))
        xyz = np.stack([ev["x"].numpy(), ev["y"].numpy(), ev["z"].numpy()], 1)
        tpc = (ev["tpc"].numpy() if "tpc" in ev
               else (ev["io_group"].numpy().astype(int) - 1) // 2)
        flags = sp2.two_track_flag(np.asarray(labels), xyz.astype(np.float64),
                                   ev["energy"].numpy().astype(np.float64),
                                   tpc, SimpleNamespace(**params))
        if label_info is not None:   # tracks only: showers/blobs are wide by nature
            flags = {lab: tpcs for lab, tpcs in flags.items()
                     if str(label_info.get(lab, {}).get("type", "")).lower() == "track"}
        for lab, tpcs in flags.items():
            if label_info is not None and lab in label_info:
                label_info[lab]["two_track_suspect"] = tpcs
        debug = dict(debug or {})
        debug["two_track_flags"] = flags
    return np.asarray(labels), int(split_index), label_info, debug


def truth_groups(truth_t0, gap=3.0):
    """Group hits into truth interactions by t0 proximity (> gap ticks splits)."""
    g = np.full(truth_t0.shape, -1, np.int64)
    fin = np.isfinite(truth_t0)
    order = np.argsort(truth_t0[fin])
    idx = np.flatnonzero(fin)[order]
    gid, last = 0, None
    for i in idx:
        t = truth_t0[i]
        if last is not None and t - last > gap:
            gid += 1
        g[i] = gid
        last = t
    return g


def evaluate(labels, ev, name):
    e = np.clip(ev["energy"].numpy().astype(np.float64), 0, None)
    tg = truth_groups(ev["truth_t0"].numpy().astype(np.float64))
    ok = (labels >= 0) & (tg >= 0)
    # PURITY: energy-weighted majority-truth-group fraction per cluster
    pur_num = pur_den = 0.0
    for lab in np.unique(labels[ok]):
        m = ok & (labels == lab)
        w = np.bincount(tg[m], weights=e[m])
        pur_num += float(w.max())
        pur_den += float(w.sum())
    # COMPLETENESS: per truth group, the largest single-cluster energy share
    com_num = com_den = 0.0
    for g in np.unique(tg[ok]):
        m = ok & (tg == g)
        w = np.bincount(labels[m], weights=e[m])
        com_num += float(w.max())
        com_den += float(w.sum())
    n_noise = int(np.sum(labels < 0))
    print(f"  [{name}] clusters {len(np.unique(labels[labels >= 0])):4d} | "
          f"noise hits {n_noise:6d} | purity {pur_num / max(pur_den, 1e-9):.4f} | "
          f"completeness {com_num / max(com_den, 1e-9):.4f}")
    return pur_num / max(pur_den, 1e-9), com_num / max(com_den, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_files", nargs="+")
    ap.add_argument("--params", default=os.path.join(CLU, "clustering_defaults.json"))
    ap.add_argument("--skip-rerun", action="store_true",
                    help="only evaluate the labels shipped in the .pt")
    ap.add_argument("--max-events", type=int, default=None,
                    help="per flow-file .pt, only process the first N events "
                         "(quick parameter checks)")
    args = ap.parse_args()
    params = json.load(open(args.params))
    P = C = P0 = C0 = 0.0
    n = 0
    for fp in args.pt_files:
        blob = torch.load(fp, weights_only=False)
        # merged format: one .pt per FLOW file, {"events": {ev_id: event_dict}}
        events = (list(blob["events"].items()) if "events" in blob
                  else [(blob["meta"]["event"], blob)])
        if args.max_events is not None:
            events = events[: args.max_events]
        print(f"{os.path.basename(fp)}: {len(events)} events "
              f"({os.path.basename(events[0][1]['meta']['flow_file'])})")
        for ev_id, ev in events:
            n += 1
            print(f" event {ev_id}: {ev['x'].shape[0]} hits")
            p0, c0 = evaluate(ev["labels"].numpy(), ev, "shipped v0.5.5")
            P0 += p0; C0 += c0
            if not args.skip_rerun:
                labels, split_index, _li, _dbg = run_clustering(ev, params)
                p, c = evaluate(labels, ev, "re-clustered   ")
                P += p; C += c
    print(f"\nMEAN over {n} events shipped   : purity {P0 / n:.4f} completeness {C0 / n:.4f}")
    if not args.skip_rerun:
        print(f"MEAN over {n} events re-cluster: purity {P / n:.4f} completeness {C / n:.4f}")


if __name__ == "__main__":
    main()
