"""Convert v0.5.6 release dumps (npz) into per-event .pt files.

Each .pt is a torch-saved dict:
  per-hit  : x, y, z, energy, io_group, tpc  (float32/int16)
             labels (int32, v0.5.5 clustering; -1 = noise)
             t0 (float32, final chain assignment, ticks; NaN never occurs
                 at 100% coverage), t0_phase1 (float32, backbone-stage value)
             truth_t0 (float32, MC truth, ticks; NaN = no truth)
  clusters : dict label -> {type, backbone, tpcs, e_total, reco_t0, truth_t0,
                            e_wrong, confidence:{z_min, z_own, chi2_loc, E,
                            n_tpc}}   <-- per-cluster confidence (v0.5.6)
  meta     : flow_file, event, split_index, chain ("v0.4.1 matching +
             v0.5.5 clustering"), ns_total_eff of this event
"""
import glob
import json
import os
import sys

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pt")


def convert(npz_path, row):
    d = np.load(npz_path, allow_pickle=True)
    lab_rows = json.loads(str(d["lab_rows"].item()))
    conf = json.loads(str(d["cluster_conf"].item())) if "cluster_conf" in d else {}
    clusters = {}
    split_index = int(d["split_index"]) if "split_index" in d else None
    for r in lab_rows:
        lab = int(r["label"])
        clusters[lab] = {
            "type": r.get("type", "?"),
            "backbone": bool(split_index is not None and lab < split_index),
            "tpcs": r.get("tpcs", []),
            "e_total": float(r.get("e_ok", 0.0)),
            "e_wrong": float(r.get("e_wrong", 0.0)),
            "reco_t0": r.get("reco_t0"),
            "truth_t0": r.get("truth"),
            "confidence": conf.get(str(lab)),
        }
    ev = {
        "x": torch.from_numpy(np.asarray(d["x"], np.float32)),
        "y": torch.from_numpy(np.asarray(d["y"], np.float32)),
        "z": torch.from_numpy(np.asarray(d["z"], np.float32)),
        "energy": torch.from_numpy(np.asarray(d["e"], np.float32)),
        "io_group": torch.from_numpy(np.asarray(d["io_group"], np.int16)),
        "tpc": torch.from_numpy(np.asarray(d["tpc"], np.int16)),
        "labels": torch.from_numpy(np.asarray(d["labels"], np.int32)),
        "t0": torch.from_numpy(np.asarray(d["ts_final"], np.float32)),
        "t0_phase1": torch.from_numpy(np.asarray(d["ts_p1"], np.float32)),
        "truth_t0": torch.from_numpy(np.asarray(d["truth_t0"], np.float32)),
        "clusters": clusters,
        "meta": {
            "flow_file": str(d["src_file"]),
            "event": int(d["event"]),
            "split_index": split_index,
            "chain": "v0.5.6 release = v0.5.5 clustering + v0.4.1 matching",
            "ns_total_eff": (row or {}).get("m_rescued", {}).get("ns_total_eff"),
        },
    }
    stem = os.path.basename(npz_path)[:-4]
    out = os.path.join(OUT, stem + ".pt")
    torch.save(ev, out)
    return out


def main():
    dump_dir, rows_glob = sys.argv[1], sys.argv[2]
    rows = {}
    for f in glob.glob(rows_glob):
        for line in open(f):
            r = json.loads(line)
            rows[(r["file"].split(".")[-3], r["event"])] = r
    os.makedirs(OUT, exist_ok=True)
    done = []
    for npz in sorted(glob.glob(os.path.join(dump_dir, "*.npz"))):
        stem = os.path.basename(npz)[:-4]
        ftag, ev = stem.split("_ev")
        out = convert(npz, rows.get((ftag, int(ev))))
        done.append(out)
        print("wrote", out)
    print(f"{len(done)} .pt files")


if __name__ == "__main__":
    main()
