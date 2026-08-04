# CLMatching v0.5.6 release — offline clustering-tuning package

**Purpose of this download (read this first, local Claude):** the owner is
traveling and will CONTINUE IMPROVING THE FRONT-END CLUSTERING (v0.5.5) of the
DUNE ND-LAr charge–light matching chain on a local PC, without NERSC access.
Everything needed for that loop is in this folder: fully-processed events
from 10 FLOW files as per-event `.pt` files (hits, truth, v0.5.5 cluster labels, final t0 assignments,
per-cluster confidence), the complete clustering source code, and a
standalone CPU re-clustering + evaluation harness. The GPU/ML parts of the
chain (light prediction, matching) are NOT needed for clustering work — the
`.pt` files already carry their outputs as fixed context.

## Version map

| version | meaning |
|---|---|
| **v0.5.5** | the front-end clustering stack: `global_track_clustering_toolbox_v11_2.py` (per-TPC RANSAC track finding + shower grouping + DBSCAN leftover rescue + track noise absorption + length gate + cross-TPC track matching) followed by the v0.2.5 **vertex-track merge** (`vertex_merge.py`); optional opt-in passes: `intersection_refine.py`, `shower_absorb.py` (both OFF in this release) |
| **v0.5.6** | the ready-release chain that produced the `.pt`: v0.5.5 clustering + the **v0.4.1 matching chain** (merged permutation pool, trigger rescue, V2, family expansion, error matrix, swap fix, 100% force coverage) + per-cluster confidence in the output |

Matching-side context (validated on 39 events; do not re-tune from here):
v0.4.1 = **95.78% NS total efficiency** (energy-weighted correct fraction of
ALL truth-valid energy in TPCs ≤ 1.4 GeV, |reco−truth| ≤ 10 ticks, at 100%
coverage). The cheating study showed clustering quality (impurity, splitting)
is one of the two remaining improvement levers — hence this package.

## This release's performance

See `performance.md` — headline NS total-eff, all-TPC efficiency, coverage
100%, per-stage timing; computed over the produced event set (all events of
flow files 0000001-0000010; if the last pre-departure run was cut short,
performance.md states the exact coverage).

## Folder contents

```
README_RELEASE.md            this file
performance.md               10-event performance report
flow_files.txt               .pt -> source FLOW hdf5 mapping (deliverable #3)
pt/flow<tag>.pt              ONE .pt PER FLOW FILE (10 total, deliverable #1)
clustering_v055/             the clustering algorithm (deliverable #2)
  global_track_clustering_toolbox_v11_2.py   core algorithm (self-contained)
  vertex_merge.py                            v0.2.5 vertex-track merge (numpy-only)
  intersection_refine.py, shower_absorb.py   optional opt-in passes (reference)
  clustering.py, config.py                   NERSC-side wrappers (reference only)
  clustering_defaults.json                   ALL 81 tunable parameters, production values
recluster.py                 THE TUNING LOOP (see below)
convert_pt.py                provenance: how npz dumps became .pt
```

## `.pt` schema (torch.load(fp, weights_only=False) -> dict)

Each `pt/flow<tag>.pt` corresponds ONE-TO-ONE with the flow file of the same
tag and contains: `flow_file` (source path), `n_events`, and `events` — a dict
`event_id -> event_dict`. Each event_dict has:

- per-hit tensors, one entry per calibrated prompt hit:
  `x, y, z` [cm], `energy` [MeV], `io_group` (1-based electronics group;
  the clustering input), `tpc` (= (io_group−1)//2, charge-TPC id),
  `labels` (int32 v0.5.5 cluster id, −1 = noise), `t0` (final matched time,
  ticks), `t0_phase1` (backbone-stage time), `truth_t0` (MC truth, ticks,
  NaN = no truth info).
- `clusters`: dict label → {type: track/shower/cluster, backbone: bool,
  tpcs, e_total, e_wrong (energy placed > 10 ticks from truth), reco_t0,
  truth_t0, **confidence**: {z_min, z_own, chi2_loc, E, n_tpc} — the
  v0.5.6 per-cluster charge–light confidence components (higher z_min /
  lower chi2_loc = more trustworthy placement; blend as you like)}.
- `meta`: flow_file, event, split_index (labels < split_index are backbone
  track/shower objects), chain tag, ns_total_eff of this event.

## The tuning loop (CPU-only; needs python + numpy + scipy + scikit-learn + torch)

```bash
python recluster.py pt/flow0000001.pt --max-events 1     # quick single-event check
cp clustering_v055/clustering_defaults.json my.json      # edit parameters...
python recluster.py pt/*.pt --params my.json             # full 126-event measure
```

`recluster.py` re-runs the FULL v0.5.5 clustering from the raw hits in the
`.pt` (file-loads the toolbox directly — no package imports, no torch model)
and reports energy-weighted **purity** (per cluster: majority-truth-interaction
fraction) and **completeness** (per truth interaction: largest single-cluster
share), compared against the shipped labels. Truth interactions are defined by
grouping `truth_t0` with a 3-tick gap. Improving purity at equal completeness
(or vice versa) is the objective — the cheating study showed ~1% of big-cluster
energy is impurity (mixed interactions inside one cluster) and small-cluster
splitting drives part of the remaining matching inefficiency.

Ideas already known to matter (from the NERSC campaign):
- impure big clusters: hits of a SECOND interaction absorbed into a track —
  candidate levers: absorption radius (`absorb_*`/attach params), the
  intersection refinement pass (`intersection_refine.py`, off by default),
  split thresholds (`split_*`).
- over-splitting at 2–6 MeV halves per-cluster matching efficiency — DBSCAN
  rescue params (`rescue_*`) and `min_inliers` / `min_length_cm` control this.
- vertex merge (`vm_*` params) is precision-validated at 97% — recall is the
  open end.

## What NOT to touch from here

The matching chain (t0 assignment) is not in this package; its results are
frozen inside the `.pt` (`t0`, `clusters[..].confidence`). If a clustering
change looks good locally (purity/completeness), it must be re-validated
through the full chain on NERSC later — the working tree there is
`/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly`
(v0.4.1 frozen copy: `/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.4.1`).
