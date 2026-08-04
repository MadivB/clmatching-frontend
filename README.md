# clmatching-frontend

Purity-first front-end charge clustering for DUNE ND-LAr charge–light
matching (v0.5.6 development line).

The front end turns calibrated prompt charge hits into compact, high-purity
clusters that a data-driven light matcher can timestamp: purity is spent
nowhere (a wrong hard merge forces one t0 onto two interactions and is
irreversible), while completeness is deliberately delegated to the light
stage. Every design decision below follows from that asymmetry.

## Results (126 cached MiniProdN5 events, energy-weighted vs truth t0 groups)

| chain | purity | completeness | clusters/event |
|---|---|---|---|
| production v0.5.5 | 0.9857 | 0.3763 | — |
| **this frontier chain** | **0.9901** | **0.3875** | **1429** |

Track-label purity: 0.9931. Runtime: 7.9 s/event (GPU) / 11.8 s (CPU) on a
median 164k-hit event, down from 23.9 s — bit-identical output.

## Chain

```
per-TPC RANSAC segments  (fast_ransac.py: GPU/CPU accelerated, bit-exact)
  -> segment splitter S1            (segment_split.py)
  -> directional cross-TPC stitch   (stitch_directional.py)   dot <= -0.97, trans <= 5 cm
  -> leftover DBSCAN + absorption   (clustering_v055 toolbox)
  -> endpoint rejoin                (endpoint_rejoin.py)      endpoint-exclusive, bridge evidence
  -> vertex pinpointing v0.4        (vertex_pinpoint.py)      eps 5 / interior 1.5 / boundary 0.75
  -> targeted blob refine           (blob_refine.py)          fine DBSCAN inside big blobs, zero demotion
  -> two-track donation split       (two_track_split.py)      crosser-continuity gated
  -> fragment absorption            (frag_absorb.py)          <=10-hit labels join nearest within 3 cm
```

The full configuration is `params_full.json`; `recluster.py` runs and
evaluates the chain end to end:

```bash
python recluster.py pt/flow0000001.pt --params params_full.json --max-events 1
python recluster.py pt/*.pt --params params_full.json     # all events
```

`mip_peel.py` is a validated but OFF-by-default surgery (peels MIP cores out
of dE/dx-flagged windows; measured 2-3% foreign-energy precision — same-
interaction companions dominate, so it is kept as a soft option).

## Acceleration

`fast_ransac.py` (enabled via `fast_ransac_enable`) replaces the two profiled
hot spots (85% of runtime): the radius-graph union-find (now scipy csgraph)
and the RANSAC hypothesis scan (GPU-assisted argmax with the winning inlier
mask always replayed on CPU through the legacy expression — output stays
bit-identical; verified on all 126 events). Backend `auto` falls back to CPU
permanently on any GPU problem (no device, busy, CUDA OOM) with a single
warning — safe on shared/interactive NERSC nodes. `ACCELERATION_NOTES.md`
holds the measured profile and the Perlmutter deployment plan for the 2M-event
campaign.

## Repository layout

- `clustering_v055/` — frozen production v0.5.5 clustering + toolbox (baseline)
- `*.py` (top level) — the new chain stages listed above
- `params_full.json` — the adopted chain configuration
- `event_display_local.ipynb` — local performance window (truth / shipped /
  tuned panels, gallery tools); rebuilt by `validation/build_local_notebook.py`
- `validation/` — the verification harness: per-stage bit-identity checks
  (`verify_*.py`), the stage profiler, and the 126-event worker scripts used
  to validate every adopted stage (purity budget, DBSCAN/refine/absorption
  scans, rejoin/split audits, fast-RANSAC sweep)
- `CODE_MAP.md` — where each tuning lever lives
- `ACCELERATION_NOTES.md` — profile + NERSC scaling strategy

Data (`pt/` caches, FLOW hdf5) and generated galleries (`plots/`) are not
committed; drop the 10 cached `.pt` files into `pt/` to reproduce every
number above.
