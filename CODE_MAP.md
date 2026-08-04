# CODE_MAP — v0.5.5 clustering internals, conventions, and improvement levers

Deep-read notes (2026-07-21, local Claude + 6 parallel reader agents over every
source file). Companion to `README_RELEASE.md`. For the visual loop use
`event_display_local.ipynb`; for numbers use `recluster.py`.

## Runtime import graph (who actually runs)

```
recluster.py ── file-loads ──> global_track_clustering_toolbox_v11_2.py   (entry point)
                                  │  _import_sibling (hardcoded filenames, private sys.modules names)
                                  ├──> global_track_clustering_toolbox.py  : _build_tpc_segments_toolbox,
                                  │      _match_segments_across_tpcs_toolbox, _fit_line_metrics, _line_distances
                                  │      └── itself file-loads global_track_clustering.py AND track_fit_ransac.py
                                  └──> global_track_clustering.py          : _assign_vertex_ids,
                                         _cluster_track_endpoints, _tpc_id_from_io
             ── file-loads ──> vertex_merge.py        (only if vertex_merge_enable)
clustering.py / config.py = NERSC-side wrappers (reference; clustering.py also file-loads v11_2 by path)
intersection_refine.py, shower_absorb.py = opt-in passes, OFF in the release defaults
```

**The three "legacy" files are NOT dead code.** v11_2 is *directory*-self-contained,
not *file*-self-contained: it pulls 5 symbols from the old toolbox and 3 from the
base module at import time. Editing either changes v11_2 behavior. The base file is
even loaded TWICE under two private module names (once by the toolbox, once by
v11_2) — monkey-patching one copy does not affect the other.

## Pipeline (v0.5.5, in execution order inside build_global_labels_toolbox)

1. **Per-TPC RANSAC segments** (`_build_tpc_segments_toolbox` → `track_fit_ransac.py`):
   greedy multi-line RANSAC (2-point seeds, SVD refit, `lam` inlier band,
   `min_inliers=35`), RSS/BIC model selection, distance/length/gap-density prunes,
   connected-component splitting (`split_radius_cm=4`), promotion of line-like
   DBSCAN leftovers to `rescue_cluster` segments.
2. **Cross-TPC stitch** (`_match_segments_across_tpcs_toolbox`): segments extended
   to the event bounding box, DOCA + angle + raw-endpoint gates, weighted score
   (endpoint .45 / angle .35 / quality .15), **mutual-best edges only** → union-find
   into global tracks.
3. **Leftover DBSCAN** (in v11_2): ONE global `DBSCAN(eps=4, min_samples=3)` over all
   non-track hits, across all TPCs at once.
4. **Track-noise absorption** (v11_2's actual addition): per-(track,TPC) finite
   cylinders, radius `1.5 × max(lam, 1.2, q90 perp)`, endpoint margin
   `max(4, radius)`; only DBSCAN-**noise** hits are candidates; winner = min
   perp/radius; tracks refit afterwards as ONE straight cross-TPC line.
5. **Endpoint vertexing** (`_cluster_track_endpoints` + `_assign_vertex_ids`):
   DBSCAN(eps=10) on track endpoints + median infinite-line DCA ≤ 4 cm (hardcoded)
   → tracks sharing a vertex merge into one label; ≥3 tracks ⇒ type `shower`.
6. **Final labels**: vertex groups → labels `0..split_index-1` (backbone), leftover
   DBSCAN blobs appended as labels `≥ split_index` (type `cluster`).
7. **vertex_merge.py (v0.2.5)**: end-line fit (8 cm window, linearity ≥ 0.75), pairs
   whose end-lines meet within 4 cm / 2.5 cm at 15–165°, crossing guard, union-find,
   groups > `vm_max_group=6` dropped entirely. See flag discrepancy below.

## Conventions (memorize these)

- Units: cm, MeV, **ticks** for all t0s. `io_group` 1-based; `tpc=(io_group-1)//2`; 70 TPCs.
- Labels: `-1` noise; `[0, split_index)` backbone (track/vertex/shower);
  `[split_index, …)` leftover blobs (`cluster`, backbone=False).
- After vertex_merge, merged-away labels remain in `label_info` with `n_hits=0` —
  don't assume backbone slots are non-empty.
- `.pt` `clusters[lab]['e_total']` is the **correctly-matched** energy only
  (`e_ok` at dump time); full cluster energy = `e_total + e_wrong`.
- recluster.py's truth = `truth_t0` grouped with a 3-tick gap (greedy 1-D chain).
  There is no MC interaction ID in the dumps.
- `torch.load(..., weights_only=False)` — plain pickled dicts, trusted source.

## Parameter truth table

- `clustering_defaults.json` = production values; they **override** the different
  defaults baked into function signatures (e.g. JSON `match_endpoint_dist_tol=40`
  vs signature 25; `rescue_min_linearity=0.88` vs 0.92). Never reason from
  signature defaults.
- recluster.py silently drops JSON keys not in the toolbox signature (typos test
  nothing!) and passes ALL keys to vertex_merge as a namespace (missing vm keys
  raise AttributeError → keep param copies key-complete).
- **Hidden hardcoded knobs not in the JSON**: `dca_max=4.0` (vertex DCA gate),
  `gap_cm=4.0`/`min_seg_size=25` (gap-density prune call site), dist_thresh
  fallback 3.0 cm, shower_absorb min 12 hits, all of intersection_refine's scoring
  weights (0.70/0.20/0.10, scales 2.2/6.0/10.0, w68 floor 0.55, r_base floor 1.2).
- ⚠ **Flag discrepancy**: README says v0.5.5 = toolbox + v0.2.5 vertex merge, but
  the shipped `clustering_defaults.json` has `vertex_merge_enable: false`.
  (Local A/B on f1 ev0: OFF → purity .9819/compl .3084; see session notes —
  compare with ON before trusting either as the production baseline.)

## Metric caveats (recluster.py evaluate())

- Purity and completeness both mask `labels<0` AND `truth<0` out of the
  **denominators** → dumping hard hits into noise inflates both. Track noise
  *energy* fraction alongside.
- Completeness credits only the largest single cluster per truth group (rewards
  under-splitting); purity rewards over-splitting; there is no joint objective.
  Baseline to beat: purity **0.9880**, completeness **0.3887** (file 1, shipped).
- MEAN is per-event unweighted; consider pooling energy across events too.
- evaluate() is O(n_clusters × n_hits); a (label × truth-group) contingency matrix
  via `np.add.at` makes 126-event sweeps much faster. Also hoist the module
  file-loads out of the per-event loop.

## Top improvement levers (curated from the full read)

1. **Energy is never used anywhere in the vanilla chain** — RANSAC votes, all
   DBSCANs, absorption, matching, vertexing are pure hit-count geometry, while the
   objective is *energy-weighted*. Cheapest wins: energy-weighted RANSAC inlier
   votes, `sample_weight=energy` in DBSCANs, energy-aware absorption gates.
2. **Global leftover DBSCAN bridges TPCs** (stage 3): blobs < 4 cm apart across a
   cathode/wall share one label → mixed-t0 clusters. Per-TPC DBSCAN (or a scaled
   TPC coordinate) + deliberate merging.
3. **Absorption favors fat tracks**: score = perp/radius lets a messy q90-inflated
   cylinder (no upper cap!) out-compete a clean MIP track for contested hits, and
   cylinders extend past endpoints by ≥4 cm into vertex regions. Cap the radius,
   use absolute perp or a per-track transverse-σ likelihood.
4. **Leftover blobs can never rejoin their track**: only DBSCAN-*noise* is
   absorbable; a 3-hit delta-ray stub forms a blob and is permanently barred.
   Add cluster-level (not hit-level) absorption.
5. **Mutual-best stitch has no fallback**: one spurious competitor edge splits a
   real track at the TPC boundary. Second pass / Hungarian over residual graph.
6. **Vertexing merges crossing tracks from different interactions**: infinite-line
   DCA + eps=10 endpoint clustering, no t0/TPC consistency check → the worst
   failure mode for charge–light matching. Same for vertex_merge (no TPC gate;
   transitive union-find chains A–B–C through both ends of B; `vm_max_group`
   discards oversized groups instead of breaking weakest edge).
7. **Completeness cliff paths**: zero cross-TPC matches ⇒ leftover DBSCAN and
   absorption are skipped entirely (single-TPC events!); gap-density filter keeps
   only the densest segment (dead regions halve tracks); `min_inliers=35` +
   `min_length_cm=30` bar short tracks, whose hits then glue onto shower blobs.
8. **RANSAC seeding drowns in showers**: uniform 2-point sampling rarely seeds a
   faint track next to a 10k-hit shower. kNN-local second point / stratified
   seeding / adaptive early-stop.
9. **RSS model selection is occupancy-dependent**: `rss_threshold=1.5e6` absolute,
   residuals unthresholded over ALL hits → K depends on event size; normalize per
   hit or trust the BIC that is already computed.
10. **Over-splitting at 2–6 MeV** (known from NERSC): levers `rescue_*`,
    `min_inliers`, `min_length_cm`, plus lever 4 above; visualize with
    `show3d_lowE()` in the local notebook.

Full per-file findings (pipelines, every gotcha, ~60 hooks): session workflow
output; regenerate anytime by re-reading the six file groups.
