# Acceleration notes for the v0.5.6 frontier clustering chain
(2026-07-29 — measured profile + researched strategy; chain frozen at
P 0.9901 / C 0.3875 / N 1429, params_full.json)

## Measured stage profile (this machine, pure Python)

| stage | f7 ev9 (164k hits) | f1 ev0 (244k hits) | share |
|---|---|---|---|
| RANSAC line extraction | 11.80 s | 25.33 s | **81-83%** |
| directional stitch (pair loop) | 1.21 s | 1.80 s | 6-8% |
| leftover DBSCAN + absorption | 0.33 s | 0.66 s | 2% |
| endpoint rejoin | 0.49 s | 0.86 s | 3% |
| vertex pinpoint | 0.23 s | 0.44 s | 1.5% |
| blob refine | 0.03 s | 0.15 s | <1% |
| two-track split | 0.41 s | 0.93 s | 3% |
| fragment absorption | 0.11 s | 0.21 s | <1% |
| **TOTAL** | **14.6 s** | **30.4 s** | |

cProfile hot spots (f7 ev9): `ransac_line_3d` 682 calls / 7.5 s (pure-Python
trial loop at ~11 ms/call); `_components_by_radius` 4.6 s, of which the
hand-written union-find is 3.7 s across 7.0M Python-level find/union calls;
stitcher `_seg_seg_dist` 54,419 calls / 1.6 s. All interpreter-bound, not
FLOP-bound (~1e9 FLOP/event is trivial).

## Throughput math (Perlmutter)

2M events x 3 s = 1,667 core-hours = ~13 h on ONE 128-core CPU node.
Even at today's 30 s/event: ~170 node-hours. Allocation is a non-issue;
the constraints are per-event latency and GPU residency for the CNN stage.

## Immediate implication

Vectorizing/JIT-compiling exactly three hot spots (batched-hypothesis RANSAC,
union-find -> scipy.sparse.csgraph.connected_components, vectorized stitch
prefilter) removes ~90% of the runtime: expected ~2-4 s/event in Python on
this laptop, at or under 3 s on Perlmutter Milan cores - the target is
reachable WITHOUT leaving Python. GPU port is the strategic track for CNN
co-residency, not a requirement for the 3 s budget.

---

# Memo: Accelerating the ND-LAr front-end clustering chain to ≤3 s/event for 2M events on Perlmutter

## 0. The one strategic fact that reorders everything

At the 3 s/event target, the entire 2M-event campaign is **~1,667 core-hours ≈ 13–20 CPU-node-hours** on Perlmutter. Even a 30 s/event partially-optimized chain is only ~170 node-hours. Compute allocation is a non-issue; the problem is (a) per-event wall time and (b) getting cluster tensors GPU-resident for the downstream CNN. Also non-negotiable: **bit-exactness against the frozen CPU reference is unattainable on GPU** (PyTorch explicitly disclaims CPU↔GPU reproducibility; cuML DBSCAN provably assigns border points differently than sklearn, issue rapidsai/cuml#2206; a GPU RANSAC cannot replay sklearn's RNG stream). Any plan promising it is not credible. Start the validation-criterion renegotiation **now**, in parallel with all engineering — it gates option 3 and is the longest-lead item that requires no code.

---

## 1. Ranked strategy

### First: Deployment engineering + validation renegotiation (~1 week, near-zero risk)
Make the campaign runnable *today* with the frozen Python chain, and establish the harness structure everything else will validate against.
- Batch events into ~1,000-event HDF5 files (~2,000 file-tasks). Never per-event files or per-event Slurm tasks (TaskFarmer dispatches 5–10 tasks/s; Lustre punishes small files; keep <1,000 files/dir, shard output as `out/000/…`).
- One sbatch job, `srun parallel --jobs 128` per CPU node, `--joblog --resume-failed` for free file-granularity checkpointing. No `--sshlogin`.
- Ship the frozen numpy/scipy/sklearn stack as a **Shifter** (or podman-hpc) image — kills the import storm at 10+ nodes *and* pins library versions, which is itself a validation requirement.
- `OMP_NUM_THREADS=1`, spawn not fork, `os.sched_getaffinity`, `srun -n 1 --cpu-bind=none`. Single-threaded workers also remove BLAS-reduction-order nondeterminism.
- Run production under **preempt QOS** (0.5× CPU / 0.25× GPU charge; safe because `--resume-failed` restarts at file granularity). Use shared QOS for dev.
- **Speedup: none per-event, but the campaign becomes executable and nearly free at any per-event cost. Do this regardless of everything below.**

### Second: CPU vectorization of the Python-loop hot spots (2–4 weeks, low risk)
Minutes/event in pure Python almost certainly means the cost is *interpreter loops*, not FLOPs (~10⁹ FLOP/event is trivial). Vectorize in numpy before touching CUDA:
- Replace per-TPC iterative `sklearn.RANSACRegressor` with a numpy batched-hypothesis scorer: draw all 2-point samples up front, compute the (B, N) point-to-line distance matrix `‖(p−a)×d‖`, integer inlier counts, argmax. This is the same math sklearn does, minus per-iteration Python overhead.
- Vectorize the O(n²) segment stitching: with 200–600 segments, all pair criteria fit in dense (S, S) matrices; threshold → edge list → `scipy.sparse.csgraph.connected_components`.
- Batch the small SVDs: stack segments (bucketed by size or padded) and use `numpy.linalg.svd` on a 3D array — one gufunc call instead of hundreds of Python calls.
- KD-tree stages: `scipy.spatial.cKDTree` with `workers=1`, batched `query_ball_point` calls; usually already fast enough once the loops around them are vectorized.
- **Realistic speedup: 10–50×** → likely ~5–15 s/event, possibly under 3 s. **Validation: the easiest option** — same hardware, same libraries; if you replicate sklearn's exact sampling sequence, RANSAC can be bit-exact; vectorized reductions may perturb the last ulp elsewhere, so expect "physics-equivalent, mostly bit-exact" and confirm with the existing harness.

### Third: GPU port of the full chain (6–10 weeks, the strategic move)
Do this even if step 2 hits 3 s/event, because the *real* payoff is keeping per-cluster hit tensors on the A100 for the CNN stage — no PCIe hop, no second I/O pass. Stack: **PyTorch ≥2.4 (CUDA 12), CuPy 13.x, RAPIDS 25.x (cuML/cuGraph), FRNN (lxxue, pin a commit), optionally PyTorch3D 0.7.x**. Batch across all ~70 TPCs and across hypotheses in single kernel launches — a naive per-TPC CuPy port would be launch-bound (2–10 µs/launch × thousands of tiny kernels/event ≈ tens of ms of pure launch overhead, plus Python dispatch).
- **Realistic speedup: <0.5–1 s/event of GPU time**, dominated by residual Python orchestration, not compute (Allen and the STCF Hough paper do whole-event tracking in ≤1 ms/event on lesser GPUs; your budget is 3,000 ms).
- **Cost:** the RANSAC replacement is a ~200-line custom implementation (fork the `torch-ransac3d` v2.0.0 `iterations_per_batch` pattern — MIT, pip-installable — it fits one line per call, so multi-line extract-remove looping and cross-TPC batching are yours to write); the rest is library integration plus the renegotiated validation harness.
- **Do not adopt:** Open3D (its Tensor-API DBSCAN/RANSAC are CPU wrappers that copy off-GPU), cuSpatial (2D-geospatial only), torch_cluster radius ops (documented CPU/GPU divergence and pathological GPU slowdowns), MinkowskiEngine (unmaintained, CUDA-12 build failures).

---

## 2. Per-stage mapping to tools

| Stage | Best tool (GPU) | Notes |
|---|---|---|
| **RANSAC line extraction** (per-TPC, iterative) | Custom batched PyTorch, forked from `torch-ransac3d` 2.0.0 pattern | One (TPC, hypothesis, point) tensor: ~5k hits × 1,024 hypotheses × 70 TPCs ≈ 40 MB/round in fp64 — all TPCs scored in one launch. Iterate extract-remove synchronously across TPCs, masking finished ones. Seeded `torch.Generator`, integer inlier counts (order-independent), lowest-index argmax tie-break, fp64 refits (A100: 9.7 TFLOPS FP64). Deterministic-by-construction alternative if RANSAC validation stalls: iterative 3D Hough (Dalitz, IPOL 2017 — same extract-and-remove shape; integer scatter-add voting; STCF showed 151× as pure matrix ops, arXiv 2607.04067). |
| **O(n²) segment stitching** | Batched pair-criteria tensors → edge list → `cugraph.weakly_connected_components` | 600² pairs is trivial as dense tensors. Result depends only on the edge set → deterministic; canonicalize labels by min hit index. CPU equivalent: `scipy.sparse.csgraph`. |
| **Global DBSCAN** (~100k pts leftover pass) | `cuml.cluster.DBSCAN`, single-GPU, `algorithm='rbc'` or `'brute'`, tune `max_mbytes_per_batch` | 10–50× claimed. Border-point assignments **will** differ from sklearn (#2206) — but sklearn itself is input-order-dependent for ties, so your equivalence harness should already tolerate this. Avoid multi-GPU DBSCAN (open inconsistency issue #7341). Zero-copy in via `__cuda_array_interface__`, out via `output_type='cupy'` + `torch.from_dlpack` — **same CUDA stream at every DLPack handoff**, or sync. |
| **KD-tree absorption / rejoin stages** | **FRNN** (lxxue) — Hoetzlein counting-sort grid, O(kn), grid *cached* across the several stages that query the same hit cloud | Alternative: PyTorch3D `ball_query` with ragged `lengths` batching across all 70 TPCs in one launch. Avoid cuVS eps-neighborhood (C++-only, dense boolean output = 10 GB at 100k×100k unless tiled) and torch_cluster. |
| **Small-SVD stages** (segment PCA/refits) | `torch.linalg.svd` on stacked (S, n, 3) batches, fp64 | Bucket segments by size or pad. One launch replaces hundreds. Batched SVD is deterministic run-to-run under `torch.use_deterministic_algorithms(True)`. |

---

## 3. Determinism / validation caveats

- **CPU vectorization (option 2):** bit-exact achievable for RANSAC if you replay sklearn's RNG sampling order; elsewhere expect last-ulp drift from changed reduction order. Validate with the existing harness; single-threaded BLAS (`OMP_NUM_THREADS=1`) is mandatory.
- **GPU port (option 3):** renegotiate to a two-tier criterion: (a) **physics/membership equivalence** vs the frozen CPU reference (cluster-membership agreement, line-parameter tolerances, downstream-metric agreement), then (b) **freeze a GPU reference** and require run-to-run bit-exactness on fixed hardware + fixed container. Achievable via: seeded generators, integer inlier counting, fixed argmax tie-breaks, `torch.use_deterministic_algorithms(True)` (note `index_add_`/`scatter_add_`/`bincount` use atomicAdd — route around them or accept the deterministic-variant slowdown), fp64 refits, pinned versions in the container.
- **cuML DBSCAN:** core-point memberships match sklearn; border-point ties and label numbering don't. Make the harness label-permutation-invariant and border-tie-tolerant.
- **Deployment layer:** GNU parallel retries are idempotent if each task rewrites its whole output file; MPS does not alter within-process arithmetic — determinism survives the entire recommended deployment. Only *new GPU kernels* need care.
- **Hough fallback:** integer voting is order-independent → deterministic in value with no seeds at all; keep it in your back pocket if RANSAC-equivalence review drags.

---

## 4. NERSC deployment shape for 2M events

**Phase A — CPU farm (available immediately, options 1–2):**
- ~2,000 × 1,000-event HDF5 files on `$SCRATCH` (all-flash Lustre; default 1-OST stripe is correct for ~1 GB single-writer files; mind the 8-week purge).
- **10 CPU nodes × ~2 h** (at 3 s/event; scale walltime linearly if slower), one sbatch job, preempt QOS. Per node: `srun parallel --jobs 128 process_file.sh :::: filelist.$SLURM_NODEID`, 128 single-core workers, 4 GB RAM each.
- Stage each worker's input file through `/dev/shm` (nodes are diskless — there is no local `/tmp`; 512 GB CPU-node RAM leaves ~256 GB shm headroom), write one output file back to sharded `$SCRATCH` dirs. `HDF5_USE_FILE_LOCKING=FALSE` if reading from CFS.

**Phase B — GPU farm (after the port; also runs the CNN stage in-process):**
- GPU nodes: 1× EPYC 7763 (64c) + 4× A100-40GB + 256 GB. **8–16 event workers per GPU under MPS** (start `nvidia-cuda-mps-control -d` in the batch script; up to 48 clients/GPU; tune `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`; MIG is not user-offered on Perlmutter). That's 32–64 workers/node, matching the 64 host cores at 16 cores/GPU.
- At ~0.5 GPU-s/event: ~278 A100-hours ≈ **70 GPU-node-hours**, e.g. 16 nodes × ~4.5 h, preempt QOS (0.25× charge). Budget /dev/shm carefully — only 256 GB total on GPU nodes.
- Same file-task granularity and `--joblog --resume-failed` restartability. Each worker keeps its event's cluster tensors on-GPU and calls the CNN before writing results — one pass, no intermediate cluster files.
- Tune the MPS worker count in shared QOS (1 GPU + 16 cores) before the production run.

---

## 5. Defer

- **Deterministic Hough rewrite** — hold as fallback, don't build speculatively; batched RANSAC meets the budget with 100× headroom.
- **CUDA Graphs / kernel fusion / custom CUDA kernels** — only if profiling shows launch-bound behavior *after* cross-TPC batching; at 3 s/event you won't need them.
- **cuVS/RAFT integration** — FRNN + cuML cover the neighbor needs; revisit only if FRNN maintenance becomes a problem (note RAFT neighbors froze at RAPIDS 24.06; cuVS is the maintained home).
- **Multi-GPU anything** — one event per worker, one GPU per worker-group; no multi-GPU DBSCAN (open correctness issue).
- **SPINE migration** — wrong scope (it replaces the algorithm, and the chain is frozen), but benchmark against it once for physics-performance context, and crib its containerized ND-LAr A100 stack and Perlmutter I/O patterns.
- **Sparse-CNN framework choice** (spconv v2 vs TorchSparse++ — 2.9× over MinkowskiEngine on A100; not MinkowskiEngine) — belongs to the CNN stage's owner; flag it, don't solve it here.
- **Allen/traccc code reuse** — templates and lessons only (local seeding, vecmem patterns); silicon-geometry-bound, not drop-ins.

**Bottom line:** ship the CPU farm this week (option 1), vectorize the Python loops next (option 2, likely lands near the 3 s target alone), and run the GPU port (option 3) as the strategic track whose acceptance criterion — physics equivalence plus a re-frozen GPU reference — you start negotiating today, because that renegotiation, not the code, is the critical path.