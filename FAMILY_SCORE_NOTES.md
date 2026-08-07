# Family score: prior-art research notes (Pandora / SPINE / Wire-Cell)

# Cluster→Family Spatial Consistency Score: Design Recommendations

## 1. Additional features, ranked by expected value

**(a) Replace boolean X-crossing / end-join flags with continuous contact topology.** You already detect these; prior art shows the *continuous* version is what carries the discrimination. Compute the closest point of approach between actual hits (not PCA lines): `v1` = hit in the candidate closest to the family, `v2` = hit in the family closest to the candidate. Then export the fractional longitudinal position of contact within each object, `s = (v − c)·â / L ∈ [−0.5, +0.5]`. Contact mid-body of *both* = X-crossing; near an end of both = end-join; at one's end but mid-body of the other = T/emission (same family). SPINE's 19-dim edge vector is built entirely on this: both CPA positions (6), unit displacement (3), its length (1), and the **outer product of the *normalized* displacement** (9) — a sign-invariant orientation tensor with magnitude carried separately ([SPINE `geometric.py`](https://github.com/DeepLearnPhysics/spine/blob/develop/spine/model/layer/gnn/encode/geometric.py)). Pandora's `CrossedTrackSplittingAlgorithm` encodes the same topology as a cut: contact within 2 cm but *both* endpoints farther than 2×2 cm from the other cluster ⇒ crossing, not merge, gated at cos θ < 0.966 (>15°) ([source](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArTwoDReco/LArClusterSplitting/CrossedTrackSplittingAlgorithm.cc)).

**(b) Linearity-weighted direction — the single cheapest fix to your flow-alignment term.** Multiply the PCA axis by `dirwt = 1 − λ₂/λ₁` before using it. A track contributes a full unit vector; a blob contributes ≈0, so the scorer never has to guess whether an axis is meaningful. Fix the sign physically: flip the axis to point toward the transversely wider end (`sc = Σ (x·v̂₀)·|x − (x·v̂₀)v̂₀|`; if `sc < 0`, negate) ([SPINE `cluster.py`](https://github.com/DeepLearnPhysics/lartpc_mlreco3d/blob/develop/mlreco/utils/gnn/cluster.py)).

**(c) Pointing / emission acceptance as a continuous ratio.** With `rT = |û × (p_t − p₀)|`, `rL = −û·(p_t − p₀)` from a cluster endpoint direction û:
- node: `|rL| ≤ minLong` and `rT ≤ maxT`
- emission: `rL` in the forward window and `rT² ≤ maxT² + rL² tan²θ` (cone-plus-cylinder — tolerance widens with distance)

Export `ρ = rT / √(maxT² + rL² tan²θ)` rather than a boolean. Family-level defaults (`EventSlicingTool`, the direct analogue of your problem): `minVertexLongitudinalDistance −7.5`, `maxVertexLongitudinalDistance 60`, `maxVertexTransverseDistance 10.5 cm`, `vertexAngularAllowance 9°`, `maxClosestApproach 15 cm`, `maxInterceptDistance 60 cm` ([EventSlicingTool.cc](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArThreeDReco/LArEventBuilding/EventSlicingTool.cc), [LArPointingClusterHelper.cc](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArHelpers/LArPointingClusterHelper.cc)).

**(d) Cone containment fraction — this is *not* your PCA envelope.** A σ-scaled ellipsoid cannot admit a shower fragment displaced 60 cm downstream along the propagation direction; a cone with an apex can. Add it as a separate term (§2).

**(e) Local endpoint direction and local dE/dx.** Direction estimated over a small neighborhood of the start point (radius `Rn = 5` voxels, optimized against true momentum), *not* the global PCA axis. Ablation: removing start point + start direction costs ~1% absolute edge accuracy; adding track endpoints buys ~0.2% ([arXiv:2007.01335](https://ar5iv.labs.arxiv.org/html/2007.01335)). Also add hit-energy mean and RMS — SPINE's interaction-stage node vector is 33-dim = 16 base + value(2) + shape(1) + points(6) + local dirs(6) + local dE/dx(2) ([`train_grappa_inter.cfg`](https://github.com/DeepLearnPhysics/spine/blob/develop/config/train_grappa_inter.cfg)).

**(f) Trunk/halo transverse-profile ratios** (for the track-through-shower guard). Core = cylinder of radius `0.2 R_M` around the axis; `haloTotalRatio = Q_halo/(Q_core+Q_halo)`; `concentration = Σ(E_i / max(0.01, r_i)) / Q_tot`; `conicalness = √(Q_con,end/Q_con,start) / (Q_end/Q_start)` over the first/last 20% of length ([TrackShowerIdFeatureTool.cc](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArTrackShowerId/TrackShowerIdFeatureTool.cc)). Note the constant conflict: Pandora hard-codes `R_M = 10.1 cm`; the standard LAr values are `X₀ = 14 cm`, `R_M = 7.2 cm`. Pick one deliberately.

**(g) Class-pair-asymmetric reach.** Do not use one distance ceiling. SPINE's shipped interaction config gates the complete graph with an upper-triangular matrix over [shower, track, michel, delta]: `[500, 500, 0, 0, 25, 25, 25, 0, 0, 0]` — shower-involving pairs get **~20× the reach** of track–track, and several pairs are forbidden outright (0 = no edge). **Units are voxels**, not cm. Independent corroboration of the loose end: Wire-Cell reattaches displaced shower fragments out to **80 cm within 15°** of the shower direction ([arXiv:2110.13961](https://ar5iv.labs.arxiv.org/html/2110.13961)); MicroBooNE's cone is 150 cm / 30° ([arXiv:1910.02166](https://ar5iv.labs.arxiv.org/html/1910.02166)).

**(h) Gap-aware extrapolation for cross-TPC families.** Step along the fit in 0.5 cm increments; points falling inside a known gap are skipped and do **not** count as misses; a sample matches within 1.5 cm; abort after 8 consecutive non-gap misses; accept at ≥10 matched points **or** ≥50% matched fraction. The complementary pointing variant requires ≥50% of the span be genuine gap plus *bidirectional* emission confirmation at 10° ([CrossGapsAssociationAlgorithm.cc](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArTwoDReco/LArClusterAssociation/CrossGapsAssociationAlgorithm.cc)).

**(i) A family-level (not pairwise) vertex-consistency term.** Pandora's `ShowerGrowingAlgorithm` scores a merge by `FoM = nVertexAssociatedSeeds − nVertexAssociatedNonSeeds` — i.e. by the vertex topology the merge *implies*, not by local geometry alone. The closed-form cluster→vertex term is the energy kick `E' = E·(b + 0.06)/(d + 10)` with `b` the impact parameter and `d` the displacement, normalized by total energy ([EnergyKickFeatureTool.cc](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArVertex/EnergyKickFeatureTool.cc)).

**(j) Best-vs-runner-up margin.** Export `Δ = s_best − s_second` per cluster. Pandora's `ClearShowersTool` requires the winner beat competitors by a factor 3; `BranchGrowingAlgorithm` **deletes** the association on an exact tie rather than assigning arbitrarily.

## 2. Metric formulations worth adopting verbatim

**Two-tier bounded-fraction cone (never a single opening angle).** For each candidate hit: `rL = d·û`, `rT = |d × û|`; bounded iff `0 ≤ rL ≤ L` and `rL·tan(θ/2) > rT`. Score = bounded *fraction* of candidate hits. Require **both** tiers: `tan = 0.5` (26.6°) with fraction ≥0.5 **and** `tan = 0.75` (36.9°) with fraction ≥0.75; `L = min(7 × L_fit, 126 cm)` (= 9 X₀). Tie-break on mean `rT`. Cones sampled at 5 apex positions in both directions from a sliding fit ([LArThreeDSlidingConeFitResult.cc](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArObjects/LArThreeDSlidingConeFitResult.cc)).

**Data-derived half-angle** where the shower is well populated: 80–85th centile of per-hit opening angles, capped at `cos = 0.95` (~18°), then demand a multi-projection consensus of containment fractions — mean >0.6, max >0.7, min >0.3 ([VertexBasedPfoMopUpAlgorithm.cc](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArThreeDReco/LArPfoMopUp/VertexBasedPfoMopUpAlgorithm.cc)).

**Asymmetric bar for absorbing tracks.** A track-like candidate joining a shower family must clear a strictly higher threshold: 3-of-3 consistent directions vs 2-of-3 for shower-like (`m_minConsistentDirectionsTrack(3)` vs `m_minConsistentDirections(2)`). This *is* your track-through-shower guard, expressed as a graded requirement rather than a veto flag.

**Upgrade the σ-unit envelope to a proper association χ².** The Bar-Shalom track-to-track test forms the state difference weighted by the combined covariance **including cross-covariance**, then tests the quadratic form against χ² with the right dof. A naive Mahalanobis / σ-unit distance is biased because the two estimates share process noise ([review](https://www.sciencedirect.com/science/article/abs/pii/S0957417424014076)). This is the principled version of "transverse sigma-units."

**Direction-sign-free polyline distance.** For cluster-vs-family (rather than cluster-vs-cluster) matching, `MDF = min(d_direct, d_flipped)` over equal-length resampled polylines solves PCA sign ambiguity for free and costs `O(n_clusters × n_families)` ([QuickBundles](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3518823/)).

**Score → partition.** Do **not** threshold + connected-components. Sweep edges in decreasing score order, running union-find, and accept an edge only if it *decreases* a global partition loss; stop below 0.5 ([arXiv:2007.01335](https://ar5iv.labs.arxiv.org/html/2007.01335)). Loss is plain BCE on edge scores.

**Cut-based fallback discriminant** (5 dimensionless ratios, all computable from what you have): TRACK immediately if straight-line length >80 cm; else shower if `pathLength/straightLine > 1.005`, or `(rT_max − rT_min)/straightLine > 0.05`, or `vertexDist/straightLine > 0.5`, or `showerFitWidth/straightLine > 0.35`; minimum 6 hits ([CutClusterCharacterisationAlgorithm.cc](https://raw.githubusercontent.com/PandoraPFA/LArContent/master/larpandoracontent/LArTrackShowerId/CutClusterCharacterisationAlgorithm.cc)).

## 3. Performance levels to benchmark against

| System | Task | Result |
|---|---|---|
| GrapPA standalone ([2007.01335](https://arxiv.org/abs/2007.01335)) | interaction grouping @ ~1 int/m³ | **mean ARI 99.2%**, purity & eff >99%, edge acc >99% |
| Full chain @ DUNE-ND density ([2102.01033](https://ar5iv.labs.arxiv.org/html/2102.01033)) | interaction grouping | **purity 99.1%, eff 99.1%, ARI 98.2%** |
| same | shower-fragment grouping | purity 98.7%, eff 99.3%, ARI 96.9%, primary ID 99.5% |
| same | voxel-level (upstream) clustering | eff & purity 97.5%, ARI 96.1% |
| Pandora / MicroBooNE ([1708.03135](https://ar5iv.labs.arxiv.org/html/1708.03135)) | CCQE μ+p | μ 95.8%, p 87.3%, **whole event 86.0%** |
| same | CC RES μ+p+π⁰ | leading γ 88.0%, **sub-leading γ 66.4%, whole event 49.9%** |
| Wire-Cell ([2110.13961](https://ar5iv.labs.arxiv.org/html/2110.13961)) | ν vertex within 1 cm | 72.9% νμCC / 65.8% νeCC / 52.1% NC |
| NuGraph2 ([2403.11872](https://arxiv.org/abs/2403.11872)) | hit → primary interaction | 98.0% eff |

**Report at the published thresholds so your numbers are comparable:** truth match requires ≥5 shared hits, purity ≥50%, completeness ≥10% (the 10% floor exists specifically to reject fragment matches). Use mean ARI as the family-level figure of merit, and report purity/efficiency per *role* (primary track, leading EM, sub-leading EM) — not just event-level.

## 4. Pitfalls

1. **The dominant failure mode is one confident wrong edge, not many weak ones.** A through-going cosmic crossing and overlapping a vertex track produced a documented **56.1%-purity** event. This is why the partition-score guard (§2) matters more than better calibration.
2. **Do not ask one score to both split contact and decide association.** SPINE handles crossing-track separation *upstream* at voxel level (Graph-SPICE), and the grouping GNN only decides membership of already-separated pieces. Keep your interior-X-crossing handling as a separate stage from the family score.
3. **Do not pre-prune the candidate graph geometrically.** The complete graph beat Delaunay, MST and 5NN; kNN and MST are reported to *fail*. Gate only by the class-pair max-length matrix.
4. **Do not reach for a learned encoder.** A sparse-CNN encoder over raw voxels "quickly and dramatically overfits" alone and gave "no measurable improvement" added to the 16 geometric features. Your hand-harvested feature set is the right architecture.
5. **A single opening-angle cut for shower membership is the classic mistake** — every mature implementation uses a bounded *fraction*, at two tiers and/or across independent projections.
6. **Vertex association must act as a veto, not only a bonus:** if the family is vertex-associated, other vertex-associated objects must be *excluded* from absorption, or a shower eats a sibling primary.
7. **Refuse to assign on ties.** Leave ambiguous clusters unmerged; propagate association strength along chains as `min` over the chain (weakest-link) rather than assuming transitivity.
8. **Unit and constant traps:** SPINE's `max_length` is in voxels (≈150 cm / 7.5 cm at 0.3 cm pitch, not 500/25 cm); Pandora's `R_M = 10.1 cm` conflicts with the standard LAr 7.2 cm; many Pandora cuts (`LongitudinalAssociation`, cone tan 0.2) are **2D per-view** values and are much tighter than the 3D family-level ones — do not mix the two tables.
9. **Low-multiplicity clusters break every σ-normalized quantity.** Enforce hit floors (Pandora: ≥20 hits per 3D cluster, ≥50 to seed a new slice; ≥5–6 hits for any track/shower score).
10. **The fixed-length feature vector is the acknowledged information bottleneck** — SPINE's own authors say so. Expect residual loss concentrated on large showers whose geometry a 33-number summary cannot represent, and on sub-leading EM objects generally.
11. **Wire-Cell's 80 cm / 15° reach is an upper bound, not a default.** Adopting it without class-pair gating will pull cosmic fragments into families; Pandora's mop-ups are far tighter for exactly this reason.