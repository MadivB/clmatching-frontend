# The spatial score: how it was built, what it measures, and how to make it fast

*(v0/v1 pairwise link + v0 family score — full design record, 2026-08-06.
Companion docs: `MATCHING_STAGE_DESIGN.md` (where the score is used),
`FAMILY_SCORE_NOTES.md` (raw prior-art research), `V070_DESIGN.md` in the
chain repo (the collective formulation).)*

---

## 0. TL;DR and artifacts

Two calibrated probability scores, both trained on truth from the 126 cached
events (files 1–8 train, 9–10 test — event-level split, no leakage):

| score | question it answers | test AUC | artifact |
|---|---|---|---|
| **J_ij (pairwise link)** v0 | are clusters i, j the same interaction? | 0.649 | `link_model_v0.json` |
| **J_ij** v1 (+pointing, +triangles) | same, with context | **0.720** (track-track 0.864) | `link_model_v1.json` |
| **S_iF (family score)** v0 | does cluster i belong to family F? | **0.915** | `family_model_v0.json` |

All three are *calibrated*: a score of 0.95 means ~95% probability against
truth (measured bin-by-bin, below). That is the design contract — the scores
enter the matching posterior in log-likelihood units, additive with the
light χ², with exactly one global weight λ to calibrate between the two
worlds. Scorer code: `cluster_links.py` (pairwise) and the family harvest in
`validation/` (family features; module extraction pending v1.1).

---

## 1. Role in the v0.7 architecture

Two levels, used at different moments of the light↔spatial iteration:

- **J_ij — the bond.** Pairwise "same interaction" probability. Used to build
  the event's candidate graph, to define blocks for collective moves
  (families move together along strong bonds), and as a feature feeding S.
- **S_iF — the verdict.** Cluster-vs-family consistency. This is what the
  trade pass optimizes: for every candidate cluster and every family at a
  time slot, S_iF (with the recorded light δ_t as prior) decides trades.
  In the eventual mean-field stage, −log S_iF is the spatial energy term
  E_space,i(t) of `MATCHING_STAGE_DESIGN.md` §3.

Two levels because the questions differ: a pair can look perfect while the
family context forbids it (a through-going track touches every family it
crosses), and a family can claim a cluster no single member would (a shower
fragment 40 cm downstream touches nothing but sits dead-center in the cone).

## 2. Method doctrine

1. **Truth-supervised, honestly split.** Labels come from `truth_t0` groups
   (3-tick gap) on the 126 events; the frontier chain's own clusters are the
   objects (so the score sees exactly the fragmentation the matcher will).
   Families for training = the perfect-assignment family (all clusters
   sharing a dominant truth group). Train files 1–8, test 9–10.
2. **Calibration before discrimination.** Every model is a standardized
   logistic (deliberately simple); we verify predicted-vs-actual per bin.
   A weak-but-honest score is usable in a posterior; a strong-but-distorted
   one is poison. Upgrades (GBDT) must go through isotonic recalibration.
3. **Measure the hard slices, not the average.** The base rate of "nearby ⇒
   same interaction" is 87% at pair level; averages flatter. We always
   report the doppelgänger slices: track–track pairs, non-contact pairs,
   crossing topologies, through-shower.
4. **The frontend lesson, encoded.** Months of frontend work proved that
   geometric plausibility ≠ same interaction (the doppelgänger principle).
   The score's job is not to overrule that lesson but to *quantify* it —
  which is why its most informative coefficients turn out to be negative
   trust in exactly the doppelgänger signatures.

## 3. Level 1 — the pairwise link J_ij

### 3.1 v0: local pair geometry (10 features)

Per cluster: centroid, PCA axis + linearity (λ₁²/Σλ²), hit count, energy,
a ≤300-hit subsample (rng(0), reproducible), track/blob type. Per pair:

| feature | definition | transform |
|---|---|---|
| `d_min` | min hit-to-hit distance (subsamples, KD-tree) | raw |
| `n_contact` | # subsample hits within 3 cm | log1p |
| `adot` | \|dot(axis_i, axis_j)\| | raw |
| `lin_min`, `lin_max` | pair linearity extremes | raw |
| `e_min`, `e_max` | pair energy extremes | log10 |
| `n_min` | smaller hit count | log10 |
| `both_track` | both typed track | 0/1 |
| `d_cent` | centroid distance | raw |

Candidates: centroid pairs < 45 cm with d_min ≤ 12 cm (109,360 pairs
harvested; base rate 0.867). **Result: AUC 0.649** (distance-only 0.556).
Calibration honest (J 0.90–0.95 bin → 92.5% actual).

**Coefficient physics** (the model rediscovering the campaign's findings):
contact (+0.22) and collinearity (+0.27) build trust; `lin_max` (−0.29) and
`both_track` (−0.23) erode it — *a clean line near you is disproportionately
someone else's through-goer*.

### 3.2 v1: context (+8 features) — the decisive upgrade

1. **Pointing** (directionality across boundaries): extended DOCA of each
   cluster's axis to the partner's centroid, both directions →
   `doca_min/max`; `fwd_max` = axial fraction of the separation.
2. **TPC topology**: `diff_tpc`, `ntpc_max` (max TPCs spanned by either).
3. **Triangle corroboration** (two-pass): score all candidate pairs with
   frozen v0 → for pair (A,B), `tri_max` = max over common neighbors C of
   min(J₀(A,C), J₀(B,C)); `tri_n90` = # witnesses ≥0.9; `tri_cross_max` =
   same restricted to witnesses in a *different TPC than both* — the
   neighboring-TPC family member vouching across the boundary.

| population | v0 | **v1** |
|---|---|---|
| all pairs | 0.649 | **0.720** |
| track–track (doppelgänger set) | 0.703 | **0.864** |
| cross-TPC | 0.704 | **0.793** |
| non-contact (>3 cm) | 0.638 | **0.717** |

New high-confidence regime: J ≥ 0.99 → 98.9% actual. Strongest coefficient
in the entire model: `ntpc_max` **−0.51** — pairs involving multi-TPC
spanners are much less likely same-interaction (the through-goer, again).
Triangles work: `tri_max` +0.28, `tri_n90` +0.21; pointing works:
`doca_min` −0.18.

## 4. Level 2 — the family score S_iF

### 4.1 Features (18), exact definitions

Family summaries: member list, concatenated-subsample PCA (axis, centroid,
transverse σ, 5–95% longitudinal span), energy, TPC set, member centroids.

- **Contact** (vs K=8 nearest members): `d_min`, total `n_contact` (≤3 cm).
- **Link prior**: `j0_max`, `j0_top2` — pairwise J₀ against those members.
- **Flow** (multi-TPC directionality): energy-weighted mean of
  \|dot(axis_i, axis_member)\| over members within 30 cm.
- **Envelope**: i's transverse offset in units of family σ (`r_sigma`);
  longitudinal extension beyond the family span (`ext`, capped 3 spans).
- **Contact-≠-one-piece (PCA co-emergence)**, vs the nearest member, using
  each cluster's own 10–90% axial span:
  - `crossing` = touching (<3 cm) ∧ angle >25° ∧ contact point *interior*
    to both spans → two pieces meeting mid-body (X topology);
  - `endjoin` = touching ∧ angle <15° ∧ contact at both *ends* →
    continuation (one piece);
  - `thru_shower` = i linear (lin >0.9) ∧ family wide (σ >4 cm) ∧ i's
    contact interior to i's own span → track passing through a shower.
- **Context**: `lin_i`, `e_i`, `fam_e`, `fam_n`, `fam_sigma`, `fam_ntpc`,
  `tpc_in` (i's TPC ∈ family's TPC set), `ang_nn`.

### 4.2 Measured performance (291,775 cluster–family rows)

**Test AUC 0.9147** at base rate 0.359. Slices: crossing topologies 0.852,
through-shower 0.944, non-contact 0.903. Calibration honest to the top bin
(S ≥ 0.99 → 99.3% actual; 0.95–0.99 → 97.6%). Top coefficients: `tpc_in`
+3.21, `r_sigma` −1.48, `j0_top2` +0.93, `n_contact` +0.62.

**Two honest caveats.** (1) In this harvest, same-family clusters virtually
always share a TPC with the family (`tpc_in=0` slice has base rate ~0), so
`tpc_in` dominates; the score has not yet been stress-tested on the
fragment-in-a-fresh-TPC case — the class-pair reach fix (§5) creates that
test population. (2) The 45 cm candidate radius truncates displaced shower
fragments; both caveats are the same bug seen twice.

## 5. Prior-art upgrades adopted for v1.1 (sourced in FAMILY_SCORE_NOTES.md)

1. **Class-pair-asymmetric reach** — shower-involving candidate pairs get
   ~20× track–track reach (SPINE gates; Wire-Cell reattaches shower
   fragments to 80 cm within 15°; MicroBooNE cone 150 cm/30°). Fixes both
   §4.2 caveats.
2. **Two-tier bounded-fraction cone** replaces the σ-ellipsoid for shower
   families: apex slid along the trunk, membership = fraction of candidate
   hits inside BOTH a 26.6° cone (≥50%) and a 36.9° cone (≥75%), length
   min(7·L_fit, 126 cm) ≈ 9 X₀. This *is* the "what a shower is and how
   propagation happens" requirement, with battle-tested constants.
3. **Continuous contact topology** replaces the boolean crossing/endjoin
   flags: fractional longitudinal position of the closest-approach point
   within each object (mid/mid = crossing, end/end = continuation,
   end/mid = emission — a prong leaving a track, same family, a case the
   booleans miss), plus SPINE's sign-invariant orientation outer product.
4. **Linearity-weighted directions** (weight axes by 1 − λ₂/λ₁) in the flow
   field; sign axes toward the transversely wider end so flow becomes
   *divergent from the apex* for showers.
5. **Gap-aware cross-TPC extrapolation**: step the axis in 0.5 cm
   increments; dead-region samples don't count as misses; accept ≥10
   matches or ≥50% matched fraction.
6. **Trade-pass safety rules**: export the best-vs-runner-up margin per
   cluster; act only on dominance (Pandora uses 3×), delete the association
   on ties. Purity-first trading.
7. **Energy-kick vertex term** E·(b+0.06)/(d+10) — scores what a merge
   *implies* for the vertex, not just local geometry.

## 6. New ideas (this work — not in prior art, ranked by value/cost)

1. **Triangle corroboration** (already shipped in v1): a strong common
   neighbor — especially in a *third* TPC — binds a pair. Biggest single
   gain measured so far. Natural extension: iterate it (feed J₁ back as the
   graph weights, rescore = one belief-propagation round).
2. **Drift-interval overlap (t-aware containment, O(1) per pair).** Each
   cluster's feasible t0 interval is fixed by requiring its drift-shifted
   span to stay inside the active volume. Same interaction ⇒ intervals
   intersect. Feature = normalized interval overlap. Cheap, computed once,
   and it is the *only* spatial feature that talks directly to time — the
   natural bridge into the light stage.
3. **dQ/dx handshake at the junction.** At a contact, compare linear charge
   density on the two facing ends: a continuing particle hands over ~equal
   dQ/dx; two crossing tracks each keep their own along their own axes; a
   shower fragment shows the characteristic drop from its parent's trunk.
   Local, O(contact-neighborhood), and orthogonal to all geometry features.
4. **Signed divergence of the family flow field.** With sign-resolved axes
   (§5.4), a shower family has coherently *diverging* flow from its apex;
   a crossing track is a flow discontinuity. Feature: local divergence
   consistency of i's signed axis with the family flow at i's position.
5. **Incremental link-field grid (the speed idea).** Voxelize the event at
   ~4 cm; each family maintains its occupancy + flow + cone fields on the
   grid. S_iF evaluation = O(subsample) lookups; a trade updates only the
   moved cluster's voxels. Iterative trading becomes O(changed), not
   O(event) — the difference between an iterating algorithm and a demo.
6. **Two-tier screening.** Vectorized centroid/axis screen over ALL pairs
   (batched matrix ops, no KD-trees) → full feature extraction only inside
   the gray zone (screen score ∈ [0.05, 0.95]). Measured harvest profile
   says this cuts per-event scoring cost ~5× at zero AUC loss.
7. **Margin-lazy rescoring.** During iteration, rescore only clusters whose
   family changed or whose best-vs-runner-up margin < threshold; confident
   assignments are frozen until their family mutates.
8. **δ_t-conditioned link.** Condition J on the *light prior distance*
   |argmin δ_t(i) − argmin δ_t(j)|: two clusters whose individual light
   curves prefer far-apart times need stronger geometry to bond. Couples
   the stages without merging them.
9. **Hierarchical families.** Score composes: fragments→shower (cone
   terms), shower+tracks→vertex group (pointing/energy-kick terms). One
   machinery, two levels — matches how interactions are actually built.
10. **Replica-margin nulls.** Run the trade pass from a few seeds; clusters
    whose assignment scatters → ambiguous → −10000 tag; consistent-null +
    spatially-coherent → missing-slot candidates (slot discovery input).

## 7. Speed engineering

Measured today (per event, Python, single core): candidate graph ~870
pairs, pairwise J₁ ≈ 60 ms; family harvest ≈ 2,300 (cluster, family) rows
≈ 250 ms. Already ≪ the 3 s/event frontend budget; the plan to keep it that
way as the trade loop iterates:

1. **Precompute once per event** (summaries, subsample KD-trees, pair graph,
   J₁): amortized across all iterations.
2. **Incremental everything** (idea 5 + 7): per trade iteration, cost ∝
   clusters that moved — target <50 ms/iteration at N=1,400.
3. **Batch, don't loop**: all model evaluations are (rows × features) GEMMs;
   the co-emergence tests are vectorizable over the contact list.
4. **GPU optional, never required**: the only O(N²)-ish piece (pair screen)
   is one batched matmul — CPU is fine; if ever needed, it's ten lines of
   torch with the fast_ransac fallback pattern.
5. **Numba escape hatch** for the grid updates if Python overhead shows up
   (same playbook as the frontend acceleration; measured, not assumed).

## 8. Validation protocol and roadmap

Every version must report, on the untouched test files: overall AUC, the
four hard slices, calibration bins, and — once the trade pass exists — the
end-to-end referee (per-event t0 accuracy / NS-eff on the 13 local events).

Roadmap: **v1.1 family score** = §5 items 1–4 (+ new-ideas 2, 3) → expected
main gain on crossing (0.852) and the currently untested displaced-fragment
population; **then Task 0** (δ_t recording via the chain's own
`collect_scan_losses`), **then the trade pass** with margin rules (§5.6),
measured by the referee before any iteration is added.
