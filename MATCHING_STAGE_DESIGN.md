# Design notes: the iterative charge–light assignment stage

*(concept stage — no code yet; 2026-08-05)*

Working design for the stage that follows the frontend clustering and the
Conformer light prediction: how every cluster — backbone to tail — gets its
t0 through one continuous optimization, instead of a matrix stage that stops
early and hands the tail to a distance heuristic.

---

## 1. Motivation

The current matching flow assigns the large clusters through the
waveform-residual ("light loss") matrix process, then force-assigns the tail
by pure spatial distance and stamps ambiguous leftovers with a null tag. Two
observations motivate a redesign:

1. **The matrix process should keep propagating.** Instead of a hard
   stopping point followed by distance matching, the same optimization
   should continue into the tail with a progressively relaxed light-loss
   requirement — the tail differs from the head only in signal-to-noise,
   not in kind.
2. **Placement must be able to reconsider itself.** Sequential (greedy)
   placement cannot express swaps: after placing clusters 1, 2, 3 at t0,
   the optimizer may later find that moving cluster 2 to t1 and bringing
   cluster 5 or 6 into t0 is globally better — and when light alone cannot
   choose between 5 and 6, the charge topology can: cluster 6's entire
   family (backbone included) already lives at t0. Swaps like this must
   happen dynamically, inside the optimization, not as a post-hoc fixup.

Additionally, the stage should be able to *correct* pre-existing backbone
mistakes (the frontend's known, charge-irreducible wrong merges), because
backbones enter with well-defined — but not infallible — t0s.

## 2. The placement likelihood

For cluster *i* and candidate time slot *t* (slots = flash times ∪ backbone
t0s ∪ null), the placement posterior factorizes as

**P_i(t) ∝ L_light,i(t) · L_space,i(t | F_t) · π_i(t)**

### 2.1 Light term — a likelihood, not a loss

L_light,i(t) = exp( −ΔL_i(t) / 2σ_i² ), where ΔL_i(t) is the waveform
residual reduction from placing i at t (the quantity minimized today) and
σ_i is a **per-cluster noise scale**: waveform noise plus the light model's
own prediction uncertainty (the Conformer should output per-bin variance —
a heteroscedastic head — so σ_i comes from the model, not from tuning).

This normalization is what makes "keep the matrix running into the tail"
principled: a small cluster's predicted light sits inside the noise, its
σ_i is large, its light likelihood is honestly flat, and the posterior is
then dominated by the spatial term. The crossover from light-driven to
space-driven assignment **emerges continuously from the noise model** —
no energy threshold, no switch to a different algorithm.

### 2.2 Spatial term — coherence with the family

F_t = everything currently placed at slot t (the "family": backbone tracks,
other clusters). Three ingredients, all cheap:

1. **Drift consistency / containment.** Assigning t shifts the cluster's
   drift coordinate. If the shifted position leaves the active volume, the
   slot is (near-)vetoed — the classic x–t0 containment test.
2. **Family adjacency.** Distance from the drift-shifted cluster to the
   nearest member of F_t (fragments hug their parents). Implemented as
   attraction to the slot's occupancy field ρ_t(x), built from the
   drift-shifted positions of the slot's current members.
3. **Topological affinity.** The frontend already knows candidate parents:
   its deferred soft evidence — declined rejoin pairs, MIP-peel
   decompositions, two-track flags — provides explicit "i belongs with j"
   priors. Frontend links define the couplings; light defines the fields.

### 2.3 Prior

π_i(t): strongly peaked at the assigned slot for backbones; flat for
orphans; the null slot carries a flat floor cost (today's −10000 tag).

## 3. The soft assignment field and the iteration

Every cluster carries a **belief distribution over slots**, Q_i(t) — the
"changing array [t, p(t)|S, p(t)|L]", kept in factored form. All clusters
update **simultaneously**; nothing is hard-placed until the end.

Annealed mean-field loop:

```
initialize:  backbones -> Q peaked at assigned slot
             others    -> Q ∝ light likelihood alone (today's matrix output)
repeat with β: 0.3 -> 5 (deterministic annealing):
  1. soft residual   R = M − Σ_i Σ_t Q_i(t) · W_i(t − t)      (per channel)
  2. cavity light    for each i: remove i's own contribution from R,
                     evaluate ΔL_i(t) against the cavity residual
  3. soft families   rebuild ρ_t(x) from drift-shifted, Q-weighted members
  4. update          Q_i(t) ← softmax_t( −β [ ΔL_i(t)/σ_i²
                                             + λ · E_space,i(t)
                                             − log π_i(t) ] )
  5. anneal          β ↑ ; monitor free energy
harden:      τ_i = argmax Q_i(t); flat posteriors -> null tag
```

The **cavity step (2) is essential** — without removing a cluster's own
contribution from the residual, clusters "explain" their own light and
self-lock.

**Why this delivers the swap dynamics.** At low β, clusters 2, 5, 6 hold
fractional mass at t0 and t1 simultaneously. Under the coupled updates,
mass flows collectively — 2 drains toward t1 *while* 6 fills t0 — and the
spatial term steers the inflow to 6 rather than 5 because 6's family sits
at t0. A swap is nothing but probability mass rearranging under joint
updates; greedy lock-in never occurs because hard placement only happens
after annealing.

## 4. This is a Potts spin glass

Assignment variables τ_i over K states, external fields from light,
couplings J_ij from spatial/topological affinity: a **Potts model in a
random field**. The identification buys known machinery:

- **Collective moves for stubborn minima.** True swap deadlocks (two moves
  that only pay off together) are handled by Swendsen–Wang-style block
  moves, where the bonds defining blocks are the frontend's soft links —
  a parent and its satellites move as one unit.
- **Ambiguity detection for free.** Run the anneal from several seeds
  (cheap — it is all matrix work). Clusters whose assignments agree across
  replicas are confident; clusters that scatter are genuinely ambiguous
  and receive the null tag. This is the principled version of "null when
  the light information is ambiguous" — self-diagnosing, not thresholded.

## 5. Correcting the backbone

Backbone t0s are strong priors, not constraints:

- Early sweeps: backbones frozen (they define the initial families).
- Mid-anneal: unfreeze with a stiff prior penalty. A backbone moves only
  if light *and* family jointly overpower the prior.
- Known frontend failure modes have visible signatures here: a
  cross-interaction fusion (stitch/rejoin doppelgänger) appears as a
  spatially **bimodal family** plus unexplained residual light at a second
  slot. The split hypothesis — already available as an explicit hit
  partition from the frontend's peel / two-track decomposition — is
  proposed as a move when the free energy prefers it.

Everything the purity-first frontend deliberately deferred becomes a move
proposal in this stage; the two designs close into one loop. The
charge-irreducible wrong merges (collinear rejoin doppelgängers, wrong
stitch edges across dead regions) are exactly the debts this stage can pay.

## 6. Calibration, compute, risks

- **λ (light vs space weight)** is the one genuinely free knob. Calibrate
  on the same single-flash data the Conformer trains on: fit λ to maximize
  the likelihood of truth assignments. Never hand-tune.
- **Compute.** N ≤ 1,600 clusters (the frontend cap now does double duty),
  K ~ tens of slots, ~100 sweeps of O(N·K) batched matrix work plus
  per-slot correlations. Small, GPU-native, and it slots directly into the
  A100-resident pipeline behind the Conformer inference.
- **Risks.**
  - *Spatial self-reinforcement* ("groupthink"): a wrong early family can
    capture its neighborhood. Defenses: cavity treatment, annealing,
    bounded λ; replica disagreement flags residual cases.
  - *Degenerate slots* (two interactions closer in time than the light
    resolution): the spatial term is the only separator — by construction
    it is the right one.
  - *Likelihood-scale mismatch* between light χ² and spatial log-density:
    absorbed into the λ calibration.

## 7. First falsifiable step (before any real light machinery)

On the 126 cached events, emulate a perfect Conformer with truth light and
run only the mean-field loop with the three-term posterior. Measure:

1. how many of the frontend's known wrong merges the backbone-audit pass
   corrects (target population: the 52 surviving rejoin pairs and the ~17
   wrong stitch edges — all proven charge-irreducible);
2. how the null-tag population compares with today's distance-forced tail;
3. replica agreement rates vs truth correctness (does disagreement really
   mark the wrong ones?).

This tests the architecture's core claims — continuous tail propagation,
emergent swaps, backbone self-correction, principled nulls — with zero
dependence on the light model's maturity.

---

**One-sentence summary:** every cluster holds a belief distribution over
times; light speaks with a volume proportional to its signal-to-noise,
space speaks through the evolving families, backbones speak through
priors; annealed simultaneous updates let placements and swaps emerge
collectively; and whatever refuses to converge across replicas is honestly
tagged null.
