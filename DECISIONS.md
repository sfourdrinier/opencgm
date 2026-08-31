# DECISIONS

Every choice the paper does not specify, recorded **when made**, with rationale.

Two reasons this file is load-bearing:

1. **Scientific honesty.** Blueprint §2 — a guess must never silently become "the GlucoFM
   recipe". Published results must trace every ambiguous choice back to a dated entry here.
2. **Agent coordination.** Parallel agents hitting the same unspecified fork will each resolve
   it differently and none will say so. A fork must be decided here *before* fan-out.

Statuses: **OPEN** (must be decided before dependent work) · **DECIDED** · **SUPERSEDED**.
Evidence labels follow blueprint §2.

`opencgm config-check <cfg>` lists every `INFERRED_RECONSTRUCTION` in a config. Each one
should have an entry here.

---

## D001 — Raw data lives outside the repository · DECIDED

`PROPOSED_EXTENSION` (project infrastructure, not method)

Raw sources live at `/media/DataSets/bloodglucose/glucofm/raw`, symlinked to `data/raw` and
gitignored. Canonical Parquet goes to `~/localDataSets/cache/glucofm` (SSD), symlinked to
`data/canonical`.

**Why:** ~40 GB of sources, several under licences that forbid redistribution. Blueprint §7
requires that raw data never be committed. Follows the workspace's inventory/hot-cache split.

---

## D002 — Restrictive licences make a source evaluation-only, not excluded · DECIDED

`SOURCE_VERIFIED` · full reasoning in `bundle/BLUEPRINT_AMENDMENTS.md` A4

NC/ND/SA sources go to **Lane E**: usable for evaluation, never inside a distributed
checkpoint.

**Why:** CC 4.0 §2(a)(1)(B) grants the right to "produce and reproduce, but not Share, Adapted
Material for NonCommercial purposes". ND restricts *distribution* of a derivative, not its
creation; evaluation does not even produce adapted material. Treating these sources as wholly
unusable discards validation data we are entitled to use.

**Bonus that justifies the effort:** a cohort we are forbidden to train on has zero leakage
risk by construction — a stronger generalization claim than any in-corpus held-out split.

**Boundary:** never in a distributed checkpoint; NC additionally bars commercial use,
including a privately-trained model shipped in a paid product.

Affects: `uchtt1dm`, `cgmacros`, `glucofm_bench`, and `d1namo` pending SA review.

---

## D003 — Stanford CGM comes from the companion repo, pinned by commit · DECIDED

`SOURCE_VERIFIED` · `bundle/BLUEPRINT_AMENDMENTS.md` A2

Use `filtered_cgm_03222026.csv` from `aametwally/Metabolic_Subphenotype_Predictor` at commit
`94d647944c001dfa34492fd593b6e4804c1f45c4`. **Not** `cgmdb.stanford.edu/data/data_cgm.csv`.

**Why:** the official page's file is meal-challenge response curves with spline-smoothed float
glucose — the exact artefact blueprint §6.5 rule 4 forbids in the strict pipeline. The repo
file is integer glucose at true 5-minute cadence, and reconciles at 56 subjects = the paper's
19 pretrain + 37 downstream exactly. Pinned by SHA because `main` can move.

Keep `data_meta.csv` from the cgmdb page for downstream labels (HbA1c, SSPG, DI).

---

## D004 — Grid binning rule: **nearest, for all five datasets** · DECIDED

`INFERRED_RECONSTRUCTION` · blueprint §9.3, config `alignment.binning`
Evidence: `reports/binning_audit.md`. Tie rule: round half **away from zero**, never
banker's rounding (§9.3 warns about this; `test_nearest_is_not_bankers_rounding` guards it).

The paper states most sources use floor and some use nearest, but never publishes the mapping.
We implemented both and measured them on the real source timestamps.

| dataset | rule | readings kept | collisions | mean \|error\| | max |
|---|---|---:|---:|---:|---:|
| big_ideas | floor | 36,271 | 627 (1.70%) | **4.50 min** | 5.0 |
| big_ideas | **nearest** | **36,898** | **0** | **0.10 min** | 0.4 |
| hall | floor | 104,620 | 797 (0.76%) | 4.45 min | 5.0 |
| hall | **nearest** | **105,416** | 1 | **0.12 min** | 0.8 |
| stanford | floor | 442,733 | 42,482 (8.76%) | 2.97 min | 5.0 |
| stanford | **nearest** | **444,006** | 41,209 (8.49%) | **0.12 min** | 2.5 |
| colas | floor | 114,252 | 1 | 0.15 min | 5.0 |
| colas | **nearest** | **114,253** | **0** | **0.08 min** | 2.5 |
| shanghai | either | 112,287 | 0 | 0.00 min | 0.0 |

**Decision: nearest everywhere.** It is never worse on any metric on any dataset, and on the
5-minute Dexcom-family sources it is decisively better: floor discards 1.70% of BIG IDEAs
readings to grid collisions and imposes a systematic ~4.5-minute phase error, because those
devices sample at ~300 s with drift, so offsets accumulate until floor pushes each reading a
whole step early. Since the model consumes absolute circadian phase as an input, a
4.5-minute systematic shift is not cosmetic.

For Shanghai the choice is provably irrelevant — its 15-minute readings land exactly on the
minute, so both rules give identical indices and zero error. Adopting nearest therefore costs
nothing there.

**Tension with the paper, recorded rather than hidden.** Our measurement says nearest
dominates, which does not match "floor for most datasets". The most likely explanation is that
the authors compute the grid offset from a different reference point than the segment's first
reading — for example a window start pre-snapped to a 5-minute boundary, under which floor
behaves like nearest. We cannot determine this from the publication. Blueprint §9.2 defines
`uᵢ = (tᵢ - t₁)/5` relative to the first reading, and that is what we measured.

Floor is retained as Tier-2 ablation T2.1 (§20). Escalated as authors' question §27 #4.

**Follow-up:** Stanford shows ~8.5% collisions under *both* rules, which no binning rule can
explain. That indicates genuine duplicate or overlapping-session timestamps in the source and
must be characterised in PR 2 before it is averaged away. Tracked as Q5.

---

## D005 — Coverage ratio = **fraction of legal starts** (Candidate A) · DECIDED

`INFERRED_RECONSTRUCTION` · blueprint §9.4 · evidence `reports/window_sampler.md`

The paper samples overlapping 24-hour windows with a per-segment "coverage ratio" of 20–80%
but never defines the term. Both readings were implemented and measured on the strict corpus
(522 segments, 694,625 legal starts):

| Candidate | Windows | Mean/segment | Fraction of legal starts |
|---|---:|---:|---:|
| **A — fraction of legal starts** | **353,127** | 676.5 | 50.8% |
| B — union of covered timeline | 2,618 | 5.0 | 0.4% |

A 135× difference, so this sets how much data the model actually sees.

**Decision: Candidate A**, on compute-plausibility grounds rather than preference. Carry each
candidate through the paper's own stated recipe — 120 epochs, global batch 128, one H100:

| Candidate | Our steps | Scaled to the paper's 109,066 h |
|---|---:|---:|
| A | 331,057 | **1,070,281 steps** |
| B | 2,454 | **7,935 steps** |

Under Candidate B the paper's entire pretraining is under 8,000 optimizer steps. That is not a
foundation-model training run, it would take minutes rather than justifying a stated hardware
platform, and 120 epochs over 8,464 windows would overfit rather than pretrain. Specifying
"120 epochs, batch 128, one H100" is only informative under an interpretation that makes those
numbers consequential. Candidate A is the only reading consistent with the recipe the authors
chose to publish.

**What we are not claiming.** This is inference from the training recipe, not from the text.
The term remains undefined in the paper. A third reading exists that we did not adopt —
coverage as the *overlap fraction between adjacent windows*, giving a stride of `(1-c)×24h` —
which lands near Candidate B in volume and is arguably the more natural English reading of
"overlapping windows with a coverage ratio". It is recorded here so the choice is visible.

Candidate A also carries real redundancy: 33,736 hours become 353,127 windows, so each hour of
source data appears in roughly 250 windows, and adjacent windows share 287 of 288 positions.
Whether that redundancy helps or merely inflates epoch cost is a Tier-2 question, not a settled
fact.

Candidate B is retained as ablation T2.2. Escalated as authors' question §27 #2.

---

## D006 — Python 3.12 · DECIDED

`PROPOSED_EXTENSION`

Reference config pins `>=3.11,<3.13`; 3.12 was already provisioned locally via `uv`.
System Python is 3.10 and is not used.

---

## D007 — Stanford duplicate rows: drop exact, average conflicting · DECIDED

`SOURCE_VERIFIED` (the duplication is a property of the source file) ·
`INFERRED_RECONSTRUCTION` (the handling rule) · closes Q5

The Stanford CGM file contains 30,703 duplicate timestamps across 12 of 56 subjects, 6.33% of
all readings. This is what produced ~8.5% grid collisions under *both* binning rules in the
D004 audit, so it is not a binning artefact.

Structure of the duplication:

| Kind | Count | Share |
|---|---:|---:|
| Same timestamp, **identical** value | 30,693 | 99.97% |
| Same timestamp, **differing** value | 10 | 0.03% |

Differing-value gaps: median 16 mg/dL, p90 35, max 35. Worst subject S87 has 5,041 duplicate
timestamps in 12,935 readings, consistent with part of that subject's series having been
concatenated twice in the source.

**Rule:**

1. Exact duplicates (same session, timestamp, and value) are dropped, keeping the first
   occurrence. They carry no information and would otherwise inflate observation density,
   patch statistics, and the augmentation decimation threshold.
2. Same-timestamp readings with differing values are averaged, per the §9.2 collision rule,
   and counted in the reconciliation report rather than silently merged.

Both are recorded per session in the canonicalization report, so the count of dropped rows is
auditable rather than invisible.

**Why not keep them:** density weighting is central to both losses (§15.2, §15.3). A patch
whose observations are duplicated would be weighted as though it were twice as well observed,
biasing the loss toward subjects with duplicated source rows. Ten of the 56 subjects would be
affected, so this is not a rounding concern.

Deduplication is applied only within a session and only on an exact timestamp match. Readings
close in time but not identical are left to the binning rule.

---

## D008 — Partial windows rescue 9.9% of discarded data · **PROPOSED_EXTENSION**, public-plus only

`PROPOSED_EXTENSION` · not in the strict reproduction · raised by Stephane

### The observation

Blueprint §9.4 admits only windows whose full 24 hours fit inside a continuous segment. That
silently discards every segment shorter than 24 hours:

| dataset | segments | ≥24 h | <24 h | hours used | **hours discarded** | |
|---|---:|---:|---:|---:|---:|---:|
| colas | 469 | 68 | 401 | 3,204 | **6,318** | **66.4%** |
| stanford | 467 | 317 | 150 | 36,125 | 1,344 | 3.6% |
| big_ideas | 32 | 28 | 4 | 3,017 | 59 | 1.9% |
| shanghai_t2dm | 109 | 109 | 0 | 28,048 | 0 | 0.0% |
| **total** | | | | 70,394 | **7,721** | **9.9%** |

Two-thirds of Colas is unreachable. Its recordings run about two days and fragment at
long gaps into pieces mostly shorter than 24 hours, so its 9,522 recorded hours yield only
3,204 usable ones.

### Why the restriction is not obviously necessary

This model is built to consume arbitrary missingness. A window overlapping an 18-hour segment
is a window with six hours masked off, which is precisely what the physical observation mask
encodes. Nothing in the architecture distinguishes "masked because the sensor dropped out"
from "masked because the segment ended". Blueprint §9.6 already refuses to impose a minimum
density on ordinary windows and warns against hidden minimum counts.

The §9.1 rule that a window must not *cross* a segment boundary remains untouched: a partial
window extends into nothing, never into data separated by a gap of more than an hour.

### Measured effect

Allowing windows that fully contain a short segment, with out-of-segment positions masked:

| dataset | strict legal starts | rescued | rescued median coverage |
|---|---:|---:|---:|
| colas | 18,918 | **+39,997** | 58.3% |
| stanford | 342,278 | +27,216 | 23.3% |
| big_ideas | 28,136 | +445 | 82.3% |
| shanghai_t2dm | 305,293 | 0 | — |
| **total** | 694,625 | **+67,658 (+9.7%)** | |

Colas's contribution more than triples.

### Why it is not free

Rescued window quality varies sharply. Stanford's have a median coverage of 23.3% and a
minimum of 0.3%, roughly a single observation in 288 positions. Such a window contributes
almost nothing to the contextual loss and could plausibly harm training by flooding batches
with near-empty samples. A coverage floor is therefore part of the proposal, not an
afterthought, and its value is an ablation rather than a guess: `min_coverage ∈ {0.0, 0.10,
0.25, 0.50}`.

### Status

**Excluded from the strict reproduction.** The strict checkpoint exists to establish what the
paper's method does on the paper's public data; adding data the paper's rule excludes would
make the comparison meaningless, which is the failure blueprint §1.4 warns about.

Runs in the public-plus lane as GOAL KR5.7. If it helps, it is a genuine improvement on the
paper's data handling and is reported as such — separately, against its own baseline, never
folded into the reproduction number.

---

## D009 — sqrt(0) gradient guard, and the overfit gate criterion · DECIDED

`INFERRED_RECONSTRUCTION` (numerical guard, no equation changed) · closes the PR 8 bring-up gate

### The bug

`torch.sqrt(0)` has an infinite derivative, and `torch.where` differentiates the branch it does
not select. So the natural expression of blueprint §11.3,

```python
torch.where(has_observations, torch.sqrt(var), 0.0)
```

yields `0 * inf = NaN` in the backward pass whenever a patch has zero variance. Forward values
were correct, so every value-parity test against the NumPy reference passed. Only `rho` received
a NaN gradient, and only on real data — because zero variance requires a patch to be entirely
unobserved, which is exactly what JEPA masking does to the online branch on every step.

Fixed by substituting 1.0 *inside* the sqrt for the untaken branch (`_safe_sqrt`). Values are
bit-identical; only the gradient of the discarded branch changes, from infinite to zero.

**Why this matters beyond the fix:** 322 unit tests passed with this bug present. It was caught
by the 32-window overfit gate on the first optimizer step. Numerical-stability failures live in
the backward pass, and value-parity tests cannot see them. `tests/golden/test_gradient_stability.py`
now asserts no parameter receives a non-finite gradient across observation densities from 100%
to 10%, and pins the naive formulation as a failing control.

### The overfit criterion

Blueprint §18.1 step 2 says "overfit 32 windows and confirm losses fall" without a threshold.
An initial `< 10% of starting loss` rule was our own invention and is wrong for this objective:
the EMA target co-adapts and dropout is active, so a JEPA loss has a non-zero floor.

Measured on 32 windows, 600 epochs:

| dropout | start | end | reduction | final sigma |
|---|---:|---:|---:|---:|
| 0.0 | 0.84217 | 0.06557 | **92.2%** | 5.603 |
| 0.1 | 0.87436 | 0.13663 | 84.4% | 5.605 |

Dropout accounts for the difference, as expected. **Gate: with dropout disabled, loss must fall
by at least 90%, and sigma must move from its 6.0 initialisation.** Both hold.

Sigma moving to 5.60 under both settings is a second signal: the learned bandwidth is doing
something data-dependent rather than sitting at its initial value, which §12.4 asks us to check.

---

## D010 — float32, not mixed precision

**Status:** decided. `INFERRED_RECONSTRUCTION` — §17.1 does not name a precision.

Measured on the RTX 3090 at the paper's batch size of 128, 300 steps after 30 warmup:

| configuration | ms/step | windows/s | peak VRAM |
|---|---:|---:|---:|
| fp16 autocast + GradScaler | 26.4 | 4,847 | 0.12 GB |
| **float32** | **23.0** | **5,556** | 0.18 GB |

Mixed precision is 15% *slower* here. The step is not arithmetic-bound: a 0.44 M-parameter
encoder over 24 tokens leaves the GPU idle between kernel launches, so autocast's per-tensor
casts and the GradScaler's unscale pass cost more than the tensor cores return. Peak memory is
0.18 GB against 24 GB, so there is nothing for AMP to buy on the memory side either.

float32 is also the conservative choice for a reproduction: it removes a source of numerical
divergence from the headline numbers. The AMP path stays in `TrainConfig` behind `--amp` for
larger configurations, but it is off by default.

## D011 — torch.compile rejected for the headline runs

**Status:** decided. Rejected.

`torch.compile(dynamic=False)` on both encoder branches gives 22.4 → 15.6 ms/step, a 1.44x
speedup. From identical initial state and an identical mask seed, the first step's loss differs:

```
eager 0.84115416   compiled 0.84222174   relative 1.27e-03
```

1.3e-3 relative is too large for reduction-order effects at this size and matches TF32 matmul
precision, which Inductor enables. That makes it a precision change rather than a fusion win, and
a precision change is not something to accept silently on the numbers this project exists to
report.

A 1.44x speedup is not worth it: concurrency (D012) gives a larger reduction in wall-clock while
changing nothing about the arithmetic. Reconsider only if a future configuration is genuinely
compute-bound, and only with TF32 explicitly disabled and re-measured.

## D012 — the five seeds train concurrently on one GPU

**Status:** decided. `PROPOSED_EXTENSION` to the run plan; affects wall-clock only, not results.

Because the step is launch-latency bound, concurrent processes interleave into each other's idle
gaps. Independent processes, 300 steps each after 50 warmup:

| processes | aggregate step/s | slowest ms/step | total VRAM | wall-clock, 5 seeds |
|---:|---:|---:|---:|---:|
| 1 | 43.3 | 23.1 | 0.18 GB | 10h 37m |
| 2 | 66.7 | 30.1 | 0.35 GB | 6h 53m |
| 3 | 75.1 | 40.3 | 0.53 GB | 6h 07m |
| **5** | **79.9** | 63.1 | 0.89 GB | **5h 45m** |

Scaling is sublinear and saturates near 80 step/s — past three processes the GPU's launch queue
is full and additional work only queues. Five is still the shortest wall-clock and needs no
scheduling logic, so all five seeds launch together.

**This changes no result.** Each process runs the identical eager float32 code it would run
alone, with its own RNG streams and its own batch order derived from `(seed, epoch)`. A
concurrently-trained seed is bit-identical to a serially-trained one; only the wall-clock differs.
Contrast D011, which was rejected precisely because it would not have been.

## D013 — deterministic CUDA kernels are on by default

**Status:** decided. `PROPOSED_EXTENSION` — the paper makes no reproducibility claim at this level.

The first end-to-end resume gate failed, but only just: over two full epochs of real training the
resumed run and the uninterrupted run agreed to about seven significant figures and drifted
slowly apart.

```
epoch 1 loss    0.10845764627527552  vs  0.10845764368459139   delta  2.6e-09
epoch 2 sigma   3.3762643622522166   vs  3.3762646955887075    delta -3.3e-07
```

No state was lost. The cause is atomics in several CUDA backward kernels, which make float32
results depend on thread scheduling. This is the failure mode worth naming: a drift this small
looks like reproducibility in any plot, and is not reproducibility.

`torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` makes two runs
of a seed bit-identical, at 23.0 -> 24.9 ms/step, about 8%. The gate then passes exactly.

On by default; `--nondeterministic` remains for exploratory work. KR2.6 is met with this on.

**Two bugs the CPU test suite structurally could not see**, both found by this gate:

1. RNG states are CPU `ByteTensor`s. `torch.load(map_location="cuda")` moves them, and
   `set_state` then rejects them. Every CPU test loads to CPU and so never exercises it.
2. `Trainer.health` called `model.eval()` and did not restore the previous mode, so dropout was
   off for every epoch after the first diagnostic. It is invisible in a single run's loss curve —
   the curve simply steps down once at the first epoch boundary and looks healthy thereafter. It
   is only visible when comparing two runs that reached the same epoch by different routes.

Both now have regression tests, including a control asserting dropout does change the loss.

## D014 — CGMacros three-class diabetes risk uses standard HbA1c bands

**Status:** decided. `INFERRED_RECONSTRUCTION`.

§19.3 names the three CGMacros classes — normoglycemia, prediabetes, T2D — but not their cut
points, while giving explicit thresholds for every other label in the section. The standard
HbA1c bands are used: `<5.7` normoglycemia, `5.7–6.4` prediabetes, `>=6.5` T2D.

This is consistent with the Stanford rule §19.3 *does* state (`HbA1c >= 5.7` positive for diabetes
risk), which is the same 5.7 boundary; the reconstruction only adds the upper cut.

Resulting balance is 15 / 16 / 14 across 45 subjects — close to even, which is itself weak
evidence the cut points are the intended ones, since a wrong upper threshold would skew it.

Tagged `INFERRED_RECONSTRUCTION`, not `PAPER_EXACT`. It is the only one of the fourteen tasks
that required a threshold the blueprint does not state.

## D015 — undefined fold metrics are reported as NaN, never imputed

**Status:** decided. `PROPOSED_EXTENSION` — §19.4 does not say what to do here.

Two tasks have very small minority classes: `hall:hyperlipidemia` has 8 positives among 56
labelled subjects, and `shanghai_t2dm:hypoglycemia` has 10 among 109. Under a five-fold
subject-grouped split that is one to two positive subjects per test fold.

Folds are stratified by label so that no fold can contain zero positives — verified by a test
over all ten repeats of the 8-positive case. Where a metric is nonetheless undefined, the fold is
recorded as NaN with a reason and excluded from aggregation, and the count of dropped folds is
carried into every summary.

The alternative, substituting 0.5 for an undefined AUC, would pull exactly the tasks with the
least evidence toward a number nobody measured, and would do it invisibly.

These two tasks are reported with their subject counts beside them. A confidence interval built
on 8 positives is wide, and the write-up must say so rather than let it sit in a table looking
like the others.

## D016 — CGMacros observations are recovered, not resampled

**Status:** decided. `SOURCE_VERIFIED` — the recovery is verified against the source file itself.

CGMacros 1.0.0 publishes only a 1-minute grid with both CGM columns linearly interpolated; the
raw exports §19.10 calls for are not in the release. See amendment A6.

Three options were available:

1. **Use the published column.** Rejected. It would report a Dexcom day as fully observed when
   four of every five values are manufactured, making the physical mask — the invariant this
   project is built on — false for one of the four downstream datasets.
2. **Drop CGMacros.** Rejected. It carries 4 of the 14 dataset-tasks and is the only downstream
   source with paired Dexcom and Libre streams.
3. **Recover the original observations.** Adopted.

Linear interpolation is exactly invertible: the observations are the points where the slope
changes. This is a decidable property of the data, not a modelling assumption, and the round-trip
test settles it — re-interpolating the recovered points reproduces the published file to under
1e-9 on every subject tested.

Recovered: 45 subjects, 105,771 Dexcom observations on a 5-minute grid, 41,510 Libre on 15-minute.

The limitation is stated in A6 and worth repeating: three collinear readings collapse to two, so
the method under-counts on perfectly straight runs. It cannot over-count. For a mask-aware model
an under-count is conservative — it lowers a window's density weight rather than inflating it.

Sensors are kept in separate sessions and evaluated separately at native cadence, per §19.10,
and paired by subject only at split time, per §19.3.

## D017 — Stanford labels come from the CGM-contemporaneous visit

**Status:** decided. `SOURCE_VERIFIED`.

Both Stanford metadata tables carry one row per experiment type, so some subjects appear twice.
This was found by a join guard, not by inspection: the duplicate rows silently multiplied those
subjects' windows and would have weighted them more heavily than everyone else's in every probe.

The two tables behave differently and needed different handling.

`filtered_metabolic_tests.csv` — 12 duplicated subjects, **zero conflicting non-null values**
across all four columns (`sspg`, `sspg_2_classes`, `di`, `di_2_classes_median`). The second row
simply omits the disposition index. Collapsing on the first non-null value is lossless.

`filtered_study_participants_characteristics.csv` — 4 duplicated subjects with **genuinely
different lab values**, because the rows are different visits. Not a deduplication problem; a
choice of which measurement labels the CGM windows. We take the visit whose venous draw
accompanies the CGM recording (`venous_with_matching_cgm_and_with_planned_athome_cgm`), falling
back to the other where it is the only row, because that is the draw contemporaneous with the
windows being labelled.

**The choice changes no label.** Zero subjects have an HbA1c range straddling the 5.7 threshold —
S19 is 5.7/5.9, S27 5.1/5.2, S28 6.8/7.0, S32 5.9/6.0, each pair on the same side. The rule is
recorded because it is the principled one, not because the outcome depends on it.

**Correction to an earlier number.** Stanford was previously reported as 56 labelled subjects.
That was the duplicate inflation. The true figure is **44 subjects with metadata**, against 56
with CGM data — so 12 Stanford subjects contribute windows to pretraining but cannot be labelled
for any downstream task. That is a coverage limitation and belongs beside the Stanford results.

A guard in the evaluation driver now raises on any duplicated label index rather than proceeding,
so this class of error cannot recur silently in another source.

## D018 — target LayerNorm is available, off by default

**Status:** decided. `PROPOSED_EXTENSION`, disabled in the strict path.

The blueprint regresses onto raw EMA target tokens. I-JEPA layer-normalises target tokens before
regression, BYOL normalises both sides, and BYOL's ablation reports runaway representation norm
when it is omitted. Our target token norm does drift, 18 -> 34 -> 16, so the concern was concrete
rather than stylistic: regressing onto an unnormalised moving target mixes representation mismatch
with target scale drift, and the loss then stops meaning what it appears to mean.

`OpenCGMStateEvent(normalize_targets=True)` applies `F.layer_norm` to the target tokens.

**Measured, it hurts.** 0.6508 against 0.6537 at epoch 10 on the 18-entry downstream benchmark,
and it deepens the covariance rank collapse rather than relieving it. It stays in the code as a
labelled option because the negative result is worth reporting, and it stays off by default
because the strict reproduction must be exactly what the blueprint specifies.

## D019 — patch statistics are computed from the raw aligned sequence, not the normalised one

**Status:** decided. `PAPER_EXACT`. Default changed; the previous behaviour survives as an ablation.

Appendix C is explicit and uses two different symbols for two different signals, and we had
collapsed them into one.

C.1 defines the raw signal: "The aligned glucose sequence is denoted as X̂ ∈ ℝ^L". C.3 defines the
normalised one separately — "The filter is applied after mask-aware normalization. Let X̃ denote
the normalized aligned glucose sequence" — and every quantity in C.3's filter and decomposition is
written in X̃.

C.2 is written entirely in X̂. Equation 9, the state-stream patch mean and standard deviation, is
`μ_i = Σ M_j X̂_j / (Σ M_j + ε)`. Equation 10, the event-stream rate of change, is
`r_j = (X̂_j − X̂_{j−b}) / b`. Counting symbol occurrences in the appendix body:

| section | X̂ | X̃ |
|---|---:|---:|
| C.2, statistics and rate of change | 4 | 0 |
| C.3, filter and decomposition | 0 | 6 |

The separation is total. Neither section uses the other's symbol once, which is not the pattern of
loose notation.

**What we had done.** `ContextEncoder.tokenize` derived every branch from the instance-normalised
signal: patch statistics from the filtered normalised state component, rate of change from the
normalised sequence. Because per-window instance normalisation removes the mean and scale of each
day, this erased absolute glucose level from both statistics branches — 48 state dimensions and 32
event dimensions carried only shape. A day centred at 105 mg/dL and one centred at 205 produced
identical statistics tokens. Under the paper's ordering the filter is normalised and the statistics
are not, so absolute level reaches the model through μ_i and σ_i while the waveform the filter sees
stays scale-free. Those are different models, and we had built the wrong one.

**The choice.** `raw_statistics=True` is now the default and matches the paper. `μ_i`, `σ_i` and
`r_j` are computed from the aligned mg/dL sequence; the Gaussian filter continues to see the
normalised sequence, per C.3. `--normalized-statistics` restores the previous behaviour and is
retained as a Tier-1 ablation, because the difference between the two is now a measurable claim
about what absolute glucose level contributes.

**This is not yet an explanation of the negative result**, and must not be written up as one until
it is measured. Two earlier leading hypotheses — target LayerNorm and the instance-normalisation
ceiling — were both falsified when measured, and the honest prior from those is that a plausible
mechanism is worth about nothing until a probe number moves. The predeclared criterion is the same
one used for the other structural experiments: +0.02 macro ROC-AUC over the 0.6532 random
initialisation, with consistent task-level direction.

**Evaluation consequence, fixed at the same time.** These flags change what `encode` computes but
add no parameter tensor, so a model built from current defaults loads a pre-D019 checkpoint under
`strict=True` without complaint and silently produces different features from identical weights.
Measured, that moved a headline probe by 0.025 — larger than every effect this project is trying to
detect. `eval/embed.py` now reads the flags back from each checkpoint's own config, defaults them
to the pre-change behaviour when a config predates them, and folds them into the embedding cache
key so pre- and post-change caches cannot collide. `tests/golden/test_encoder_provenance.py` pins
all three properties, including a control asserting that the flag really does change the embedding.

**Related defect, still open.** C.2 states "Empty patches are zeroed by the validity mask". Our
`StatsBranch` emits its learned bias for a fully unobserved patch, so we do not satisfy that
sentence. It is now known to be `PAPER_EXACT` rather than a matter of taste, and needs its own
decision and test.

## D020 — an empty patch emits nothing, not a learned bias

**Status:** decided. `PAPER_EXACT`. Default changed; the previous behaviour survives as an ablation.

Appendix C.2 closes its definition of the patch statistics with one sentence: "Empty patches are
zeroed by the validity mask." We did not satisfy it, and the reason is the same class of error as
D019 — the code did something reasonable that the paper does not say.

`StatsBranch` received `mean` and `std`, which are already zero for a patch with no observations,
and projected them: `gelu(Linear(0, 0))`. A `nn.Linear` carries a bias and GELU is not zero at the
origin, so a fully unobserved patch emitted `gelu(b)` — a **learned, non-zero, input-independent
vector**, measured L1 norm 8.34 over 48 dimensions at initialisation. Zeroing the inputs does not
zero the output; the zeroing has to happen after the projection.

**Why it is not cosmetic.** That vector is identical for every empty patch in every window, so the
contextual Transformer receives a confident, consistent token wherever the sensor was silent, and
cannot distinguish it from a measurement. It also biases exactly the cohorts we can least afford
to distort: ShanghaiT2DM sits at 33% density and CGMacros Libre at 31%, so empty patches are
common there and rare in Hall (99%) and Stanford (100%). A learned constant standing in for
"no data" is therefore partly a dataset marker.

**The choice.** `zero_empty_patches=True` is now the default. `StatsBranch.forward` takes a
per-patch validity mask and multiplies the projected output by it, for both the state statistics
(48 dimensions, validity = any observed position in the patch) and the event statistics (32
dimensions, validity = any valid rate-of-change entry). `--nonzero-empty-patches` restores the old
behaviour and is retained as a Tier-1 ablation, since the difference is now a measurable claim
about what "no data" should mean to the encoder.

**Measured, and it does not help.** Seed 17, epoch 40, against the identical run without it:
**0.6780 against 0.6818** macro ROC-AUC, and +0.0355 against +0.0386 over the clinical baseline.
The difference is -0.004, roughly half the seed-level standard deviation of 0.0082, so this is a
null result rather than a small negative one — a single seed cannot resolve a difference this size.
(It does win on one secondary count, 29 of 36 per-task comparisons ahead against 27, which is the
kind of detail that would be easy to quote selectively and is not evidence of anything.)

**It stays on anyway**, because the paper specifies it and this project's standard is fidelity to
the disclosed method rather than to whatever scores best. A change adopted because it helped would
need the same five-seed evidence as any other claim; a change adopted because the paper says so
needs only to be correct. Recording it as measured-and-neutral is also the honest outcome: three
of the five "obviously right" fixes in this project have now failed to move a probe number, and
that base rate is itself worth remembering.

**Fallback semantics.** `ARCHITECTURE_FLAGS["zero_empty_patches"]` is `False`, so a checkpoint
whose config predates the flag is re-read with the behaviour it was trained under, and a resume
that would change it refuses. Verified against a live checkpoint from the running sweep.

## D021 — the CGM-JEPA comparator is ported from the authors' code, not inferred

**Tag: SOURCE_VERIFIED** (superseding an `INFERRED_RECONSTRUCTION` draft written the same day).

GlucoFM's headline claim is not its 66.7 average. It is **+4.11 PR-AUC and +4.34 ROC-AUC over
CGM-JEPA**, with both models re-pretrained on the same corpus (§4.2, Table 3). That distinction
matters to this project more than any other single fact about the paper: an absolute score falls
when you hold 30.9% of the pretraining hours, but a margin between two models trained on the same
reduced corpus does not. Reproducing the margin is the honest target; reproducing the number is not.

**The first draft of this comparator was inferred from appendix B and was wrong.** Appendix B gives
the width, depth, heads, mask ratio, batch, learning rate and epoch count, so it reads as a complete
specification. It is not, and four choices were guessed. The authors' implementation is public and
MIT licensed — `github.com/cruiseresearchgroup/CGM-JEPA`, master @ 2026-05-11, with weights on
HuggingFace at `CRUISEResearchGroup/CGM-JEPA` — so none of them needed guessing. Read against
`models/encoder.py`, `models/predictor.py`, `utils/embed.py`, `utils/modules.py`,
`utils/mask_utils.py`, `config/config_pretrain.py` and `pretrain/pretrain_cgm_jepa.py`, the guesses
were wrong in these ways, **each of which weakened the baseline**:

| | inferred draft | authors' actual | why it mattered |
|---|---|---|---|
| EMA target | used raw | `F.layer_norm` before the loss (`pretrain_cgm_jepa.py:180`) | without it the student lowers the loss by shrinking the teacher — this project's own D018 |
| masked patches | replaced by a learned mask token at the encoder input | **dropped entirely** before attention (`encoder.py:97`) | placeholders make the pretext task easier and burn encoder capacity |
| encoder dropout | 0.1 | **0.0** (`config_pretrain.py:46`) | unrequested regularization on a 0.5M-parameter model |
| FFN width | `dim * 2` = 192 | `mlp_ratio 4.0` = 384 | half the feed-forward capacity |
| patch embedding | `nn.Linear(12, 96)` | `Conv1d(1, 96, k=3, s=3)` within each patch, then `Linear(384, 96)` | a local filter bank inside the hour, strictly more expressive |
| predictor | 1 layer at full width 96, 6 heads | 1 layer at **48 dims, 2 heads**, mask-query conditioned on target position | a deliberately weak predictor forces information into the encoder, which is the thing being probed |
| loss | smooth L1 | plain L1 (`pretrain_cgm_jepa.py:25-30`) | minor |
| EMA schedule | cosine 0.997 → exactly 1.0 | linear ramp per epoch, `ipe_scale` 1.25, ending ≈0.9994 | reaching 1.0 freezes the teacher for the final epochs |

The draft's stated justification — that each guess made CGM-JEPA *more like GlucoFM* and therefore
erred conservatively — was simply bad reasoning, and it is recorded here because the reasoning is
the reusable lesson. Making a comparator resemble the model it is being compared against does not
systematically help the comparator. Dropout 0.1, the borrowed EMA schedule and the half-width FFN
were all "more like GlucoFM" and all plausibly *hurt* CGM-JEPA relative to its own recipe. The only
conservative direction for a baseline is its authors' own configuration.

**The gate.** The ported encoder counts **521,584 parameters**. The authors' counts 522,160; the
576 difference is exactly the `Linear(5, 96)` time-feature embedding their `DataEmbedding` always
constructs and which `use_time_feature: False` (`config_pretrain.py:29`) leaves permanently dead. We
omit the dead module rather than carry untrained weights to make a number match. Both round to the
**0.52M the GlucoFM paper reports for CGM-JEPA in Table 3** — independent confirmation that the port
is the same model, and the reason this is recorded as a passing test rather than a claim.
The inferred draft came to roughly 0.35M and would have lost the comparison on capacity, not method.

**Standing rule this establishes.** Before reconstructing any comparator from a paper's appendix,
check whether its authors released code. This project spent effort reasoning carefully from an
incomplete description while a complete one was two web requests away.

**Author overlap, recorded because it affects how the result should be read.** CGM-JEPA shares
three authors with GlucoFM. The paper's headline is therefore its own group's earlier model losing
to its newer one. That makes an independent reproduction of this comparison more valuable, not less,
and it raises rather than lowers the bar for our baseline being the authors' actual recipe.

**Scope.** Vanilla CGM-JEPA only. The family also contains X-CGM-JEPA, which the paper's own Table 3
puts within a rounding error on average (54.6/62.1 against 54.7/62.4) and *ahead* on several
individual tasks. Reproducing vanilla CGM-JEPA reproduces the headline; it does not reproduce
"best of the family". X-CGM-JEPA needs the Glucodensity KDE pipeline (32x32 grids, spatial patch 8)
and is deferred, not dismissed. Any claim we publish must say "over CGM-JEPA", never "over the best
CGM-specific foundation model", unless X-CGM-JEPA is also run.

## D022 — the comparator sees raw mg/dL, because its authors' default does

**Tag: SOURCE_VERIFIED**, resolving a contradiction inside the GlucoFM paper.

Appendix B says CGM-JEPA's inputs are "linearly interpolated to 288-point daily sequences, **and
normalized**", and says GlucoFM retrained it "using the official configuration". Those two
statements are inconsistent: the official configuration is `normalize_x: False`
(`config/config_pretrain.py:17`) — raw mg/dL, no per-window standardization.

**We follow the authors.** Two reasons, and the second is the one that decides it:

1. "The official configuration" is the more specific and more verifiable of the paper's two claims,
   and the code is authoritative over a prose summary of the code.
2. It is the direction that **helps the comparator**. Raw values preserve absolute glucose level,
   and this project's own D019 measured absolute level as worth up to 0.21 ROC-AUC on obesity.
   Standardizing per window would strip that signal from the baseline while our model retains it —
   a large, quantified, one-sided handicap on the one experiment where we have an interest in the
   outcome. Where the evidence is genuinely ambiguous, the tie goes to the baseline.

`to_patches(..., normalize=True)` exists so appendix B's reading can be run as a **labelled
sensitivity arm**. If the two arms disagree materially, both get reported; the authors' default
remains the headline. The result is not to be quoted from whichever arm scores better for us.

**The interpolation stands and is not a defect.** CGM-JEPA fills every gap and treats the filled
values as observed. GlucoFM keeps the physical mask throughout, and this project forbids
interpolation everywhere else as a non-negotiable. That disagreement is precisely what the
comparison tests — our Tier-1 ablation already measured the never-interpolate rule as worth 0.017
ROC-AUC inside our own architecture, and the comparator is where it gets tested across architectures.
The `interpolate_dense` path is therefore used deliberately here, and only here.

---

## D023 — Lane D is a teacher-student pilot, not a probe-on-encoder · `PROPOSED_EXTENSION`

The PPG+CGM paired dataset on disk (`ppg_cgm_paired_zenodo_20577959`, 5 participants, 3327 BVP
segments at 64 Hz, paired with CGM in mmol/L at 15-min cadence, CC-BY-4.0) is too small to
materially change the strict CGM pretraining corpus. Three integration strategies were
considered:

| Strategy | What it does | Why we did not pick it |
|---|---|---|
| **A — Lane A pretraining expansion** | Add ~3,600 CGM-only windows from the 5 subjects | 5 subjects → 1 test subject per fold; subject-identity overfit; the PPG is wasted |
| **C — Capped Lane A contribution** | Add ~150 windows with per-subject cap | 4% corpus bump swallowed by 5-seed noise; same fold-collapse problem; PPG still wasted |
| **B — Lane D teacher-student pilot** | Frozen strict-ep40 CGM encoder as teacher; small 64 Hz→1 Hz PPG encoder as student; alignment + glucose heads | The blueprint already assigns this dataset to lane D (`role: ppg_bridge`), §6.9 says "too small to materially change CGM-only pretraining", and §23 defines the canonical teacher-student use |

**The choice.** Strategy B, framed as the §23 PPG-bridge pilot.

**Architecture.**

- *Teacher.* Frozen strict ep40 checkpoint (`runs_5090/rawstats120/seed17/ckpt_ep040.pt`),
  mean-pooled 256-dim `opencgm_mean` head, eval mode, gradients off.
- *Student input.* Each BVP JSON segment (~30 min × 64 Hz) is split into 5-minute patches
  matching the CGM grid. The per-subject glucose CSV supplies the aligned CGM window at the
  same timestamps.
- *Student encoder.* A small 1D conv stack on 64 Hz BVP downsampling to ~1 Hz features
  (HR-rate summary), then per-patch mean-pool → one student token per CGM timestep.
  **No attempt to inject 64 Hz raw samples into the existing CGM encoder** — the rate
  mismatch is 3,840× and would force architectural surgery that breaks the teacher-student
  framing.
- *Two heads.*
  1. **Teacher-latent head.** 256-dim projection, loss = 0.5·MSE + 0.5·(1−cosine) on the
     teacher's tokens at the same timestamps, masked by the CGM observation mask (mask
     before loss, same rule as §10.1).
  2. **Direct glucose head.** Per-token mmol/L regression with causal Gaussian NLL (§12),
     masked. Sanity check, not the headline.
- *Evaluation.* Mean-pool student tokens across the window → cosine alignment to the
  teacher-window-embedding (also mean-pooled) and RMSE/MAE on glucose. **Never interpolate
  CGM** when computing alignment or loss.

**Validation protocol.**

- **Subject-disjoint 5-fold × 5 student seeds** (student-side seeds 1003, 1019, 1043, 1071,
  1103 — distinct from the CGM pretraining seeds).
- **Two baselines:** (a) identity (student latent = population-mean teacher latent); (b)
  per-subject ridge regression from BVP-summary features to CGM at each timestamp.
- **Per-fold numbers and subject-level bootstrap CI** — 5 subjects is small, so a fold-mean
  CI is misleading. Report the per-fold numbers.
- **Auxiliary probe (not headline):** does the student latent, fed into the existing
  14-task probe infrastructure, transfer to a small downstream task? Cheap piggyback on the
  existing strict probe code.

**What's out of scope, written down so it doesn't creep back in.**

- **No pretraining integration.** Lane D does not modify the strict encoder. The teacher
  stays frozen. KR5 ("improvements live in separate checkpoints and separate tables")
  applies without contortion.
- **No weight release.** `weight_release: permissive_pending_review` on `ppg_cgm_paired`
  stays pending; the pilot checkpoint is internal/research-only.
- **The capillary Stuba dataset (`ppg_capillary`)** stays out of this pilot. Lane D
  registration for it is `ppg_auxiliary`, not `ppg_bridge`. Out of scope for GlucoFM
  reproduction (§6.9) and for this teacher-student pilot.

**Where this lives on disk.**

```
src/opencgm_stateevent/ppg/
  encoder.py        # small 64 Hz → 1 Hz 1D-conv student encoder
  heads.py          # teacher-latent head + direct glucose head
  align.py          # BVP-segment → CGM-window timestamp alignment
  losses.py         # mask-aware alignment loss + causal Gaussian NLL

scripts/
  ppg_teacher_student.py   # the pilot trainer
  evaluate_ppg_pilot.py    # the alignment + RMSE/MAE reporter

reports/eval/ppg_pilot/
  ckpt_seed{NN}.pt        # separate from any strict-corpus checkpoint
  alignment_curve.csv
  glucose_regression.csv
  per_fold.csv
  run_record.json
findings/ppg_pilot.md     # the writeup, framed as a separate scientific question
```

**Headline outcome expected: 0.670 ± 0.003 unchanged.** That is the honest outcome. Pretending
otherwise — by adding 5 subjects to pretraining and reporting a "63% corpus" — would either
not move the number or risk moving it backward. A separately-reported teacher-student result
that doesn't pretend to be the headline is a stronger document than a quietly-inflated
corpus percentage.

---

## D024 — teacher is the strict ep40 checkpoint, not ep120 · `INFERRED_RECONSTRUCTION`

The pilot teacher (D023) is the strict-pretrain ep40 checkpoint
(`runs_5090/rawstats120/seed17/ckpt_ep040.pt`), not ep120.

**Why.** STATE.md (epoch 120 measurements): mean ROC-AUC 0.6750 ± 0.0082 at ep40,
0.6749 ± ... at ep120 — within 1 sd. The model peaks at ep40 on 30.9% corpus; ep120 is mildly
worse by ~1 sd on the headline. The +0.0269 [+0.0222, +0.0316] CI excludes zero, the seed-mean
is robustly positive, but the absolute gain over ep40 is small and negative.

**Inference this commits to.** When the corpus is too small for the architecture's capacity,
more epochs do not help — they slightly hurt. The teacher for a teacher-student pilot should
be the best checkpoint, not the longest-trained one.

**Tag.** `INFERRED_RECONSTRUCTION` because the paper trained for 120 epochs and a faithful
reproduction would do the same. We are *not* training ep40 only to inflate numbers; we are
training 120 epochs because the blueprint says so (D012) and reporting the best epoch.

**What this changes.** Nothing in the strict corpus pipeline. The pilot teacher is a read-only
load of the existing ep40 checkpoint.

## D025 — The heads bundle carries its own licence, one step stricter than the encoder

**Tag:** `SOURCE_VERIFIED` (licence terms read from the registry and the
CC 4.0 deed) · **Supersedes:** the first version of `scripts/publish_web_assets.py`, which
withheld heads instead of labelling them.

**The question.** `artifacts/glucofm_heads.json` holds 18 probe heads. Eight are fitted on
CGMacros (`CC-BY-NC-SA-4.0`, `license_confidence: unverified`, Q4) and three on Stanford
(source repository states no licence). The website and the HTTP API serve this file to every visitor, which
is distribution. Which heads may go out, and under what terms?

**What the first attempt got wrong.** It withheld both CGMacros and Stanford heads, leaving 7
of 18 (4 with signal). For Stanford that was incoherent: Stanford is Lane A and contributes
171,140 of the corpus's 353,127 windows, so it is already inside the published encoder.
Withholding three classifiers fitted on Stanford labels while shipping the encoder that
pretrained on half the Stanford dataset protects nothing — the exposure is the encoder, and
the encoder is published, and the registry records the source as unlicensed.

**The decision.** Publish all 18. Split the licences by artefact:

| Artefact | Licence | Why |
|---|---|---|
| `glucofm_encoder.onnx` | CC-BY-NC-4.0 | Lane A only; no share-alike source ever entered pretraining |
| `glucofm_heads.json` | **CC-BY-NC-SA-4.0** | Contains classifiers fitted on CGMacros; share-alike is inherited |

**Why this is not a loophole.** Share-alike is not a permission problem here, it is a labelling
obligation: CC BY-NC-SA 4.0 §3(b) requires adapted material to be offered under the same
licence. This project is unfunded non-commercial research, so NonCommercial is satisfied on its
face, and the weights already forbid commercial use. Offering the bundle under BY-NC-SA
discharges the share-alike term exactly as written. Keeping the two artefacts separately
licensed stops that term reaching the encoder, which has no CGMacros in it and owes nothing to
CGMacros' licence.

**The standing rule is narrowed, not broken.** "Lane E never enters a distributed checkpoint"
still holds and is unchanged: no Lane E data entered pretraining, and the encoder is clean. The
rule was written about the checkpoint and is now stated as such. A classifier fitted on frozen
embeddings is a separate artefact with its own provenance and its own licence.

**What still forbids distribution.** UCHTT1DM is CC-BY-NC-**ND**. No-derivatives has no
labelling escape — a fitted classifier cannot be redistributed under any label — so any head
fitted on it stays out permanently. There are none today; the filter enforces it anyway.

**Enforced by.** `tests/unit/test_source_rights.py::test_published_heads_bundle_declares_the_licence_its_sources_impose`
fails if a share-alike source appears without the matching bundle licence, and
`::test_encoder_is_never_licensed_by_a_source_it_did_not_train_on` fails if share-alike leaks
into `LICENSE-WEIGHTS`.

**Open.** Q4 is still open: CGMacros' `CC-BY-NC-SA-4.0` label came from a third-party survey,
not PhysioNet, whose `LICENSE.txt` was empty. If it turns out to be more permissive, the bundle
licence can be relaxed. If it turns out to be ND, those eight heads come out.
