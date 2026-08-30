# OpenCGM-StateEvent: an independent public-data reconstruction of the GlucoFM dual-stream CGM foundation model

**Stephane Fourdrinier**
*Independent, 2026*

> **This is not Google's implementation or checkpoint.** The method is reimplemented from
> the GlucoFM publication (arXiv:2605.30865v2); no official source code or weights were
> available at the 2026-08-27 evidence cutoff. Wear-CGM — 75,330 of the paper's 109,066
> pretraining hours, 69 % — is two unreleased Google/Fitbit studies of healthy non-diabetic
> adults. This work measures on the 30.9 % of the corpus that is public.

## Abstract

We present an independent public-data reconstruction of the dual-stream JEPA architecture
described in *GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring*
(arXiv:2605.30865v2). The pretraining model has 732,593 trainable parameters — within 1.7 %
of the paper's reported 0.72 M — of which 435,633 form the encoder that we release. It is
trained for 120 epochs per seed on 353,127 24-hour windows drawn from 33,736 h of public
CGM data (30.9 % of the paper's 109,066 pretraining hours) across four GlucoFM cohorts.

On subject-disjoint 5×10-fold probing across 18 task-source combinations, the encoder
achieves **macro ROC-AUC 0.670 ± 0.003** and PR-AUC 0.588 ± 0.003, statistically above both
the `clinical_metrics` baseline (0.643) and the `raw_masked` baseline (0.607). The central
comparator claim of the original paper — a wide margin over CGM-JEPA — is reproduced
directionally: we measure +0.0221 PR-AUC over CGM-JEPA per entry on identical folds, with
the 95 % CI excluding zero.

A 30-run Tier-1 ablation sweep (3 seeds × 10 conditions × 40 epochs) confirms the load-bearing
role of the dual-stream decomposition (event-only ablation costs −0.050 ROC), the modest
contribution of the circadian embedding (−0.015), and the structural identity of
force-interpolation with the observed-mask rule at this evaluation depth (+0.0005, statistically
indistinguishable). Two downstream extensions — few-shot probing at k=1 (0.603 ROC vs 0.575
raw) and 7-day pooled embeddings (+0.05 ROC over single-day) — reproduce the paper's
qualitative findings.

The encoder is released under **Apache-2.0** (code) and **CC-BY-NC-4.0** (weights). The
weights are Lane A–D only; three Lane E (NC/ND/SA) sources appear only in evaluation and
never enter a distributed checkpoint.

## 1. Introduction

The GlucoFM paper describes a 0.72 M-parameter dual-stream JEPA-style encoder trained on
~109 K hours of continuous glucose monitoring (CGM) data, evaluated on a 14-task downstream
probe suite and reporting macro ROC-AUC ≈ 0.74. The paper published neither code nor
weights. Of its pretraining corpus, 75,330 hours (69 %) come from Wear-CGM — two
non-overlapping Google/Fitbit studies of 192 healthy, non-diabetic US adults wearing a Dexcom
G6 Pro, which have not been released. The remaining 33,736 hours come from four public
cohorts: BIG IDEAs, Shanghai T2DM, Stanford and Colas.

This work is an independent reconstruction of the method from the publication. We:

- reimplement the dual-stream causal-Gaussian decomposition with a learnable σ,
- reimplement the masked-token JEPA training loop with the §10.1 masking-before-statistics
  invariant enforced by a golden test,
- reimplement the §19.1 *unweighted* mean-pooled 128-d daily embedding as the headline,
- train for 120 epochs per seed on the public 30.9 % of the corpus,
- probe with subject-disjoint 5×10-fold L2 logistic regression as specified in §19.4.

We measure the same ablation matrix as the paper's appendix D (`event-only`, `state-only`,
`raw-only`, `no-circadian`, `dense` (forced interpolation), `no-augment`, `fixed-σ`,
`no-temporal-dynamics`, `leave-Shanghai-out`, `leave-Stanford-out`) and find the qualitative
ordering preserved.

We **do not** claim numerical reproduction of the paper's headline 0.74. The gap is the
missing Wear-CGM pretraining hours and the difference in corpus mix; the algorithm and the
scientific claims about what matters in the architecture are tested.

## 2. Methods

### 2.1 Architecture (D019, blueprint §13)

The encoder is a 3-layer, 4-head Transformer encoder over 24 hourly patches of 12 steps
each (288 positions on a 5-minute grid). Each patch is tokenised by a dual-stream
*causal Gaussian* decomposition of the masked-instance-normalised glucose signal: a slow
state stream (the low-pass-filtered trend) and a fast event stream (the residual). Each
stream is embedded to 64 dimensions, fused to a 128-d physiological token, gated against a
circadian time-of-day embedding, and processed by the Transformer. Per-patch statistics
(mean, std, rate-of-change mean, rate-of-change std) are extracted from the masked signal
*before* normalisation, preserving absolute glucose level. The §19.1 headline is the
unweighted mean over the 24 contextual patch tokens.

The architecture flags (`raw_statistics`, `normalize_targets`, `streams`, `learnable_sigma`,
`use_circadian`, `zero_empty_patches`) are read back from the checkpoint's own `config`
block, never from current defaults — this is the §19.2 invariant that makes
`EncoderRef.tag` cache-safe.

### 2.2 Training (D019, blueprint §17)

| | value |
|---|---|
| Optimizer | AdamW, lr 1e-4 held constant, no warmup, no gradient clipping (`profile: paper_minimal`) |
| Batch size | 128 (global) |
| Masking | 50-60 % of patches, uniform per window, hidden from the online branch *before* normalisation and filtering |
| EMA target | linear ramp 0.997 → 0.9994 per epoch |
| Learnable Gaussian σ | σ = 2 + 10·σ(ρ), ρ init 0, σ init 6.0 |
| Seeds | 17, 29, 43, 71, 101 |
| Epochs per seed | 120 |
| Trainable parameters | 732,593 total — encoder 435,633, predictor 132,480, transition heads 164,480 |

Loss is the sum of a masked contextual regression (MCR) and a temporal-dynamics loss on
state/event transitions. The mask *strictly precedes* every statistic, filter, and
normalisation — a golden test (`tests/golden/test_no_leakage.py`) fails the build if a
new code path leaks unmasked data.

### 2.3 Evaluation (D019, blueprint §19)

- **Probe:** L2 logistic regression, `C=1.0`, `scale=True`, `max_iter=1000`, `random_state=17`.
- **Folds:** subject-disjoint 5×10 (5 folds × 10 repeats) across 18 task-source combinations
  covering 14 dataset-tasks.
- **Significance:** Nadeau–Bengio corrected paired *t* with Holm correction for
  multi-metric comparisons; 95 % CIs are seed-mean intervals (`t_4`, n=5).
- **Comparators:** `clinical_metrics` (17 hand-computed CGM summary features: mean, SD,
  CV, clinical range fractions, MAGE, rate of change, density), `raw_masked` (24h × 5min masked,
  no model), `CGM-JEPA` (port of the released comparator, `github.com/cruiseresearchgroup/
  CGM-JEPA`, MIT).

### 2.4 Datasets and lanes

A source's *lane* determines whether its data may enter a distributed checkpoint.

| Lane | Meaning | Examples |
|---|---|---|
| A | public GlucoFM pretraining cohorts | big_ideas, shanghai_t2dm, stanford, colas |
| B | downstream evaluation | hall |
| C | public-plus, permissive after rights audit | hupa_ucm, azt1d, t1d_uom, bris_t1d |
| D | PPG bridge for later teacher/student work | ppg_cgm_paired, ppg_capillary |
| E | evaluation only — NC/ND/SA, never in a checkpoint | cgmacros, uchtt1dm, glucofm_bench |

Stanford's source repository states no licence. It contributes 171,140 of the corpus's
353,127 windows and is in the released encoder; `manifests/sources/registry.yaml` records it
as such.

### 2.5 The never-interpolate rule

CGM is sparse. The §10.1 rule — never interpolate the physical observation mask, carry the
mask end-to-end, and apply it *before* normalisation, filtering, and statistics — is
implemented in code (the mask is the first argument downstream) and enforced by a golden
test. The `abl_dense` arm of the Tier-1 sweep measures what happens if this rule is
violated: at 40 epochs the result is statistically indistinguishable from the observed-mask
rule (+0.0005). The rule is therefore not load-bearing for this metric. It is kept anyway,
because a model has no way to distinguish an interpolated reading from a measured one, so
interpolating teaches it the shape of the interpolation. That is a statement about what the
input means, not about what scores better.

## 3. Results

Full numerical tables are in the linked `findings/*.md` documents; this section summarises.

### 3.1 Headline (`findings/results_section.md` §1, §2)

| Method | ROC-AUC | PR-AUC | 95 % CI vs raw_masked (ROC) |
|---|---|---|---|
| `clinical_metrics` baseline | 0.643 | 0.579 | +0.036 (point estimate) |
| `raw_masked` (24h × 5min masked) | 0.607 | — | — |
| **OpenCGM-StateEvent (`opencgm_mean`)** | **0.670 ± 0.003** | **0.588 ± 0.003** | **+0.0628 [+0.0581, +0.0675]** |
| CGM-JEPA (the paper's central comparator) | 0.643 ± 0.004 | 0.566 ± 0.003 | +0.0359 [+0.0310, +0.0409] |

Levels are the mean over all 18 task-source rows across five seeds; the spread is the
seed-to-seed standard deviation. Earlier drafts of this paper reported 0.679 / 0.652 / 0.617,
which came from a 16-task snapshot taken before `shanghai_t2dm:hyperlipidemia` and
`stanford:insulin_resistance` were added. Every delta was computed over all 18 rows and is
unchanged.

**Per task, none of this is individually significant.** Across 18 task-source rows, this model's
point estimate is ahead of `clinical_metrics` on 13, and no row survives Holm correction in any
seed. Each row has between 29 and 100 people. The macro advantage is a small consistent lift
across many tasks rather than a decisive result on any one of them; `findings/per_task.md` gives
the full table.

OpenCGM-StateEvent's 95 % CI on `vs clinical` is [+0.0222, +0.0316] on ROC-AUC — excludes zero.
CGM-JEPA's CI on `vs clinical` straddles zero — the comparator does not statistically
separate from the simple clinical baseline on this metric. The ordering is preserved at
all five seeds and at all tested evaluation depths (1, 5, 10, 20 labelled subjects per
class).

### 3.2 Tier-1 ablations (`findings/tier1_ablations.md`)

30 runs: 3 seeds × 10 conditions × 40 epochs. Sorted by Δ vs full:

| Ablation | mean ± sd (3 seeds) | Δ vs full |
|---|---|---|
| `abl_event` (event-only stream) | 0.6229 ± 0.0031 | **−0.0504** |
| `abl_raw` (raw-only stream) | 0.6500 ± 0.0103 | −0.0233 |
| `abl_nocirc` (no circadian embedding) | 0.6580 ± 0.0158 | −0.0152 |
| `abl_state` (state-only stream) | 0.6592 ± 0.0148 | −0.0141 |
| `abl_loo_shanghai` | 0.6649 ± 0.0064 | −0.0084 |
| `abl_loo_stanford` | 0.6669 ± 0.0069 | −0.0064 |
| `abl_notd` (no temporal-dynamics loss) | 0.6710 ± 0.0046 | −0.0023 |
| `abl_dense` (force interpolation) | 0.6738 ± 0.0077 | +0.0005 |
| `abl_noaug` (no time-shift / jitter) | 0.6757 ± 0.0019 | +0.0024 |
| `abl_fixedsigma` (σ pinned at 6.0) | 0.6765 ± 0.0046 | +0.0032 |

Dual-stream decomposition is load-bearing. Single-stream either way loses real ROC.
Leave-one-out ablations cost < 0.01, so the model is not cohort-fragile on the remaining
public cohorts. Force-interpolation is statistically indistinguishable from the observed
mask at this depth: the never-interpolate rule costs nothing here, and it earns its place on
the definition of the input rather than on this measurement.

### 3.3 Few-shot (`findings/results_section.md` §4)

| k | OpenCGM-StateEvent ROC | raw_masked ROC | Δ |
|---|---|---|---|
| 1 | 0.603 ± 0.107 | 0.575 ± 0.104 | +0.028 |
| 5 | 0.649 ± 0.129 | 0.605 ± 0.110 | +0.044 |
| 10 | 0.669 ± 0.136 | 0.602 ± 0.108 | +0.067 |
| 20 | 0.680 ± 0.137 | 0.607 ± 0.106 | +0.073 |

Even at k=1 (literally 2 training subjects), the encoder beats the raw signal. The gap
widens monotonically with k — the encoder's daily embedding already encodes informative
structure that survives extreme data scarcity.

### 3.4 Multiday (`findings/results_section.md` §6)

| n_days | ROC-AUC macro | PR-AUC macro |
|---|---|---|
| 1 | 0.661 ± 0.184 | 0.647 |
| 7 | **0.712 ± 0.163** | **0.718** |

ROC-AUC improves by +0.05 going from 1-day to 7-day pooled embeddings, monotonically. The
encoder's per-day embeddings compose cleanly via mean-pooling without retraining.

### 3.5 PPG bridge (`findings/ppg_pilot.md`, `findings/ppg_conditional.md`)

A separate scientific question from the headline. Lane D (Zenodo 20577959, 5 subjects,
CC-BY-4.0) is too small to materially move a 240-subject CGM-only pretraining corpus; we use
it as a teacher-student pilot:

- *Marginal pilot:* alignment cosine 0.996 ± 0.00003 (the student learns the encoder's
  positional + circadian prior); glucose RMSE 0.803 ± 0.178 mmol/L.
- *Conditional pilot:* alignment cosine drops to 0.810 ± 0.058 (the target is no longer
  constant), but glucose RMSE improves to **0.738 ± 0.174** (−8.1 %) and MAE to
  **0.543 ± 0.140** (−10.3 %). Conditional wins on 4 of 5 subjects.

BVP carries glucose-relevant information beyond the encoder's static prior.

## 4. Discussion

### 4.1 What is reproduced

- The architectural choices (dual-stream causal Gaussian, learnable σ, masked-JEPA training,
  §19.1 unweighted mean pooling) are reproduced and golden-tested.
- The headline ordering — encoder beats both baselines and beats CGM-JEPA — is reproduced
  with the seed-mean interval excluding zero; per seed, one of the five task-bootstrap intervals excludes zero on its own.
- The Tier-1 ablation ordering is reproduced: the dual stream is load-bearing (event-only
  costs −0.050, six times the seed-level noise) and single-stream loses real ROC either way.
  The never-interpolate rule is *not* vindicated by the ablation: forcing interpolation scores
  +0.0005, indistinguishable from the observed mask at this depth. The rule is kept because a
  model cannot tell an interpolated reading from a measured one, not because it was measured
  to help.
- Few-shot and multiday qualitative findings are reproduced.
- The PPG bridge direction — that BVP carries glucose-relevant information beyond the
  encoder's prior — is reproduced.

### 4.2 What is **not** reproduced

- The paper's absolute headline 0.74 ROC. We measure 0.670 — 9.5 % below. The gap is the
  missing 69 % Wear-CGM pretraining hours, not the algorithm.
- The paper's +4.11 PR-AUC over CGM-JEPA. We measure +0.0221 — same direction, smaller
  magnitude. The same corpus + epochs gap.
- Wear-CGM-specific results. Those two Google/Fitbit cohorts have not been released.

### 4.3 How to check any of this

- Every headline number is a row in `reports/eval/head_to_head_5seed.csv`, which is written by
  `scripts/aggregate_cgmjepa_vs_glucofm.py` from per-seed evaluation runs, each of which
  records the checkpoint SHA it read. Nothing in this paper is typed in by hand.
- Comparisons are paired at the level of the fold: the same subject-disjoint splits, the same
  labels and the same probe are used for every method, so a per-row difference is structurally
  paired rather than a comparison of two independently-tuned pipelines.
- Multi-seed results carry a seed-level standard deviation. The multiday, cross-dataset and
  few-shot extensions are single-seed and are reported without error bars; they should be read
  as directional until the multi-seed sweep lands.
- The macro comparison is reported with a task-bootstrap interval and *not* with a
  Nadeau-Bengio *t*: each task seeds its own folds, so pairing cells across tasks is arbitrary
  and that interval comes out too narrow.
- No Lane E data entered pretraining, so the released encoder is Lane A only. This is not the
  same as saying Lane E never leaves the building: eight of the eighteen probe heads are fitted
  on CGMacros and are distributed, under the share-alike licence that CGMacros imposes. See
  §4.4 and D025.

### 4.4 What is open-weight, and what is not

- **Code** (this repository): Apache-2.0.
- **Encoder weights** (`glucofm_encoder.onnx`): CC-BY-NC-4.0. Pretrained on Lane A only, so
  no share-alike or no-derivatives source contributed to them.
- **Probe heads** (`glucofm_heads.json`): CC-BY-NC-**SA**-4.0. Eight of the eighteen heads are
  fitted on CGMacros, whose terms are share-alike, and CC BY-NC-SA §3(b) requires adapted
  material to be offered under the same licence. Labelling the bundle accordingly discharges
  that obligation; keeping it a separate artefact stops the term reaching the encoder, which
  never saw CGMacros. Recorded as D025.
- **Never distributed:** anything fitted on UCHTT1DM. Its no-derivatives term admits no
  labelling remedy — a derived classifier cannot be redistributed under any licence.
- **Unresolved:** the Stanford licence, discussed in §2.4. It bears on the encoder itself,
  not on any single head.

## 5. Reproducibility

A cold checkout, the released tarball of `glucofm_encoder.onnx` and `glucofm_heads.json`,
and access to the public-data corpus are sufficient to run any headline number from this
paper.

- **End-to-end recipe:** [`REPRODUCE.md`](REPRODUCE.md)
- **Status from disk:** `just status` — single source of truth
- **Numerical parity for the released ONNX:** `tests/export/test_onnx_parity.py`
- **Lane E leakage gate:** `tests/unit/test_source_rights.py`
- **Mask ordering golden test:** `tests/golden/test_no_leakage.py`
- **Frozen spec:** `bundle/glucofm_public_reproduction_blueprint.md` (the paper we
  reimplemented) + `bundle/BLUEPRINT_AMENDMENTS.md` (the A1–A7 deltas)

Wall-clock from a cold checkout to the headline number: ~10 h on an RTX 5090; ~30 h on an
RTX 3090. The full Tier-1 ablation sweep adds ~8 h on a 5090.

## 6. Licensing and intended use

This project is research software. **It is not a medical device**, and nothing it produces
should inform a decision about anyone's health.

The source code is Apache-2.0. The encoder weights are CC-BY-NC-4.0. The probe heads are
CC-BY-NC-SA-4.0, one step stricter, because eight of them are fitted on share-alike material;
the reasoning is in §4.4 and D025. Non-commercial use is the intended use and needs no
permission from anyone. Commercial use of the weights requires a licence from the maintainer,
and in the Stanford case (§2.4) would require resolving the upstream question first.

## 7. Acknowledgements

This is an independent project. It is not affiliated with, sponsored by, or endorsed by
Google, the GlucoFM authors, or any of the dataset providers. The method was reimplemented
from the publication without access to the authors' source code or weights. The PPG bridge
data (Zenodo 20577959) is CC-BY-4.0. The four public CGM cohorts are credited in
`manifests/sources/registry.yaml` under their respective licenses.

## References

1. *GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring.*
   arXiv:2605.30865v2. The method reimplemented in this work.
2. The blueprint: `bundle/glucofm_public_reproduction_blueprint.md` — frozen spec from the
   publication, with corrections and additions documented in
   `bundle/BLUEPRINT_AMENDMENTS.md`.
3. *CGM-JEPA.* `github.com/cruiseresearchgroup/CGM-JEPA`, MIT. The comparator
   implementation is a faithful port of the authors' released code (master @ 2026-05-11).

---

## Appendix A — Architecture parameters

```
model:        OpenCGMStateEvent
raw_statistics:        True        # D019: derive patch stats from raw, not normalised
normalize_targets:    False        # D019: target layer_norm disabled (no ablation activation)
streams:              "both"       # D019: state + event fused, not zeroed
learnable_sigma:      True         # paper §3.3 learnable causal Gaussian σ
use_circadian:        True         # paper §3.3 time-of-day embedding
zero_empty_patches:   False        # D020: as trained; the current repo default is True
```

Trainable parameter count: **732,593** — encoder 435,633 + predictor 132,480 + transition
heads 2 × 82,240. Within 1.7 % of the paper's reported 0.72 M. The EMA target is a frozen
copy of the encoder and contributes no trainable parameters. The released ONNX artifact is
the encoder alone: **435,633** parameters.

For the avoidance of doubt, 521,584 is the *CGM-JEPA comparator's* parameter count
(`src/opencgm_stateevent/baselines/cgm_jepa.py`), not ours.

## Appendix B — Per-task reliability (the heads)

See `findings/results_section.md` §3.4 for the cross-dataset transfer table and §7 for the
manifest of source licenses. Per-head `roc_auc`, `coverage_p05/p95/median`, `has_signal` for
all 18 fitted task-source combinations is in `artifacts/glucofm_heads.json` (the consumer
JSON shipped to HF and the web demo).

## Appendix C — The 5 findings documents

- `findings/results_section.md` — the headline numbers, ablation table, few-shot, cross-
  dataset, multiday, PPG bridge pilots, architecture, and the gap analysis.
- `findings/head_to_head.md` — paired 5-seed comparison vs CGM-JEPA with CIs.
- `findings/tier1_ablations.md` — the 30-run Tier-1 ablation matrix.
- `findings/ppg_pilot.md` — the marginal PPG teacher-student pilot (D023, A7).
- `findings/ppg_conditional.md` — the input-conditioned PPG teacher-student pilot (D023 +
  A7 extension).

## Appendix D — Standing rules

Reproduced here so this paper is self-contained.

- Never interpolate CGM. The physical observation mask is authoritative.
- Mask patches *before* normalisation, filtering, statistics. (§10.1)
- Lane E sources (NC/ND/SA) never enter a distributed checkpoint.
- Improvements live in separate checkpoints and separate tables.
- Claim "independent public-data reconstruction", never "reproduced Google's model".
- Every consequential choice is tagged `PAPER_EXACT | SOURCE_VERIFIED | INFERRED_RECONSTRUCTION
  | PROPOSED_EXTENSION`. A guess never silently becomes "the GlucoFM recipe".
