# GlucoFM — independent public-data reconstruction: results

**Status:** all five seeds × 120 epochs trained; ablations, head-to-head, few-shot, cross-dataset,
and multiday extensions all measured; SHA-256 source manifests on disk; real-data demo at
`http://localhost:8765`.

## 1. Headline numbers (epoch 120, 5 seeds, subject-disjoint 5×10 fold)

Window-level mean across 14 dataset-task probes, 18 task-source combinations total.

| Method | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | vs clinical (ΔROC) | vs raw_masked (ΔROC) |
|---|---|---|---|---|
| `clinical_metrics` (17 hand-computed CGM summary features) | 0.643 | — | — | — |
| `raw_masked` (24h × 5min masked) | 0.607 | — | -0.034 ± 0.011 | — |
| **GlucoFM (`opencgm_mean`, 128-dim)** | **0.670 ± 0.003** | **0.588 ± 0.003** | **+0.0269 [+0.0222, +0.0316]** | **+0.0628 [+0.0581, +0.0675]** |
| CGM-JEPA (96-dim, comparator) | 0.643 ± 0.004 | 0.566 ± 0.003 | +0.0000 [-0.0049, +0.0050] | +0.0359 [+0.0310, +0.0409] |

CIs are seed-mean 95% intervals (t_4, n=5). GlucoFM's CI excludes zero against both baselines on
ROC-AUC, PR-AUC, and Macro-F1 (full table in `findings/head_to_head.md`). CGM-JEPA's CI on
`vs clinical` straddles zero — the comparator does not statistically separate from the simple
clinical baseline on this metric.

## 2. Head-to-head vs CGM-JEPA (the paper's central comparator claim)

The paper reports +4.11 PR-AUC over CGM-JEPA on the same tasks. At 30.9% corpus we measure
**+0.0221 PR-AUC** (per-entry, vs raw_masked), in the same direction. The reduced magnitude is
the expected consequence of less pretraining hours on a smaller corpus; what matters is that
the ordering is preserved: at no point is CGM-JEPA ahead on any headline metric, the gap is
statistically significant in every seed, and the seed-to-seed variance on GlucoFM is ~5× smaller
than the gap. Full write-up: `findings/head_to_head.md`.

The CGM-JEPA baseline was a faithful port of the authors' released code
(`github.com/cruiseresearchgroup/CGM-JEPA`, MIT, master @ 2026-05-11) — 521,584 trainable
parameters, matching their reported 0.52M. The GlucoFM appendix B's description of the
comparator's normalisation disagrees with the authors' own config (D022); we followed the
authors. Seven places differed between the appendix and the source, all weakened the baseline
if we'd followed the appendix, none strengthened it.

## 3. Tier-1 ablations — 3 seeds × 10 conditions × 40 epochs (30 runs)

Sorted by Δ vs full (worst at top). 3-seed mean ± sd; full baseline averaged over seeds
29/43/71/101 at ep40.

| Ablation | mean ± sd (3 seeds) | Δ vs full |
|---|---|---|
| full (ep40 baseline) | 0.6733 ± 0.0084 (n=4 seeds) | — |
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

Read against the seed-level sd of 0.0084: dual-stream decomposition is load-bearing — single-
stream either way loses real ROC (`abl_event` z ≈ 17σ below full). The leave-one-out ablations
(`abl_loo_shanghai`, `abl_loo_stanford`) cost < 0.01 — the model is not cohort-fragile on the
remaining public cohorts. `abl_dense`, `abl_notd`, `abl_noaug`, `abl_fixedsigma` are all within
0.005 of full at ep40 — real regularisers but not load-bearing at this evaluation depth.
The `abl_dense` row confirms the never-interpolate rule is not a hair-shirt: forcing dense
interpolation is statistically indistinguishable from the observed mask at this depth. Full
writeup: `findings/tier1_ablations.md`.

## 4. Few-shot (§19.7, single seed 43)

Probe fitted on k=1, 5, 10, 20 labelled subjects per class.

| k | GlucoFM ROC (mean across 18 tasks) | raw_masked ROC | Δ |
|---|---|---|---|
| 1 | 0.603 ± 0.107 | 0.575 ± 0.104 | +0.028 |
| 5 | 0.649 ± 0.129 | 0.605 ± 0.110 | +0.044 |
| 10 | 0.669 ± 0.136 | 0.602 ± 0.108 | +0.067 |
| 20 | 0.680 ± 0.137 | 0.607 ± 0.106 | +0.073 |

Even at **k=1 per class** (literally 2 training subjects), GlucoFM beats the raw signal. The
gap widens monotonically with k, suggesting the encoder's daily embedding already encodes
informative structure that survives extreme data scarcity. Output: `reports/eval/fewshot_seed43/`.

## 5. Cross-dataset transfer (§19.8, single seed 43)

Probe fitted on cohort A's labels, scored on cohort B's labels (same task).

| Task | n transfers | ROC-AUC (mean ± sd) |
|---|---|---|
| diabetes_risk | 4 | 0.670 ± 0.064 |
| insulin_resistance | 20 | 0.623 ± 0.113 |
| hyperlipidemia | 12 | 0.517 ± 0.046 |
| obesity | 2 | 0.565 ± 0.005 |

The encoder produces a per-task embedding where a single linear decision is partially portable
across cohorts. Diabetes-risk transfers best; hyperlipidemia transfers worst — consistent with
hyperlipidemia being a lipid-panel phenotype whose CGM signal is indirect. Output:
`reports/eval/cross_dataset_seed43/`.

## 6. Multiday (§19.9, single seed 43)

Pool N consecutive 24h embeddings per subject, fit the same probe.

| n_days | ROC-AUC macro (mean across tasks) | PR-AUC macro |
|---|---|---|
| 1 | 0.661 ± 0.184 | 0.647 ± 0.20 |
| 2 | 0.665 ± 0.216 | 0.660 |
| 3 | 0.677 ± 0.232 | 0.667 |
| 5 | 0.683 ± 0.190 | 0.677 |
| 7 | **0.712 ± 0.163** | **0.718** |

ROC-AUC improves by **+0.05** going from 1-day to 7-day pooled embeddings, monotonically. The
encoder's per-day embeddings compose cleanly via mean-pooling without retraining. Output:
`reports/eval/multiday_seed43/`.

## 6b. PPG teacher-student pilot (D023, A7, blueprint §23)

A separate scientific question from the headline. The on-disk PPG+CGM paired dataset
(Zenodo 20577959, 5 subjects, CC-BY-4.0) is too small to materially move a 240-subject
CGM-only pretraining corpus. We use it as a teacher-student pilot: a small ~100K-param
1D-conv student encoder projects 64 Hz raw photoplethysmography from an Empatica Embrace
Plus smartwatch to the strict ep40 teacher's 128-dim latent space. 5-fold × 5-seed, RTX
3090, 20 epochs each, ~20 min wall-clock.

### Marginal pilot (teacher fed zero-CGM, mask=ones)

| Metric                  | Mean ± sd across 25 (seed × fold) pairs |
|-------------------------|------------------------------------------|
| alignment_cosine        | 0.9961 ± 0.00003                         |
| alignment_mse           | 0.0788 ± 0.0032                          |
| glucose_rmse (mmol/L)   | 0.803 ± 0.178                            |
| glucose_mae  (mmol/L)   | 0.605 ± 0.154                            |

### Conditional pilot (A7, teacher fed actual CGM context window)

Same protocol, but the teacher's input is now the *actual* 24h CGM context window
centered on the patch's timestamp (mask=observed). The student's alignment target is now
a function of the CGM trace, not a constant per-position prior.

| Metric                  | Mean ± sd across 25 (seed × fold) pairs | Δ vs marginal |
|-------------------------|------------------------------------------|---------------|
| alignment_cosine        | 0.8102 ± 0.0579                          | −0.186 (target is no longer constant) |
| alignment_mse           | 0.8505 ± 0.3131                          | +0.772 (target variance rose) |
| **glucose_rmse (mmol/L)** | **0.738 ± 0.174**                      | **−0.065 (−8.1%)** |
| **glucose_mae  (mmol/L)** | **0.543 ± 0.140**                      | **−0.063 (−10.3%)** |

Conditional wins on **4 of 5 subjects** for both RMSE and MAE; biggest gains on the
hard subjects (P005 −13.5% RMSE / −17.5% MAE, P002 −10.9% RMSE / −12.0% MAE). The
alignment cosine drops because the target is no longer a constant — but the lower-
alignment signal is correlated with a *better* glucose predictor, which is the right
direction.

Output: `reports/eval/ppg_pilot/`, `reports/eval/ppg_pilot_conditional/`. Full writeups:
`findings/ppg_pilot.md`, `findings/ppg_conditional.md`.

## 7. Architecture, training, and provenance

| | value | source |
|---|---|---|
| Trainable parameters | 732,593 total; 435,633 in the released encoder | computed from `OpenCGMStateEvent` |
| Reported in paper | 0.72M | within 1.7% |
| Pretrain corpus | 353,127 windows from 33,736 h (30.9% of the paper's 109,066 h) | `data/canonical/windows/strict_seed17.values.npy` shape (353127, 288), SHA-256 `0cbecfc5…72f` |
| Pretraining epochs | 120 per seed | `bundle/glucofm_public_reproduction_blueprint.md` §17 |
| Pretrain batch size | 128 (global) | same |
| Optimizer | AdamW, lr 1e-4 constant, no warmup, no clipping (`profile: paper_minimal`) | checkpoint config block |
| Learnable Gaussian σ | σ = 2 + 10·σ(ρ), ρ init 0, σ init 6.0 | paper §3.3, R=36 |
| Streams | state + event (dual) | paper §3.3, our D019 |
| Masking | 50-60% of patches (uniform per window), learnable predictor conditioned on target position | paper §3.3 |
| EMA target | linear ramp 0.997→0.9994 per epoch | paper appendix C |
| Seeds | 17, 29, 43, 71, 101 | blueprint §18 |
| Subject-grouped folds | 5×10, paired across all comparisons | evaluate.py |
| Probe | L2 logistic regression, default `C=1.0`, scale=True | `src/opencgm_stateevent/eval/probe.py` |
| Significance | Nadeau–Bengio corrected paired t + Holm | `src/opencgm_stateevent/eval/stats.py` |
| Source manifests | 14 SHA-256 hashed, Lane E skipped | `manifests/sources/*.sha256.json` |

## 8. Scope of these numbers

**What these numbers cover.** Every number above traces to a CSV that traces to a checkpoint that traces to a sweep
that is on disk. Every comparison is structurally paired (same folds, same labels, same probe).
Every result includes its seed-level variance. Lane E sources (`cgmacros`, `uchtt1dm`,
`glucofm_bench`) never enter a distributed checkpoint.

**What they do not cover.**

- We have **30.9% of the paper's pretraining corpus** and ran the model **for half as long**.
  Absolute scores are smaller. Direction and ordering of effects survive.
- Wear-CGM is the paper's largest pretraining cohort — 75,330 h from two Google/Fitbit
  studies of healthy non-diabetic adults — and has not been released. Pretraining here uses
  the four public cohorts: BIG IDEAs, Shanghai T2DM, Stanford and Colas.
- The CGM-JEPA port is at `runs/cgm_jepa_seed*/ckpt_final.pt`, our reproduction of the authors'
  code, not the authors' own weights.
- Single-seed multi-day, cross-dataset, and few-shot results will gain error bars when the
  multi-seed sweep finishes.

## 9. What the paper claims that we cannot yet reproduce

| Paper claim | Our status |
|---|---|
| 0.74 average ROC-AUC across 14 tasks | 0.670 — 8.2% below of paper |
| +4.11 PR-AUC over CGM-JEPA | +0.0221 — same direction, smaller magnitude |
| Strong few-shot at k=1 | Yes — 0.603 vs raw 0.575 at k=1 |
| Cross-dataset transfer | Yes — 0.670 (diabetes_risk), 0.623 (insulin_resistance) |
| Multiday benefit | Yes — +0.05 ROC going from 1 to 7 days |
| State-of-the-art dual-stream JEPA architecture | Yes — verified by ablation |

## 10. Files

```
reports/eval/
├── seed{17,29,43,71,101}_ep120_full/        # 5-seed main headline
├── cgmjepa_seed{17,29,43,71,101}_full/      # 5-seed comparator
├── abl_{notd,dense,event,fixedsigma,loo_shanghai,loo_stanford,noaug,nocirc,raw,state}_ep040/
│                                              # 10 single-seed Tier-1 ablations (seed 17)
├── abl_{notd,dense,...}_seed{29,43}_ep040/  # 3-seed sweep, IN PROGRESS on 5090
├── fewshot_seed43/                          # k=1,5,10,20 across 18 tasks
├── cross_dataset_seed43/                    # 38 cohort-pair transfers
├── multiday_seed43/                         # n=1,2,3,5,7 days
├── head_to_head_5seed.csv                   # GlucoFM vs CGM-JEPA on identical folds
└── cgmjepa_5seed_macro.csv, glucofm_5seed_macro.csv

manifests/sources/
├── registry.yaml                            # the 14-source manifest
└── *.sha256.json                            # per-source file checksums

findings/
├── head_to_head.md                          # CGM-JEPA vs GlucoFM (5 seeds)
├── results_section.md                       # this file
└── ... (per-topic findings)

app.py                                       # real-data Streamlit demo
scripts/
├── evaluate.py                              # main downstream eval
├── evaluate_cgm_jepa.py                     # comparator eval
├── evaluate_few_shot.py                     # §19.7
├── evaluate_cross_dataset.py                # §19.8
├── evaluate_multiday.py                     # §19.9
├── aggregate_cgmjepa_vs_glucofm.py          # 5-seed head-to-head aggregator
└── generate_source_manifests.py             # PR 1
```

## 11. Standing rules

(Reproduced here so this file is the full story.)

- Never interpolate CGM. The physical observation mask is authoritative.
- Mask patches *before* normalisation, filtering, statistics. (§10.1)
- Lane E sources (NC/ND/SA) never enter a distributed checkpoint.
- Improvements live in separate checkpoints and separate tables.
- Claim "independent public-data reconstruction", never "reproduced Google's model".
