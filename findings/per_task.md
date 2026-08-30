# Per-task results, all five seeds

**Date:** 2026-08-30
**Data:** `reports/eval/per_task_5seed.csv`, built by `scripts/aggregate_per_task.py` from the
fold-level scores in `reports/eval/seed{17,29,43,71,101}_ep120_full/` and
`reports/eval/cgmjepa_seed*_full/`. Gated by `tests/unit/test_per_task_table.py`.

Each row is one health question asked of one patient cohort (CGMacros wears two sensors, so its
four questions appear twice — 18 rows, 14 distinct question-cohort pairs). Scores are ROC-AUC:
0.5 is coin-flipping, 1.0 is perfect. "This model" is GlucoFM (ours), averaged over five
independently pretrained seeds. "Clinical baseline" is a logistic model on 17 hand-computed CGM
summary features — mean, SD, coefficient of variation, time in each clinical range, MAGE, rate
of change, and the day's own data density. It sees the same day of glucose the model does,
just summarised by hand instead of learned. "CGM-JEPA" is the published comparator model, run on exactly
the same data splits. "Difference" is this model minus the clinical baseline, with a 95%
confidence interval from a paired test on the 50 shared cross-validation folds (Nadeau-Bengio
corrected for fold overlap). "Clear win?" asks whether that difference stays statistically
significant after correcting for the fact that we ran 72 such tests (Holm correction).

| Cohort | Question | Sensor | People | This model | Clinical baseline | CGM-JEPA | Difference vs clinical | Clear win? |
|---|---|---|---|---|---|---|---|---|
| CGMacros | Diabetes risk | Dexcom | 45 | 0.773 | 0.723 | 0.753 | +0.050 [-0.004, +0.105] | no |
| CGMacros | Diabetes risk | Libre | 44 | 0.784 | 0.795 | 0.805 | -0.011 [-0.058, +0.036] | no |
| CGMacros | High blood lipids | Dexcom | 45 | 0.535 | 0.406 | 0.362 | +0.130 [+0.029, +0.231] | no |
| CGMacros | High blood lipids | Libre | 44 | 0.668 | 0.501 | 0.581 | +0.168 [+0.058, +0.277] | no |
| CGMacros | Insulin resistance | Dexcom | 45 | 0.855 | 0.810 | 0.846 | +0.046 [-0.039, +0.130] | no |
| CGMacros | Insulin resistance | Libre | 44 | 0.859 | 0.822 | 0.821 | +0.037 [-0.035, +0.108] | no |
| CGMacros | Obesity | Dexcom | 45 | 0.742 | 0.711 | 0.728 | +0.031 [-0.038, +0.100] | no |
| CGMacros | Obesity | Libre | 44 | 0.639 | 0.566 | 0.574 | +0.073 [-0.031, +0.178] | no |
| Hall | Diabetes risk | — | 57 | 0.693 | 0.757 | 0.729 | -0.064 [-0.129, +0.001] | no |
| Hall | Glucotype | — | 57 | 0.875 | 0.913 | 0.910 | -0.038 [-0.071, -0.006] | no |
| Hall | High blood lipids | — | 56 | 0.373 | 0.334 | 0.519 | +0.039 [-0.112, +0.190] | no |
| Hall | Insulin resistance | — | 56 | 0.597 | 0.753 | 0.674 | -0.155 [-0.250, -0.061] | no |
| Shanghai T2DM | High blood lipids | — | 83 | 0.493 | 0.501 | 0.448 | -0.008 [-0.066, +0.050] | no |
| Shanghai T2DM | Hypoglycemia | — | 100 | 0.558 | 0.531 | 0.572 | +0.027 [-0.059, +0.113] | no |
| Shanghai T2DM | Insulin resistance | — | 68 | 0.483 | 0.447 | 0.415 | +0.036 [-0.066, +0.138] | no |
| Stanford | Beta-cell dysfunction | — | 29 | 0.665 | 0.644 | 0.567 | +0.020 [-0.058, +0.099] | no |
| Stanford | Diabetes risk | — | 29 | 0.762 | 0.724 | 0.712 | +0.038 [-0.018, +0.093] | no |
| Stanford | Insulin resistance | — | 29 | 0.707 | 0.641 | 0.562 | +0.066 [-0.002, +0.134] | no |

## What this table shows

The model is ahead of the clinical baseline on 13 of the 18 rows, and the average of the 18
differences is +0.027 — exactly the headline macro figure. But the per-row picture is noisy:
these cohorts have 29 to 100 people each, so single-task confidence intervals are wide. Only
two rows (both CGMacros high-blood-lipids) have an interval that stays above zero even before
multiple-testing correction, and after Holm correction **no row is individually significant vs
the clinical baseline — 0 of 18, not the 16 of 18 the website and paper claimed before 2026-08-30.** The
same is true within every individual seed's own Holm-corrected results. Against the weaker
raw-readings baseline, 2 of 18 rows survive correction (CGM-JEPA: 0 vs clinical — so that half
of the published sentence holds — and 1 vs raw readings).

The honest summary of the evidence is: the model's advantage is a small, consistent lift spread
across most tasks — visible and significant in the macro average over tasks, where the
per-entry difference is +0.0269 with a seed-level interval of [+0.0222, +0.0316] — not a set of
individually decisive per-task wins. The claim "significantly ahead on 16 of 18 task-cohort combinations" was retracted on 2026-08-30 and replaced everywhere with the measured 13-of-18 point estimates and 0-of-18 after Holm.
combinations after Holm correction" is not supported by the data under any criterion we could
construct (per-seed Holm, pooled five-seed Holm, uncorrected p-values, or intervals excluding
zero), and the website text should be changed.

## What this table does not show

The interval on each row measures how much the result would wobble if the cross-validation
folds were drawn differently. It does not include run-to-run variation from retraining the
model: that spread is in the `*_seed_sd` columns of the CSV, and per task it is 0.004–0.04 —
comparable to many of the differences, one more reason not to over-read any single row. (It
shrinks to about 0.004 in the macro average, which is why the macro claim is stable across
seeds.) A per-row interval also says nothing about tasks outside this benchmark. Note also that
several rows sit *below* 0.5 for every method (Hall and Dexcom high-blood-lipids, Shanghai
insulin resistance): on those tasks nothing here — model or baseline — is genuinely predictive,
and differences between methods on them mean little.

## A second discrepancy: the published absolute levels

The published headline levels (model 0.670 ± 0.003, clinical 0.652, raw 0.617, CGM-JEPA 0.652)
do not match this data. The true 18-task averages are model **0.670**, clinical **0.643**, raw
**0.607**, CGM-JEPA **0.643**. The published *differences* (+0.0269, +0.0628, +0.0000) are
exactly right. The published levels match, to within 0.001, the average over 16 tasks that
excludes `shanghai_t2dm:hyperlipidemia` and `stanford:insulin_resistance` — which suggests the
levels were computed from an earlier snapshot missing those two rows, and may also be where the
number "16" originally came from. The levels quoted in `findings/results_section.md` §1,
`findings/head_to_head.md`, and `web/lib/facts.ts` were corrected on 2026-08-30 alongside the tally.
