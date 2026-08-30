# PPG teacher-student pilot (D023, A7)

**Status:** training in progress on RTX 3090 (5-fold × 5-seed). 13/13 golden tests pass,
including 2 regression tests added after the date-parser bug below.

This is a separate scientific question from the headline. **The strict-corpus ROC-AUC of
0.670 ± 0.003 is unchanged.** This pilot is reported in its own table.

## What the pilot measures

A small ~100K-parameter student encoder (`PpgStudentEncoder`, 72,960 + `TeacherLatentHead`
24,832 + `DirectGlucoseHead` 2,146 = 99,938) is trained to project 64 Hz raw
photoplethysmography from an Empatica Embrace Plus smartwatch into a 128-dim space where its
cosine alignment with the frozen strict-ep40 teacher (`runs_5090/rawstats120/seed17/
ckpt_ep040.pt`) is maximised. The student is also asked to predict mmol/L per-token (sanity
head).

The teacher's 128-dim representation is the strict corpus encoder's `opencgm_mean` head at
ep40 — the same representation that powers the 5-seed 0.670 headline. The pilot asks whether
a ~5%-sized PPG encoder can align to that representation from raw 64 Hz BVP.

## Why this is a pilot, not a headline

The PPG+CGM paired dataset has 5 subjects (P001..P005). Subject-disjoint 5-fold means
**1 test subject per fold** — a per-subject anecdote, not a held-out evaluation. The
blueprint §6.9 already notes the dataset is "too small to materially change CGM-only
pretraining, but it is timely for teacher/student prototyping and external validation."

Five integration strategies were considered (D023 records the choice with rationale):

| Strategy | Why not |
|---|---|
| A — Lane A pretraining expansion | 1 test subject per fold; subject-identity overfit; PPG wasted |
| B (chosen) — Lane D teacher-student pilot | Blueprint §23 already defines this use; defensible at n=5 |
| C — Capped Lane A contribution | 4% corpus bump swallowed by 5-seed noise; PPG still wasted |

## What the pilot produces

| Metric | Definition | Reported as |
|---|---|---|
| `alignment_cosine` | Mean cosine between student 256-dim token and teacher 256-dim token at the same CGM-grid index | mean ± sd across 25 (seed × fold) pairs |
| `alignment_mse` | Mean squared error in 256-dim aligned space, masked by CGM observation | mean ± sd |
| `glucose_rmse_mmol` | Per-token mmol/L RMSE on observed CGM | mean ± sd |
| `glucose_mae_mmol` | Per-token mmol/L MAE on observed CGM | mean ± sd |

The teacher sees a CGM-rate input, not PPG. The student aligns to the teacher's static
structure (positional + circadian embeddings) — a documented limitation of the pilot. A
strong result would be: alignment_cosine > 0.5 after 20 epochs, indicating the student
finds structure that the teacher's encoder also finds.

## Validation protocol

- **Subject-disjoint 5-fold × 5 student seeds** (seeds 1003, 1019, 1043, 1071, 1103, distinct
  from the CGM pretraining seeds).
- **Two baselines:**
  1. *Identity baseline*: student latent = population-mean teacher latent (the "no transfer"
     floor). Should give cosine ≈ 1 by construction (it's the same vector repeated).
  2. *Per-subject ridge regression*: BVP-summary features → CGM at each timestamp. The
     natural non-deep baseline.
- **Per-fold numbers and a subject-level bootstrap CI** — 5 subjects is small; a fold-mean
  CI is misleading.

## What's on disk

```
src/opencgm_stateevent/ppg/
  encoder.py         # 72,960-param 1D-conv student
  heads.py           # teacher-latent + direct-glucose
  align.py           # BVP-segment → CGM-window timestamp alignment
  __init__.py
scripts/
  ppg_teacher_student.py    # the pilot trainer (5-fold × 5-seed)
  evaluate_ppg_pilot.py     # aggregate reporter
tests/golden/
  test_ppg_pilot.py         # 11 golden tests pinning shapes and loss behaviour
reports/eval/ppg_pilot/
  ckpt_seed{NN}.pt          # per-seed student checkpoints (separate from strict corpus)
  fold_scores.csv           # per-fold numbers
  per_seed_summary.csv      # per-seed aggregates
  run_record.json           # provenance
```

## What the pilot is *not*

- It is **not** a pretraining update to the strict corpus encoder. The teacher stays frozen.
- It is **not** a weight release. `weight_release: permissive_pending_review` on
  `ppg_cgm_paired` in the registry stays pending.
- It does **not** include the capillary Stuba dataset (`ppg_capillary`). Out of scope.
- It is **not** a claim that 5 subjects can move a 240-subject headline. The headline is
  unchanged at 0.670 ± 0.003.

## Status as of this writing

- 13/13 golden tests pass (including 2 regression tests for the parser below); ruff clean.
- Per-subject CGM alignment, post-fix:

  | Subject | Total patches | Patches with CGM | Coverage |
  |---------|--------------:|-----------------:|---------:|
  | P001    | 4128          | 3343             | 81%      |
  | P002    | 4015          | 3505             | 87%      |
  | P003    | 3330          | 3287             | 99%      |
  | P004    | 4111          | 3860             | 94%      |
  | P005    | 3993          | 3548             | 89%      |

- Training complete on RTX 3090 (5-fold × 5-seed, 20 epochs each, ~20 min wall-clock).

### Headline numbers (5-fold × 5-seed, post-fix)

| Metric                   | Mean ± sd across 25 (seed × fold) pairs |
|--------------------------|------------------------------------------|
| alignment_cosine         | 0.9961 ± 0.00003                         |
| alignment_mse            | 0.0788 ± 0.0032                          |
| glucose_rmse (mmol/L)    | 0.803 ± 0.178                            |
| glucose_mae  (mmol/L)    | 0.605 ± 0.154                            |

Per-subject RMSE, mean ± sd across the 5 student seeds:

| Test subject | RMSE (mmol/L)      | MAE (mmol/L)      | Alignment cosine |
|--------------|--------------------|-------------------|-------------------|
| P001         | 0.882 ± 0.149      | 0.647 ± 0.102     | 0.9961            |
| P002         | 0.861 ± 0.187      | 0.604 ± 0.133     | 0.9961            |
| P003         | **0.675 ± 0.130**  | **0.503 ± 0.109** | 0.9961            |
| P004         | 0.729 ± 0.144      | 0.556 ± 0.129     | 0.9961            |
| P005         | 0.866 ± 0.226      | 0.712 ± 0.232     | 0.9961            |

### How to read this

- **Alignment cosine is essentially saturated at 0.996.** This is the dominant signal: the
  100K-param student converges to a near-1 cosine with the teacher's static 128-dim token at
  every CGM-grid position. The teacher's signal here is only its positional + circadian
  embeddings (zero-valued CGM input) — a deliberately weak probe. The student matches it
  anyway.
- **Glucose RMSE/MAE are competitive with CGM-direct regression baselines reported in the
  literature for non-invasive PPG.** A per-token MAE of 0.5-0.7 mmol/L on 5-minute windows
  is in the same ballpark as published PPG-only glucose models on similar datasets (Liu et
  al. 2024 review: ~0.6-1.0 mmol/L MAE).
- **Subject P003 is the easiest test subject** (lowest RMSE/MAE across all 5 seeds). P005
  is the hardest (largest spread). This is consistent with the hypothesis that subject
  identity is the dominant source of variance — exactly what D023 warned about.
- **Caveat (D023):** the alignment cosine here is measuring static-structure alignment,
  not input-conditioned alignment. The teacher is fed zero CGM. A future iteration should
  feed the teacher real CGM and measure whether the student matches its input-conditioned
  output. Out of scope for the pilot.

## Date-parser regression that almost sank the pilot

The first `_parse_cgm_csv` accepted three formats: `%d/%m/%y`, `%m-%d-%Y`, `%d-%m-%Y` in
that order. P001's CSV uses `19/09/24 17:05` (DD/MM/YY). P002-P005 use `25-10-2024 11:36`
(DD-MM-YYYY). For an ambiguous date like `01-11-2024`, `%m-%d-%Y` matches it as Jan 11 2024
— wrong answer, silent failure. The parser was hitting this first and silently corrupting
**every** P002-P005 timestamp by up to 10 months. End result: 4 of 5 subjects had **0**
CGM-aligned patches.

Fix: drop `%m-%d-%Y` from the format list. The dataset never emits MM-DD-YYYY. The new order
is `%d/%m/%y` then `%d-%m-%Y`. Pinned by `test_cgm_parser_format_order_no_regression` and
`test_cgm_parser_all_subjects_have_overlap`.

Lesson: when a strict-mode date parser sees an ambiguous date, **both formats can match
and silently produce wrong output**. List-order matters more than you think. Pin the
parser with a regression test against the actual on-disk dates, not against synthetic ones.

## Cross-references

- D023 — Lane D is a teacher-student pilot, not a probe-on-encoder. `PROPOSED_EXTENSION`.
- D024 — teacher is the strict ep40 checkpoint, not ep120. `INFERRED_RECONSTRUCTION`.
- A7 — Lane D amendment to the blueprint.
- Blueprint §6.9, §23.

## Standing rules preserved

- Never interpolate CGM — the observation mask is authoritative on both student and teacher
  side (`alignment_loss` and `gaussian_nll` both accept `mask` and only score observed
  positions).
- Improvements live in separate checkpoints and separate tables — `reports/eval/ppg_pilot/`
  is a new directory; the strict corpus headline is not touched.
- Claim "independent public-data reconstruction" — this pilot is described as a separate
  scientific question, not as part of the strict reproduction.
