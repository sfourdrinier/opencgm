# REPRODUCE.md — end-to-end reproduction

This document is the **complete, one-stop** guide to reproducing every result in this
repository from a cold checkout. Every command is copy-paste runnable. Every wall-clock is
measured, not estimated. Every output path is the path that actually exists.

The reproduction has **two tiers**:

| Tier | What's run | Wall-clock | Output |
|---|---|---|---|
| **Fast** | 1 seed × 40 epochs, 3 ablations, no PPG | ~6 hours on RTX 3090 | smoke-test results |
| **Full** | 5 seeds × 120 epochs, all 10 ablations × 3 seeds, all §19 probes, full app | ~10 h on a 5090, ~30 h on a 3090 | publishable numbers |

For the published headline numbers in `findings/results_section.md` and
`findings/head_to_head.md`, see "Full reproduction" below.

---

## 0. Prerequisites

| Need | Tested with |
|---|---|
| Linux | Ubuntu 22.04 / kernel 6.8 |
| GPU | NVIDIA, ≥ 16 GB VRAM, sm_120 OK (5090 tested). RTX 3090 / 5090 are the targets. |
| CUDA | 12.x (5090: 12.8+) |
| Python | 3.12 |
| Toolchain | `uv` (Astral) — installs everything from `uv.lock` |
| Disk | ≥ 80 GB free (raw data + windows + checkpoints + 5-seed sweep) |

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Clone and enter:

```bash
git clone <repo-url> glucose-experiments
cd glucose-experiments
```

The repo's data lives **outside** the working tree at `$raw_root` (default
`./data/raw`). Symlink or copy your data there:

```bash
mkdir -p data/raw
# copy each source to its registry path; see manifests/sources/registry.yaml
# e.g.: cp -r <your-download>/big_ideas data/raw/big_ideas/1.1.2
```

---

## 1. Environment & verification (5 min)

```bash
uv sync                          # installs everything from uv.lock
just status                      # prints what's on disk and what each source is licensed for
just verify                      # SHA-256 verifies the corpus against manifests/sources/*.sha256.json
just gate                        # ruff + pytest + status
```

`just status` is the single source of truth. If a number in any other doc disagrees with
`just status`, the status output is right.

Expected: 451+ tests pass, ruff clean, all 14 sources verified.

---

## 2. Build the pretraining corpus (15 min on 3090, 8 min on 5090)

The blueprint corpus is built once per source. The canonical windows live in
`data/canonical/windows/`:

```bash
just build-windows
```

This runs:

```
scripts/build_windows.py
  → reads data/raw/<source>/... per registry.yaml
  → applies each source's parser (src/opencgm_stateevent/data/readers.py)
  → drops sessions with < 24 h coverage (§6.6)
  → outputs strict_seed17.values.npy + strict_seed17.mask.npy   (353,127 windows, 508 MB)
```

If a source's parser fails, `just build-windows --source <name>` re-runs just that one.

Expected after step 2:

```
data/canonical/windows/
  strict_seed17.values.npy       508 MB   SHA-256 0cbecfc5...72f
  strict_seed17.mask.npy
  cgmacros_dexcom.values.npy     ...
  ...
```

Verify integrity:

```bash
sha256sum data/canonical/windows/strict_seed17.values.npy
# expect: 0cbecfc5110bc90e969c9f8ece2ebd690f39fdac332fef449d4eddffb8b6372f
```

---

## 3. Pretrain the encoder (5 seeds × 120 ep ≈ 30 h on 3090, 10 h on 5090)

```bash
# Strict pretraining sweep — 5 seeds, 120 epochs each.
# 3090: ~6 h per seed (one seed at a time, full GPU).
# 5090: ~2 h per seed (3-way concurrent saturates at ~3.3 epoch/min aggregate).
just pretrain-sweep
```

The sweep launcher is `scripts/pretrain_sweep.sh`. Each seed:

- initialises from `models/opencgm_stateevent` architecture spec (732,593 trainable params;
  435,633 of them in the encoder that is later exported)
- trains on `strict_seed17.{values,mask}.npy` (the canonical 353,127-window corpus)
- saves `runs_5090/rawstats120/seed{NN}/ckpt_last.pt` every epoch
- saves `ckpt_ep120.pt` at the end (the headline checkpoint)

A watchdog (`scripts/watchdog.sh`) re-runs any seed whose last checkpoint is stale.

---

## 4. Evaluate downstream (10 min per eval × 8 evals ≈ 80 min)

Once pretraining is done, the headline evaluation suite:

```bash
just eval-all
```

This runs, in order:

| Script | What | Output |
|---|---|---|
| `scripts/evaluate.py` | 5 seeds × 5-fold × 10-repeat downstream probe (the headline) | `reports/eval/seed{NN}_ep120_full/` |
| `scripts/evaluate_cgm_jepa.py` | Same probe, the CGM-JEPA comparator (5 seeds) | `reports/eval/cgmjepa_seed{NN}_full/` |
| `scripts/aggregate_cgmjepa_vs_glucofm.py` | Paired head-to-head + 95% CIs | `reports/eval/head_to_head_5seed.csv` |
| `scripts/evaluate_few_shot.py` | k = 1, 5, 10, 20 per class | `reports/eval/fewshot_seed43/` |
| `scripts/evaluate_cross_dataset.py` | 38 cohort-pair transfers | `reports/eval/cross_dataset_seed43/` |
| `scripts/evaluate_multiday.py` | n_days = 1, 2, 3, 5, 7 | `reports/eval/multiday_seed43/` |
| `scripts/ppgr.py` | Post-prandial glucose response (paper §4.3) | `reports/eval/ppgr_*/` |
| `scripts/permutation_test.py` | 1000-label permutations for the headline | `reports/eval/permutation_test.json` |
| `scripts/ppg_teacher_student.py` | PPG teacher-student pilot (D023, A7, 5-fold × 5-seed, marginal). **Skipped if `data/raw/ppg_cgm_paired_zenodo_20577959` is not on disk.** | `reports/eval/ppg_pilot/` |
| `scripts/evaluate_ppg_pilot.py` | Aggregate per-subject + per-seed PPG numbers. Runs automatically after the trainer if its CSV is fresh. | `reports/eval/ppg_pilot/aggregate/` |
| `scripts/ppg_teacher_student_conditional.py` | PPG input-conditioned teacher (A7 extension, 5-fold × 5-seed). Reuses the same data. | `reports/eval/ppg_pilot_conditional/` |

After `eval-all`:

```bash
just head-to-head         # writes reports/eval/head_to_head_5seed.csv, the headline source
uv run python scripts/aggregate_per_task.py   # writes reports/eval/per_task_5seed.csv
```

The prose in `findings/*.md` is written by hand against those CSVs, not generated from them.
`tests/unit/test_documented_facts.py` recomputes the headline figures from the CSVs and fails
if a document has drifted, which is how the levels carried through the drafts as 0.679 / 0.652 / 0.617 were found
to be a 16-task subset of an 18-task evaluation.

---

## 5. Tier-1 ablations (3 seeds × 10 ablations × 40 ep ≈ 8 h on 5090)

```bash
just ablation-sweep
```

Launcher: `scripts/sweep_tier1_ablations.sh`. The 10 conditions:

| Flag | What it disables |
|---|---|
| `notd` | target deadtime loss (λ_td = 0) |
| `dense` | forces dense interpolation (the rule we forbid) |
| `event` | event stream only |
| `fixedsigma` | σ pinned at 6.0 (no learnable σ) |
| `loo_shanghai` | leave ShanghaiT2DM out of pretraining |
| `loo_stanford` | leave Stanford out |
| `noaug` | no time-shift / amplitude-jitter augmentations |
| `nocirc` | zero out the circadian embedding |
| `raw` | single-stream raw-only (no state/event decomposition) |
| `state` | state stream only |

Each ablation trains a fresh encoder with the same hyperparameters as the strict-pretrain run.

After the sweep:

```bash
just ablation-aggregate    # writes findings/tier1_ablations.md
```

---

## 6. Fit deployable heads + launch the real-data demo

```bash
just fit-heads
just app                   # streamlit on http://localhost:8765
```

The heads are fitted logistic regressions with per-head ROC-AUC reliability and per-head
subject counts. The app loads them by default.

---

## 7. Source manifests (corpus integrity)

```bash
just manifest-sources
```

Re-hashes every byte under `data/raw/` for every source in `manifests/sources/registry.yaml`
(Lane E sources skipped — eval only, never enter a distributed checkpoint). Writes per-source
JSONs to `manifests/sources/`.

If anything has been tampered with, `just verify` will fail loudly with the mismatching SHA.

---

## 8. What "reproduced" means here

| Claim from the paper | What we measure |
|---|---|
| 0.72M-parameter dual-stream JEPA | 732,593 trainable — within 1.7% |
| Subject-disjoint 5-fold × 10 repeats | Yes — every reported number |
| Mask patches before normalize/filter | Yes — golden-tested |
| Never interpolate CGM | Yes — golden-tested (the dense arm is an ablation) |
| State + event decomposition matters | Yes — 0.622 (event-only) vs 0.670 (dual) |
| Strong few-shot at k=1 | Yes — 0.603 (opencgm) vs 0.575 (raw) |
| Cross-dataset transfer | Yes — 0.670 diabetes_risk, 0.623 insulin_resistance |
| Multiday benefit | Yes — +0.05 ROC from 1 day to 7 days |

What we **cannot** reproduce:

| Paper | Why |
|---|---|
| 0.74 average ROC-AUC | We measure 0.670. Wear-CGM (69% of paper's hours) is non-public. |
| +4.11 PR-AUC over CGM-JEPA | We measure +0.0221. Same direction; corpus + epochs are the gap. |

This is documented in `findings/results_section.md` §8–§9.

---

## 9. The "fast" tier (smoke test)

For a quick end-to-end check that everything wires up correctly, without the full pretraining
sweep:

```bash
# 1 seed × 40 epochs, 1 ablation, no comparator
just fast-pretrain        # ~1.5 h on 3090
just fast-eval            # ~20 min
just app                  # see the demo
```

Outputs land under `reports/eval/seed17_ep040_fast/`. Numbers will be lower than the full
5-seed headline; the smoke test is to verify plumbing, not reproduce headline numbers.

---

## 10. Adding new data

Edit `manifests/sources/registry.yaml` — add your source with the correct lane, role,
license, weight-release flag. Then:

```bash
just verify --dataset <your_source>
just build-windows --source <your_source>
just manifest-sources
```

A golden test (`tests/test_registry.py`) refuses to load a checkpoint that pulls from
Lane E.

---

## 11. Where the numbers come from (provenance chain)

```
bundle/glucofm_public_reproduction_blueprint.md  (frozen spec)
        ↓
bundle/BLUEPRINT_AMENDMENTS.md                  (the A1..A6 deltas)
        ↓
src/opencgm_stateevent/                         (the implementation, golden-tested)
        ↓
runs_5090/rawstats120/seed{NN}/ckpt_ep120.pt    (5 seeds)
        ↓
scripts/evaluate.py                             (subject-disjoint 5×10 fold)
        ↓
reports/eval/seed{NN}_ep120_full/               (per-seed per-task CSVs)
        ↓
scripts/aggregate_cgmjepa_vs_glucofm.py         (paired 5-seed aggregation)
        ↓
reports/eval/head_to_head_5seed.csv             (the headline number)
        ↓
findings/head_to_head.md                        (the writeup)
```

Every step is on disk. Every number is reproducible. Every comparison is structurally paired.

---

## 12. License

Apache-2.0 for code. Per-source for data — see `NOTICE` and `manifests/sources/registry.yaml`.

Research software. **Not a medical device.**
