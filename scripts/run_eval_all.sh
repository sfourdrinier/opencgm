#!/usr/bin/env bash
# Full evaluation suite — every script that produces a number we report.
#
#   bash scripts/run_eval_all.sh
#
# Runs, in order:
#   1. 5-seed headline (5× evaluate.py)
#   2. 5-seed CGM-JEPA comparator (5× evaluate_cgm_jepa.py)
#   3. Paired 5-seed head-to-head aggregation
#   4. Few-shot (k = 1, 5, 10, 20 per class)
#   5. Cross-dataset transfer (38 cohort-pair transfers)
#   6. Multiday pooling (n = 1, 2, 3, 5, 7 days)
#   7. PPGR (paper §4.3)
#   8. Permutation test for the headline
#   9. PPG teacher-student pilot (5-fold x 5-seed, ~20 min on 3090)
#
# Each step skips if its output CSV already exists.
# Wall-clock on RTX 3090: ~80 min. On RTX 5090: ~40 min.
# Plus step 9 (~20 min on 3090) if the PPG data zip is on disk.

set -uo pipefail
cd "$(dirname "$0")/.."

# Default checkpoint for the single-seed extensions
EP_CKPT=${EP_CKPT:-runs_5090/rawstats120/ckpt_ep040.pt}

echo "=== 1/9 headline (5 seeds × 120 ep) ==="
for seed in 17 29 43 71 101; do
    bash scripts/evaluate_seed.sh "$seed" 120 full
done

echo "=== 2/10 CGM-JEPA comparator (5 seeds) ==="
for seed in 17 29 43 71 101; do
    bash scripts/evaluate_cgmjepa_seed.sh "$seed"
done

echo "=== 3/10 head-to-head aggregation ==="
uv run python scripts/aggregate_cgmjepa_vs_glucofm.py

echo "=== 4/10 few-shot (k = 1, 5, 10, 20) ==="
uv run python scripts/evaluate_few_shot.py --checkpoint "$EP_CKPT"

echo "=== 5/10 cross-dataset transfer ==="
uv run python scripts/evaluate_cross_dataset.py --checkpoint "$EP_CKPT"

echo "=== 6/10 multiday (n = 1, 2, 3, 5, 7 days) ==="
uv run python scripts/evaluate_multiday.py --checkpoint "$EP_CKPT"

echo "=== 7/10 PPGR (paper §4.3) ==="
uv run python scripts/ppgr.py --checkpoint "$EP_CKPT"

echo "=== 8/10 permutation test ==="
uv run python scripts/permutation_test.py

# Step 9: PPG teacher-student pilot (D023, A7). Skip if no PPG data zip on disk.
if [ -d "data/raw/ppg_cgm_paired_zenodo_20577959" ]; then
    echo "=== 9/10 PPG teacher-student pilot (5-fold x 5-seed, marginal) ==="
    if [ ! -f reports/eval/ppg_pilot/fold_scores.csv ]; then
        uv run python scripts/ppg_teacher_student.py \
            --teacher-ckpt "$EP_CKPT" \
            --device cuda
    fi
    if [ -f reports/eval/ppg_pilot/fold_scores.csv ] && \
       [ ! -f reports/eval/ppg_pilot/aggregate/aggregate.csv ]; then
        uv run python scripts/evaluate_ppg_pilot.py
    fi

    echo "=== 10/10 PPG input-conditioned teacher (A7 extension, 5-fold x 5-seed) ==="
    if [ ! -f reports/eval/ppg_pilot_conditional/fold_scores.csv ]; then
        uv run python scripts/ppg_teacher_student_conditional.py \
            --teacher-ckpt "$EP_CKPT" \
            --teacher-targets-cache artifacts/ppg_teacher_targets.npz \
            --device cuda
    fi
else
    echo "=== 9-10/10 PPG pilots — SKIPPED (no PPG data zip) ==="
fi

echo
echo "all done. CSVs in reports/eval/. Findings in findings/."
