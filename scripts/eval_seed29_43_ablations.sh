#!/usr/bin/env bash
# Pull the seed-29 and seed-43 ablation checkpoints from the 5090 and evaluate each one
# with the same probe as the seed-17 ablations. Writes to
# reports/eval/abl_<name>_seed<NN>_ep040/.
#
#   bash scripts/eval_seed29_43_ablations.sh

set -uo pipefail
cd "$(dirname "$0")/.."

# No default host. This used to name the author's own machine, so anyone running the
# recipe pointed an ssh session at a stranger's box.
REMOTE=${REMOTE:?set REMOTE=user@host for the second machine, or run the sweep locally}
REMOTE_DIR=${REMOTE_DIR:-${REMOTE_REPO:-~/src/glucose-experiments}/runs_5090}
LOCAL_RUNS=${LOCAL_RUNS:-runs_5090}
SEEDS=${SEEDS:-"29 43"}
EPOCHS=${EPOCHS:-40}
# ckpt filenames are zero-padded to 3 digits (ckpt_ep040.pt).
EPOCH_PAD=$(printf "%03d" "$EPOCHS")

ABLATIONS=(
    abl_dense
    abl_event
    abl_fixedsigma
    abl_loo_shanghai
    abl_loo_stanford
    abl_noaug
    abl_nocirc
    abl_notd
    abl_raw
    abl_state
)

mkdir -p "$LOCAL_RUNS"

# 1. Rsync each (seed, ablation) ckpt from the 5090.
for seed in $SEEDS; do
    for ab in "${ABLATIONS[@]}"; do
        remote_ckpt="${REMOTE_DIR}/${ab}_seed${seed}/ckpt_ep${EPOCH_PAD}.pt"
        local_dir="${LOCAL_RUNS}/${ab}_seed${seed}"
        mkdir -p "$local_dir"
        if [[ ! -f "$local_dir/ckpt_ep${EPOCH_PAD}.pt" ]]; then
            echo "pulling $ab seed=$seed ..."
            scp -q "${REMOTE}:${remote_ckpt}" \
                "$local_dir/ckpt_ep${EPOCH_PAD}.pt" || { echo "  FAILED pull $ab seed=$seed"; continue; }
        else
            echo "$ab seed=$seed: already local"
        fi
    done
done

# 2. Evaluate each (seed, ablation) ckpt locally.
for seed in $SEEDS; do
    for ab in "${ABLATIONS[@]}"; do
        local_ckpt="${LOCAL_RUNS}/${ab}_seed${seed}/ckpt_ep${EPOCH_PAD}.pt"
        out="reports/eval/${ab}_seed${seed}_ep${EPOCH_PAD}"
        if [[ ! -f "$local_ckpt" ]]; then
            echo "SKIP $ab seed=$seed (no ckpt)"
            continue
        fi
        if [[ -f "$out/summary.csv" ]]; then
            echo "DONE $ab seed=$seed"
            continue
        fi
        echo "evaluating $ab seed=$seed -> $out"
        uv run python scripts/evaluate.py \
            --checkpoint "$local_ckpt" \
            --out "$out" \
            --device cuda 2>&1 | tail -5
    done
done

echo
echo "all done. CSV summaries at reports/eval/abl_*_seed{29,43}_ep040/summary.csv"