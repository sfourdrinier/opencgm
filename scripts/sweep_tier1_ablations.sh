#!/usr/bin/env bash
# Launch Tier-1 ablation seeds {29, 43} on the 5090, 5-way concurrent.
# Seed 17 ablations already exist at ep40 (single-seed); this gives us 3 seeds × 10 ablations.
#
# Each ablation is a 40-epoch pretraining that reuses the same windows cache and same probe,
# so the per-ablation eval will land in `reports/eval/abl_*_seed29_ep040` etc. and be compared
# head-to-head with `reports/eval/abl_*_ep040` (seed 17).
#
# Skip-if-done guard: if `ckpt_ep040.pt` exists for (seed, ablation), skip.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"29 43"}
EPOCHS=${EPOCHS:-40}
LOG_DIR=${LOG_DIR:-logs/ablations}
mkdir -p "$LOG_DIR"

# 10 Tier-1 ablations. Flags map:
#   abl_raw: --normalized-statistics
#   abl_noaug: --no-augment
#   abl_dense: --dense-interpolation
#   abl_event: --streams event
#   abl_state: --streams state
#   abl_fixedsigma: --fixed-sigma
#   abl_nocirc: --no-circadian
#   abl_notd: --lambda-td 0.0
#   abl_loo_shanghai: --exclude-dataset shanghai_t2dm
#   abl_loo_stanford: --exclude-dataset stanford
declare -A FLAGS=(
    [abl_raw]="--normalized-statistics"
    [abl_noaug]="--no-augment"
    [abl_dense]="--dense-interpolation"
    [abl_event]="--streams event"
    [abl_state]="--streams state"
    [abl_fixedsigma]="--fixed-sigma"
    [abl_nocirc]="--no-circadian"
    [abl_notd]="--lambda-td 0.0"
    [abl_loo_shanghai]="--exclude-dataset shanghai_t2dm"
    [abl_loo_stanford]="--exclude-dataset stanford"
)

# Run on 5090. The pretrain script is identical; we just point PYTHONPATH to the repo and
# invoke the existing CLI. The 5090 paths are ${REMOTE_REPO:-~/src/glucose-experiments}.
# No default host. This used to name the author's own machine, so anyone running the
# recipe pointed an ssh session at a stranger's box.
REMOTE=${REMOTE:?set REMOTE=user@host for the second machine, or run the sweep locally}
REMOTE_DIR=${REMOTE_DIR:-src/glucose-experiments}

for seed in $SEEDS; do
    for ab in "${!FLAGS[@]}"; do
        ckpt="${REMOTE_DIR}/runs_5090/${ab}_seed${seed}/ckpt_ep040.pt"
        if ssh -o BatchMode=yes "$REMOTE" "test -f $ckpt" 2>/dev/null; then
            echo "seed $seed $ab: ckpt exists, skipping"
            continue
        fi
        log="${LOG_DIR}/${ab}_seed${seed}.log"
        flags="${FLAGS[$ab]}"
        echo "seed $seed $ab: launching  ($flags)  -> $log"
        ssh -o BatchMode=yes "$REMOTE" "cd $REMOTE_DIR && nohup uv run python -m opencgm_stateevent.train.run \
            --seed $seed --epochs $EPOCHS --tag ${ab}_seed${seed} \
            --out runs_5090/${ab}_seed${seed} --num-workers 2 \
            $flags > $log 2>&1 &" || echo "FAILED $seed $ab"
    done
done
echo
echo "watch: tail -f $LOG_DIR/*.log | grep '=='"
