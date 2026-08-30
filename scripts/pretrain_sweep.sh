#!/usr/bin/env bash
# PR 9: the five pretraining seeds. Blueprint §17.1, §18.
#
# All five run concurrently on the one GPU (D012). The step is launch-latency bound, so they
# interleave into each other's gaps: about 6h15 for all five against 11h30 serially, in under a
# gigabyte of VRAM. Each process runs the identical deterministic float32 code it would run
# alone, so a concurrently-trained seed is bit-identical to a serially-trained one.
#
# Resumable: re-running this script picks each seed up from its last epoch checkpoint. That is
# safe because batch order is derived from (seed, epoch) rather than held in dataloader state,
# and it is verified by scripts/resume_gate.py.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"17 29 43 71 101"}
EPOCHS=${EPOCHS:-120}
TAG=${TAG:-strict}
mkdir -p logs

for seed in $SEEDS; do
  out="runs/${TAG}_seed${seed}"
  resume=""
  if [[ -f "$out/ckpt_last.pt" ]]; then
    resume="--resume $out/ckpt_last.pt"
    echo "seed $seed: resuming from $out/ckpt_last.pt"
  fi
  # shellcheck disable=SC2086
  nohup uv run python -m opencgm_stateevent.train.run \
    --seed "$seed" --epochs "$EPOCHS" --tag "$TAG" --out "$out" \
    --num-workers 3 --log-every 500 $resume \
    >> "logs/${TAG}_seed${seed}.log" 2>&1 &
  echo "seed $seed: pid $! -> logs/${TAG}_seed${seed}.log"
done

echo
echo "watch:   tail -f logs/${TAG}_seed*.log | grep '=='"
echo "status:  uv run python scripts/sweep_status.py"
wait
