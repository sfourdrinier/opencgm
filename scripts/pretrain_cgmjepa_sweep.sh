#!/usr/bin/env bash
# The five CGM-JEPA comparator seeds (D021, D022).
#
# The justfile's `pretrain-cgmjepa` recipe referenced this script and it did not exist, so
# `just pretrain-cgmjepa` -- and `just all`, which depends on it -- failed on a cold checkout.
#
# This is a port of the authors' released code (MIT, github.com/cruiseresearchgroup/CGM-JEPA,
# master @ 2026-05-11), not a reimplementation from the GlucoFM appendix. Where the authors and
# the appendix disagree, we follow the authors: see D021 and D022 for all seven divergences.
#
# Seeds match the GlucoFM sweep so the head-to-head is paired seed-for-seed. The authors' own
# official seed is 43, which is in the set.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"17 29 43 71 101"}
EPOCHS=${EPOCHS:-101}
TAG=${TAG:-cgmjepa}
DEVICE=${DEVICE:-cuda}
mkdir -p logs

for seed in $SEEDS; do
  out="runs/${TAG}_seed${seed}"
  if [[ -f "$out/ckpt_ep$(printf '%03d' "$EPOCHS").pt" ]]; then
    echo "seed $seed: already complete at epoch $EPOCHS -- skipping"
    continue
  fi
  nohup uv run python scripts/pretrain_cgm_jepa.py \
    --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" --out "$out" \
    >> "logs/${TAG}_seed${seed}.log" 2>&1 &
  echo "seed $seed: pid $! -> logs/${TAG}_seed${seed}.log"
done

echo
echo "watch:  tail -f logs/${TAG}_seed*.log"
wait
