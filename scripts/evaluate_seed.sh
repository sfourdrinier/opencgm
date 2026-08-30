#!/usr/bin/env bash
# Run the headline evaluation for one seed × epoch.
#
#   bash scripts/evaluate_seed.sh <seed> <epoch> [tag]
#
# Defaults to the headline checkpoint path (runs_5090/rawstats120/seed<seed>/ckpt_ep<epoch>.pt)
# and writes results to reports/eval/seed<seed>_ep<epoch>_full/.

set -uo pipefail
cd "$(dirname "$0")/.."

seed="${1:-17}"
epoch="${2:-120}"
tag="${3:-full}"

# The sweep writes runs/<tag>_seed<N>/ (and the second machine's copies land in
# runs_5090/), while seed 17 has no suffix because it was the first run. An earlier version
# of this script looked for runs_5090/rawstats120/seed<N>/, a layout nothing has ever
# produced -- so anyone who followed REPRODUCE and trained for a day was told the checkpoint
# did not exist. Search the layouts that are real, and say what was tried when none match.
epoch_padded=$(printf '%03d' "$((10#${epoch}))")
ckpt=""
for cand in \
  "runs/${TAG:-rawstats120}_seed${seed}/ckpt_ep${epoch_padded}.pt" \
  "runs_5090/${TAG:-rawstats120}_seed${seed}/ckpt_ep${epoch_padded}.pt" \
  "runs/${TAG:-rawstats120}/ckpt_ep${epoch_padded}.pt" \
  "runs_5090/${TAG:-rawstats120}/ckpt_ep${epoch_padded}.pt"; do
  if [[ -f "$cand" ]]; then ckpt="$cand"; break; fi
done
if [[ -z "$ckpt" ]]; then
  echo "no checkpoint for seed ${seed} at epoch ${epoch_padded}. Looked in:" >&2
  printf '  %s\n' "runs/${TAG:-rawstats120}_seed${seed}/" "runs_5090/${TAG:-rawstats120}_seed${seed}/" \
                  "runs/${TAG:-rawstats120}/" "runs_5090/${TAG:-rawstats120}/" >&2
  echo "Run 'just pretrain-sweep' first, or set TAG to the tag you trained with." >&2
  exit 1
fi
out="reports/eval/seed${seed}_ep${epoch}_${tag}"

if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: checkpoint not found: $ckpt"
    echo "Run: just pretrain-sweep first, or pass a different path."
    exit 1
fi

echo "seed $seed epoch $epoch -> $out"
uv run python scripts/evaluate.py \
    --checkpoint "$ckpt" \
    --out "$out"
