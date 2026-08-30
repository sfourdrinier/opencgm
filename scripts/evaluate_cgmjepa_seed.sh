#!/usr/bin/env bash
# Run the CGM-JEPA comparator evaluation for one seed.
#
#   bash scripts/evaluate_cgmjepa_seed.sh <seed>
#
# Writes results to reports/eval/cgmjepa_seed<seed>_full/.

set -uo pipefail
cd "$(dirname "$0")/.."

seed="${1:-17}"

ckpt="runs_5090/cgmjepa_seed${seed}/ckpt_final.pt"
out="reports/eval/cgmjepa_seed${seed}_full"

if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: checkpoint not found: $ckpt"
    echo "Run: just pretrain-cgmjepa first, or pass a different path."
    exit 1
fi

echo "CGM-JEPA seed $seed -> $out"
uv run python scripts/evaluate_cgm_jepa.py \
    --checkpoint "$ckpt" \
    --out "$out"
