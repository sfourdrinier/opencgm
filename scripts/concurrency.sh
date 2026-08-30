#!/usr/bin/env bash
# How well do concurrent seeds share one 3090? Blueprint §18.1 step 3.
#
# Launches N independent training processes at once and reports aggregate throughput. Independent
# OS processes, not a multiprocessing pool, because that is the shape the seed sweep will take.
#
# Concurrency is the only speedup available here that changes nothing about the arithmetic: each
# process runs the same eager float32 code it would run alone.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-300}
WORKERS=${WORKERS:-4}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for N in "$@"; do
  pids=()
  for ((r = 0; r < N; r++)); do
    uv run python scripts/bench_one.py \
      --seed $((17 + r)) --steps "$STEPS" --workers "$WORKERS" \
      --out "$TMP/n${N}_r${r}.json" >/dev/null 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done

  uv run python - "$TMP" "$N" <<'PY'
import json, sys, pathlib
tmp, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rs = [json.loads(p.read_text()) for p in sorted(tmp.glob(f"n{n}_r*.json"))]
agg = sum(r["steps_per_sec"] for r in rs)
slowest = max(r["ms_per_step"] for r in rs)
per_seed = 2759 * 120 / (1000 / slowest)          # 331,057 steps at 120 epochs
h = lambda s: f"{int(s)//3600}h {int(s)%3600//60:02d}m"
print(f"{n:>2} proc  aggregate {agg:6.2f} step/s   slowest {slowest:6.2f} ms/step   "
      f"vram {sum(r['peak_vram_gb'] for r in rs):5.2f} GB   "
      f"per seed {h(per_seed):>8}   all 5 seeds {h(2759*120*5/agg):>8}")
PY
done
