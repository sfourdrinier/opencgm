"""State of the pretraining sweep, read from disk.

`STATE.md` and anything I remember can be stale; `runs/*/epochs.jsonl` cannot. This reads the run
directories and reports progress, the health diagnostics that matter (§15.6), and an ETA from
measured per-epoch times rather than from the benchmark.

Three things are worth watching and all three are printed:

* **sigma** must stay inside [2, 12] and keep moving. Pinned at a bound means the learned
  bandwidth has stopped being data-dependent.
* **effective rank** must not fall. A collapsing representation loses rank before its loss looks
  wrong (§15.6).
* **latent std** near zero on any dimension is the same warning by a cruder measure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def human(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"


def read(run: Path) -> list[dict]:
    f = run / "epochs.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--tag", default="strict")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    runs = sorted(args.root.glob(f"{args.tag}_seed*"))
    if not runs:
        print(f"no runs matching {args.root}/{args.tag}_seed*")
        return

    print(f"{'run':<20} {'epoch':>9} {'loss':>9} {'mcr':>8} {'td':>8} "
          f"{'sigma':>7} {'rank':>6} {'lat.std':>8} {'eta':>7}")
    print("-" * 88)
    for run in runs:
        rows = read(run)
        if not rows:
            print(f"{run.name:<20} {'—':>9}  (no epochs yet)")
            continue
        last = rows[-1]
        done = last["epoch"] + 1
        per_epoch = sum(r["seconds"] for r in rows[-5:]) / len(rows[-5:])
        eta = human(per_epoch * (args.epochs - done))
        print(
            f"{run.name:<20} {done:>4}/{args.epochs:<4} {last['loss']:>9.5f} "
            f"{last['mcr']:>8.5f} {last['td']:>8.5f} {last['sigma']:>7.3f} "
            f"{last['health_effective_rank']:>6.2f} "
            f"{last['health_latent_std_min']:>8.4f} {eta:>7}"
        )

    print()
    for run in runs:
        rows = read(run)
        if len(rows) < 2:
            continue
        warn = []
        s = rows[-1]["sigma"]
        if s < 2.1 or s > 11.9:
            warn.append(f"sigma {s:.3f} is at a bound — bandwidth may have stopped adapting")
        ranks = [r["health_effective_rank"] for r in rows]
        if len(ranks) >= 5 and ranks[-1] < max(ranks) * 0.7:
            warn.append(
                f"effective rank {ranks[-1]:.2f} is well below its peak {max(ranks):.2f}"
                " — possible representation collapse"
            )
        if rows[-1]["health_latent_std_min"] < 1e-3:
            warn.append("a latent dimension has near-zero variance")
        for w in warn:
            print(f"  {run.name}: {w}")


if __name__ == "__main__":
    main()
