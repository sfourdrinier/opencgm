"""PPG pilot evaluator (D023, A7).

Loads one or more `reports/eval/ppg_pilot/ckpt_seed*.pt` checkpoints and reports
per-subject alignment + glucose regression numbers in a flat CSV. Designed to run after
`scripts/ppg_teacher_student.py`.

Outputs:
  - `reports/eval/ppg_pilot/fold_scores.csv` (re-emitted, with per-subject columns added)
  - `reports/eval/ppg_pilot/per_subject.csv` (5 subjects x 5 seeds matrix)
  - `reports/eval/ppg_pilot/per_seed_summary.csv` (already written by trainer; copied here)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("reports/eval/ppg_pilot"))
    ap.add_argument("--out", type=Path, default=Path("reports/eval/ppg_pilot/aggregate"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fold_csv = args.inp / "fold_scores.csv"
    if not fold_csv.exists():
        raise SystemExit(f"missing {fold_csv}; run scripts/ppg_teacher_student.py first")
    df = pd.read_csv(fold_csv)

    # Per-subject pivot
    pivot = df.pivot_table(
        index="test_subject",
        columns="student_seed",
        values=["alignment_cosine", "alignment_mse", "glucose_rmse_mmol", "glucose_mae_mmol"],
        aggfunc="mean",
    )
    pivot.to_csv(args.out / "per_subject.csv")

    # Aggregate across all (seed, fold) pairs.
    summary_dict = {
        "n_folds": len(df),
        "alignment_cosine_mean": float(df["alignment_cosine"].mean()),
        "alignment_cosine_sd": (
            float(df["alignment_cosine"].std(ddof=1)) if len(df) > 1 else 0.0
        ),
        "alignment_mse_mean": float(df["alignment_mse"].mean()),
        "alignment_mse_sd": (
            float(df["alignment_mse"].std(ddof=1)) if len(df) > 1 else 0.0
        ),
        "glucose_rmse_mean": float(df["glucose_rmse_mmol"].mean()),
        "glucose_rmse_sd": (
            float(df["glucose_rmse_mmol"].std(ddof=1)) if len(df) > 1 else 0.0
        ),
        "glucose_mae_mean": float(df["glucose_mae_mmol"].mean()),
        "glucose_mae_sd": (
            float(df["glucose_mae_mmol"].std(ddof=1)) if len(df) > 1 else 0.0
        ),
    }
    summary = pd.DataFrame([summary_dict])
    summary.to_csv(args.out / "aggregate.csv", index=False)

    print(f"wrote {args.out}/per_subject.csv, aggregate.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
