"""Aggregate the 3-seed Tier-1 ablation matrix.

Inputs:
  reports/eval/abl_<NAME>_ep040/summary.csv                 (seed 17, the original 10 ablations)
  reports/eval/abl_<NAME>_seed29_ep040/summary.csv          (seed 29, in progress)
  reports/eval/abl_<NAME>_seed43_ep040/summary.csv          (seed 43, in progress)
  reports/eval/seed<17|29|43|71|101>_ep040_full/summary.csv (the "full" baseline at ep40)

Outputs:
  reports/eval/tier1_ablations_3seed.csv     per-(ablation, seed) macro ROC + 3-seed mean ± sd
  reports/eval/tier1_ablations_full_baseline.csv  the ep40 full baseline mean and per-seed values

The macro ROC per (ablation, seed) is the mean of `roc_auc_mean` over all task rows for the
`opencgm_mean` method in that summary.csv (the headline L2 logistic regression probe, mean
across the 10 fold-repeats per task, then averaged across all tasks for that ablation).

Per the BLUEPRINT §21 + DECISIONS.md D019, the dual-stream `opencgm_mean` is the headline
embedding for downstream probing; that is the only method reported in this matrix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ABLATIONS = [
    "abl_dense",
    "abl_event",
    "abl_fixedsigma",
    "abl_noaug",
    "abl_nocirc",
    "abl_notd",
    "abl_raw",
    "abl_state",
    "abl_loo_shanghai",
    "abl_loo_stanford",
]

# The full baseline lives at ep40 in seed{29,43,71,101}_ep040_full/ (no seed17 ep40 full
# eval was kept on disk; the headline 5-seed 0.670 is at ep120 and is not a like-for-like
# comparison for these ep40 ablations). We exclude seed17 from the baseline pool since
# only the ablations were evaluated at ep40 for that seed.
FULL_BASELINE_SEEDS = [29, 43, 71, 101]
ABLATION_SEEDS = [17, 29, 43]

EVAL_DIR = Path("reports/eval")


def macro_roc(path: Path) -> float | None:
    """Mean ROC across all task rows for the `opencgm_mean` method. None if file missing."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    sub = df[df.method == "opencgm_mean"]
    if sub.empty:
        return None
    return float(sub.roc_auc_mean.mean())


def per_task_rocs(path: Path) -> pd.Series | None:
    """Per-task ROC for the opencgm_mean method, useful for paired deltas."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    sub = df[df.method == "opencgm_mean"]
    if sub.empty:
        return None
    return sub.set_index("task")["roc_auc_mean"]


def main() -> None:
    rows = []
    for ab in ABLATIONS:
        for seed in ABLATION_SEEDS:
            if seed == 17:
                p = EVAL_DIR / f"{ab}_ep040" / "summary.csv"
            else:
                p = EVAL_DIR / f"{ab}_seed{seed}_ep040" / "summary.csv"
            v = macro_roc(p)
            rows.append({"ablation": ab, "seed": seed, "macro_roc": v, "path": str(p)})
    df = pd.DataFrame(rows)

    # Pivot to wide for the headline table
    wide = df.pivot_table(
        index="ablation", columns="seed", values="macro_roc", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    for s in ABLATION_SEEDS:
        if s not in wide.columns:
            wide[s] = np.nan
    wide["mean"] = wide[ABLATION_SEEDS].mean(axis=1, skipna=True)
    wide["sd"] = wide[ABLATION_SEEDS].std(axis=1, skipna=True, ddof=1)
    wide["n_seeds"] = wide[ABLATION_SEEDS].notna().sum(axis=1)

    # Full baseline at ep40 (mean over seed29/43/71/101)
    full_rows = []
    for s in FULL_BASELINE_SEEDS:
        v = macro_roc(EVAL_DIR / f"seed{s}_ep040_full" / "summary.csv")
        full_rows.append({"seed": s, "macro_roc": v})
    full_df = pd.DataFrame(full_rows)
    full_mean = float(full_df["macro_roc"].mean())
    full_sd = float(full_df["macro_roc"].std(ddof=1))
    print(f"full ep40 baseline: mean={full_mean:.4f} sd={full_sd:.4f} (n={len(full_df)})")
    print(f"  per-seed: {full_df.to_dict('records')}")

    # Compute delta vs full (mean)
    wide["delta_vs_full"] = wide["mean"] - full_mean

    # Sort by delta ascending (worst ablation first)
    wide = wide.sort_values("delta_vs_full").reset_index(drop=True)

    out_csv = EVAL_DIR / "tier1_ablations_3seed.csv"
    wide.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    full_csv = EVAL_DIR / "tier1_ablations_full_baseline.csv"
    full_df.to_csv(full_csv, index=False)
    print(f"wrote {full_csv}")

    print()
    print(wide.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"full ep40 baseline mean: {full_mean:.4f} (sd={full_sd:.4f}, n={len(full_df)})")


if __name__ == "__main__":
    main()
