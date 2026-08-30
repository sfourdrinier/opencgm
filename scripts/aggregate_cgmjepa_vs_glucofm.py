"""Aggregate the 5-seed CGM-JEPA head-to-head against the 5-seed GlucoFM head-to-head.

Each seed's `reports/eval/cgmjepa_seed*_full/macro_comparisons.csv` already has Nadeau-Bengio
corrected paired tests within that seed's fold structure; here we aggregate across seeds to get
seed-to-seed variance on the headline macro numbers.

Two outputs:
  * `reports/eval/cgmjepa_5seed_macro.csv` -- per (weighting, baseline, metric) seed-mean +/- sd
  * `reports/eval/head_to_head_5seed.csv` -- GlucoFM macro vs CGM-JEPA macro on the same rows

The GlucoFM numbers come from `reports/eval/seed*_ep120_full/macro_comparisons.csv`. Both used
identical fold structure because both used `build_folds` with the same subject-disjoint seed and
n_repeats. The comparison is paired across the (task, repeat) cells.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [17, 29, 43, 71, 101]
CGM_DIR = Path("reports/eval")
GLU_DIR = Path("reports/eval")


def load_macro(name: str, seeds: list[int], file_template: str) -> pd.DataFrame:
    rows = []
    for s in seeds:
        path = CGM_DIR / file_template.format(seed=s) / "macro_comparisons.csv"
        df = pd.read_csv(path)
        df["seed"] = s
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (task, metric, method, baseline) and compute seed-mean, seed-std, seed-CI."""
    rows = []
    for (task, metric, method, baseline), g in df.groupby(["task", "metric", "method", "baseline"]):
        diffs = g["mean_difference"].to_numpy()
        n = len(diffs)
        mean = diffs.mean()
        sd = diffs.std(ddof=1) if n > 1 else 0.0
        # 95% CI on the seed-mean uses t_{n-1}; with n=5 that's t_4 = 2.776
        if n > 1:
            from scipy.stats import t as student_t
            tcrit = student_t.ppf(0.975, df=n - 1)
            ci_lo = mean - tcrit * sd / np.sqrt(n)
            ci_hi = mean + tcrit * sd / np.sqrt(n)
        else:
            ci_lo = ci_hi = mean
        rows.append({
            "task": task, "metric": metric, "method": method, "baseline": baseline,
            "n_seeds": n,
            "mean_diff": mean,
            "seed_sd": sd,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "seeds_with_ci_excluding_zero":
                int((diffs > 0).all() if mean > 0 else (diffs < 0).all()),
            "all_seeds_same_sign": bool(np.all(diffs > 0) or np.all(diffs < 0)),
        })
    return pd.DataFrame(rows).sort_values(["task", "baseline", "metric"])


def main() -> None:
    cgm = load_macro("cgm-jepa", SEEDS, "cgmjepa_seed{seed}_full")
    glu = load_macro("glucofm", SEEDS, "seed{seed}_ep120_full")

    cgm_agg = aggregate(cgm)
    glu_agg = aggregate(glu)
    cgm_agg.to_csv("reports/eval/cgmjepa_5seed_macro.csv", index=False)
    glu_agg.to_csv("reports/eval/glucofm_5seed_macro.csv", index=False)

    # Pair them up. The fair comparator is `opencgm_mean`: same pooling kind (mean), both
    # single-token reduction to a fixed-dim vector. `opencgm_mean_max` concatenates two pools
    # so it doesn't compare like-for-like; `opencgm_density_weighted` is the same model with a
    # different head output. They all live in the per-row tables for completeness; this is the
    # headline.
    headline_methods = {"opencgm_mean"}
    glu_head = glu_agg[glu_agg["method"].isin(headline_methods)]
    rows = []
    for _, gc in cgm_agg.iterrows():
        gg = glu_head[
            (glu_head["task"] == gc["task"])
            & (glu_head["metric"] == gc["metric"])
            & (glu_head["baseline"] == gc["baseline"])
        ]
        if len(gg) != 1:
            continue
        gg = gg.iloc[0]
        rows.append({
            "task": gc["task"], "metric": gc["metric"], "baseline": gc["baseline"],
            "glucofm_method": "opencgm_mean",
            "glucofm_mean": gg["mean_diff"],
            "glucofm_ci": f"[{gg['ci_low']:.4f}, {gg['ci_high']:.4f}]",
            "cgmjepa_mean": gc["mean_diff"],
            "cgmjepa_ci": f"[{gc['ci_low']:.4f}, {gc['ci_high']:.4f}]",
            "delta": gg["mean_diff"] - gc["mean_diff"],
            "glucofm_strictly_ahead": bool(
                gg["ci_low"] > 0 and gc["ci_high"] < 0
            ),
        })
    h2h = pd.DataFrame(rows).sort_values(["task", "baseline", "metric"])
    h2h.to_csv("reports/eval/head_to_head_5seed.csv", index=False)
    print(h2h.to_string(index=False))


if __name__ == "__main__":
    main()
