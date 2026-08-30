"""Aggregate the 5-seed per-task results into one reviewer-facing table.

The public write-up claims a per-task tally ("significantly ahead of the clinical baseline on
16 of 18 task-cohort combinations after Holm correction"), but the only aggregated file on disk
carries two macro rows. This script writes `reports/eval/per_task_5seed.csv`: one row per
task-source combination (18 rows -- 14 dataset-task probes, of which CGMacros' four tasks appear
once per sensor), so a reader can check the tally instead of trusting it.

What each statistic means, stated plainly because the two kinds of interval answer different
questions:

* Levels (`*_roc_auc`, `*_pr_auc`) are the mean over the 50 fold scores, then the mean over the
  five pretraining seeds; `*_seed_sd` is the spread of that per-seed mean across seeds. The two
  baselines are deterministic functions of the input windows, identical in every seed directory
  (asserted below), so they carry no seed spread.

* Deltas and their 95% CIs come from the Nadeau-Bengio corrected paired test on the 50
  (repeat, fold) cells, with the model's fold scores averaged over the five seeds first.
  Averaging first is well defined because `build_folds` seeds the partition from the task key
  alone, so every seed scored the identical folds -- the pairing is structural. The interval
  therefore counts fold-resampling noise (inflated by the NB train-overlap term) for the
  seed-averaged encoder. It does NOT count encoder-seed noise -- that is `*_delta_seed_sd`,
  reported beside it -- and, being per-task, it makes no claim about how the model fares on
  tasks not in this benchmark. This is a different situation from the macro comparison, where
  pairing cells across tasks is arbitrary and the standing rule is not to quote a NB p-value
  at all; within one task the pairing is exact.

* `*_p_holm` is Holm-adjusted over one model's whole planned family here: 18 tasks x 2 baselines
  x 2 metrics = 72 tests, mirroring `evaluate.py`'s one-family policy. GlucoFM and CGM-JEPA are
  adjusted as separate families, as they were in their separate runs. `*_significant` is
  p_holm < 0.05. `*_n_seeds_sig_holm` is the supplementary per-seed view: in how many of the
  five single-seed runs the within-run Holm-adjusted test was significant with the same sign.

    uv run python scripts/aggregate_per_task.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from opencgm_stateevent.eval.stats import compare, holm_adjust

SEEDS = [17, 29, 43, 71, 101]
METRICS = ("roc_auc", "pr_auc")

#: (label, probe-method name in the CSVs, per-seed directory template)
MODELS = {
    "glucofm": ("opencgm_mean", "reports/eval/seed{seed}_ep120_full"),
    "cgmjepa": ("cgm_jepa", "reports/eval/cgmjepa_seed{seed}_full"),
}
BASELINES = {"clinical": "clinical_metrics", "raw": "raw_masked"}

OUT = "reports/eval/per_task_5seed.csv"


def _load(template: str, name: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        frames.append(pd.read_csv(f"{template.format(seed=seed)}/{name}").assign(seed=seed))
    return pd.concat(frames, ignore_index=True)


def _pivot(fold_scores: pd.DataFrame, task: str, method: str, metric: str) -> pd.DataFrame:
    """Fold scores as a (repeat, fold) x seed frame, aligned on the shared partition."""
    d = fold_scores[(fold_scores.task == task) & (fold_scores.method == method)]
    return d.pivot_table(
        index=["repeat", "fold"], columns="seed", values=metric, dropna=False
    ).sort_index()


def _assert_baselines_identical(fold_scores: dict[str, pd.DataFrame]) -> None:
    """The baselines are seed-free and fold-aligned; anything else voids the pairing."""
    for baseline in BASELINES.values():
        reference = None
        for frame in fold_scores.values():
            d = frame[frame.method == baseline].sort_values(["task", "seed", "repeat", "fold"])
            per_seed = d.pivot_table(
                index=["task", "repeat", "fold"], columns="seed", values="roc_auc", dropna=False
            )
            if not per_seed.apply(lambda row: row.nunique(dropna=True) <= 1, axis=1).all():
                raise AssertionError(f"{baseline} fold scores differ across seeds")
            cur = per_seed[SEEDS[0]]
            if reference is None:
                reference = cur
            elif not np.allclose(reference.fillna(-1), cur.fillna(-1)):
                raise AssertionError(f"{baseline} fold scores differ across model runs")


def build_table() -> pd.DataFrame:
    fold_scores = {m: _load(tpl, "fold_scores.csv") for m, (_, tpl) in MODELS.items()}
    summaries = {m: _load(tpl, "summary.csv") for m, (_, tpl) in MODELS.items()}
    comparisons = {m: _load(tpl, "comparisons.csv") for m, (_, tpl) in MODELS.items()}
    _assert_baselines_identical(fold_scores)

    tasks = sorted(fold_scores["glucofm"].task.unique())
    if sorted(fold_scores["cgmjepa"].task.unique()) != tasks:
        raise AssertionError("the two runs cover different task sets")

    # One Holm family per model: every test this table reports for that model. §19.5 policy.
    tests: dict[str, list] = {m: [] for m in MODELS}
    for model, (method, _) in MODELS.items():
        fs = fold_scores[model]
        for task in tasks:
            for metric in METRICS:
                a = _pivot(fs, task, method, metric).mean(axis=1).to_numpy()
                for baseline in BASELINES.values():
                    b = _pivot(fs, task, baseline, metric)[SEEDS[0]].to_numpy()
                    usable = ~(np.isnan(a) | np.isnan(b))
                    if usable.sum() < 2:
                        continue
                    tests[model].append(compare(
                        a[usable], b[usable], task=task, metric=metric,
                        method=f"{method}_5seed_mean", baseline=baseline,
                    ))
        tests[model] = holm_adjust(tests[model])

    pooled = {
        (m, c.task, c.baseline, c.metric): c for m in MODELS for c in tests[m]
    }

    rows = []
    for task in tasks:
        cohort, rest = task.split(":", 1)
        task_name, source = rest[:-1].split("[")
        row: dict = {"task": task, "cohort": cohort, "task_name": task_name, "source": source}

        # Subject and window counts, from the shared fold structure (repeat 0 spans everyone).
        d = fold_scores["glucofm"]
        r0 = d[(d.task == task) & (d.method == "opencgm_mean") & (d.repeat == 0) & (d.seed == 17)]
        row["n_subjects"] = int(r0.n_test_subjects.sum())
        row["n_windows"] = int((r0.n_train + r0.n_test).iloc[0])
        row["n_folds"] = len(d[(d.task == task) & (d.method == "opencgm_mean") & (d.seed == 17)])

        # Levels: models get seed mean +/- sd; deterministic baselines get their single value.
        for model in MODELS:
            method = MODELS[model][0]
            s = summaries[model][
                (summaries[model].task == task) & (summaries[model].method == method)
            ]
            for metric in METRICS:
                row[f"{model}_{metric}"] = float(s[f"{metric}_mean"].mean())
                row[f"{model}_{metric}_seed_sd"] = float(s[f"{metric}_mean"].std(ddof=1))
        s = summaries["glucofm"]
        for short, baseline in BASELINES.items():
            b = s[(s.task == task) & (s.method == baseline)]
            for metric in METRICS:
                values = b[f"{metric}_mean"]
                if values.std(ddof=1) > 1e-12:
                    raise AssertionError(f"{baseline} varies across seeds on {task}")
                row[f"{short}_{metric}"] = float(values.iloc[0])

        # Deltas: pooled NB test + Holm, seed spread, and the per-seed significance tally.
        for model, (method, _) in MODELS.items():
            comp = comparisons[model]
            for short, baseline in BASELINES.items():
                for metric in METRICS:
                    prefix = f"{model}_vs_{short}_{metric}"
                    c = pooled.get((model, task, baseline, metric))
                    per_seed = comp[
                        (comp.task == task) & (comp.method == method)
                        & (comp.baseline == baseline) & (comp.metric == metric)
                    ]
                    row[f"{prefix}_delta"] = c.mean_difference if c else float("nan")
                    row[f"{prefix}_ci_low"] = c.ci_low if c else float("nan")
                    row[f"{prefix}_ci_high"] = c.ci_high if c else float("nan")
                    row[f"{prefix}_seed_sd"] = float(per_seed.mean_difference.std(ddof=1))
                    row[f"{prefix}_p_holm"] = c.p_holm if c else float("nan")
                    row[f"{prefix}_significant"] = bool(c.significant) if c else False
                    row[f"{prefix}_n_seeds_sig_holm"] = int((
                        (per_seed.p_holm < 0.05)
                        & (np.sign(per_seed.mean_difference)
                           == np.sign(c.mean_difference if c else 1.0))
                    ).sum())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)


def main() -> None:
    table = build_table()
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(table)} task-source rows")

    # Reconciliation against the macro headline (mean over the 18 per-task deltas is the
    # per-entry macro, up to cells the macro drops as incomplete).
    for label, column in [
        ("GlucoFM vs clinical", "glucofm_vs_clinical_roc_auc_delta"),
        ("GlucoFM vs raw", "glucofm_vs_raw_roc_auc_delta"),
        ("CGM-JEPA vs clinical", "cgmjepa_vs_clinical_roc_auc_delta"),
    ]:
        print(f"  mean of per-task ROC deltas, {label:<22} {table[column].mean():+.4f}")
    print(f"  mean of per-task GlucoFM ROC-AUC levels     {table['glucofm_roc_auc'].mean():.4f}")

    for model in MODELS:
        for short in BASELINES:
            column = f"{model}_vs_{short}_roc_auc"
            n = int((table[f"{column}_significant"]
                     & (table[f"{column}_delta"] > 0)).sum())
            print(f"  {model} significantly ahead of {short} (ROC, Holm): {n}/{len(table)}")


if __name__ == "__main__":
    main()
