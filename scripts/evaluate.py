"""Run the downstream benchmark. Blueprint PR 10.

Fourteen dataset-tasks, subject-grouped five-fold cross-validation repeated ten times on folds
that are identical across every compared method, then Nadeau-Bengio corrected paired comparisons
Holm-adjusted across the whole family.

Everything is written to disk unrounded: fold-level scores, the fold manifest hash, the encoder
provenance, and the comparison table. §19.5 requires the fold-level scores to be retained, and a
summary table alone cannot be re-analysed.

    uv run python scripts/evaluate.py --checkpoint runs/strict_seed17/ckpt_last.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from opencgm_stateevent.eval import baselines, labels
from opencgm_stateevent.eval.embed import load_encoder, load_or_embed
from opencgm_stateevent.eval.probe import HEADLINE, run_probe
from opencgm_stateevent.eval.splits import build_folds, fold_manifest
from opencgm_stateevent.eval.stats import compare, holm_adjust
from opencgm_stateevent.eval.windows import WindowSet, build_all

#: Which window sources feed each label table. CGMacros contributes two sensor streams to the
#: same four tasks, evaluated separately at native cadence (§19.10).
SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}


def macro_cells(runs: dict, method: str, metric: str, *, by_dataset: bool) -> dict:
    """Macro-averaged score per (repeat, fold) cell, keyed so pairing can be checked. §19.5.

    Averaging happens inside a cell, which every task shares because `build_folds` depends only
    on `(task, repeat)` and never on the representation. That is what keeps the model-versus-
    baseline difference genuinely paired.

    `by_dataset` controls the weighting: CGMacros contributes two sensor streams to the same four
    tasks, so an unweighted entry average gives that cohort twice the weight of every other one.
    Both weightings are reported because neither is obviously correct.

    Cells where any task is undefined are dropped whole. Averaging over the surviving tasks
    instead would silently change what the macro-average means from one cell to the next.
    """
    cells: dict[tuple[int, int], dict[str, list[float]]] = {}
    for (key, run_method), run in runs.items():
        if run_method != method:
            continue
        group = key.split(":")[0] if by_dataset else key
        for r in run.results:
            cells.setdefault((r.repeat, r.fold), {}).setdefault(group, []).append(
                getattr(r, metric)
            )

    groups = {g for cell in cells.values() for g in cell}
    out = {}
    for coordinate, cell in sorted(cells.items()):
        if set(cell) != groups:
            continue
        per_group = [float(np.mean(v)) for v in (cell[g] for g in sorted(groups))]
        if any(np.isnan(per_group)):
            continue
        out[coordinate] = float(np.mean(per_group))
    return out


def paired_macro(runs: dict, method: str, baseline: str, metric: str, *, by_dataset: bool):
    """Model and baseline macro scores over the cells both of them define.

    Filtering each side independently and trusting equal lengths is not enough: if one method
    drops a different cell, the arrays line up by position while describing different folds, and
    the paired test silently subtracts mismatched partitions. Intersecting the keys makes the
    pairing structural rather than incidental.
    """
    a = macro_cells(runs, method, metric, by_dataset=by_dataset)
    b = macro_cells(runs, baseline, metric, by_dataset=by_dataset)
    shared = sorted(set(a) & set(b))
    return (np.array([a[k] for k in shared]), np.array([b[k] for k in shared]), shared)


def task_bootstrap_ci(
    runs: dict, method: str, baseline: str, metric: str, *, by_dataset: bool,
    draws: int = 2000, seed: int = 20260827,
) -> tuple[float, float]:
    """Percentile CI for the macro difference, resampling *tasks* rather than folds.

    This exists because the Nadeau-Bengio interval on macro cells understates the uncertainty,
    for a reason worth stating plainly: each task draws its own partition -- the fold seed
    includes the task key -- so "fold 2" of Hall and "fold 2" of Stanford are unrelated. Cell to
    cell variation therefore measures fold-resampling noise on an average of 18 independent
    tasks, shrunk by roughly sqrt(18), while the far larger question of *which tasks* happen to
    favour the model is averaged away and never counted.

    Resampling the tasks themselves puts that source of uncertainty back. It is the wider and
    wider interval, and it is the one the write-up should quote for a macro claim.
    """
    per_task = {}
    for (key, run_method), run in runs.items():
        if run_method not in (method, baseline):
            continue
        group = key.split(":")[0] if by_dataset else key
        values = run.scores(metric)
        if len(values):
            per_task.setdefault(group, {}).setdefault(run_method, []).append(float(values.mean()))

    tasks = [t for t, m in per_task.items() if method in m and baseline in m]
    if len(tasks) < 2:
        return float("nan"), float("nan")

    differences = np.array([
        np.mean(per_task[t][method]) - np.mean(per_task[t][baseline]) for t in tasks
    ])
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(draws)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def align(ws: WindowSet, label_table: pd.DataFrame, column: str):
    """Windows whose entry carries a label, with the label broadcast to each window.

    Labels join on `entry` and folds group on `subject`. For every source but ShanghaiT2DM these
    are the same string; for ShanghaiT2DM the entry is one visit and the subject is the patient,
    which is what keeps a patient's two visits on the same side of a split (§9.5).
    """
    table = label_table.dropna(subset=[column]).set_index("entry")[column]
    if table.index.duplicated().any():
        # A duplicated label row multiplies that subject's windows on a `.loc` lookup, weighting
        # them more heavily than everyone else's. It is a silent distortion, so it is an error.
        raise ValueError(
            f"{column}: duplicate label rows for "
            f"{sorted(table.index[table.index.duplicated()].unique())}"
        )
    keys = ws.entries if ws.entries is not None else ws.subjects
    keep = np.array([k in table.index for k in keys])
    if not keep.any():
        return None
    values = table.loc[list(keys[keep])].to_numpy()
    if len(values) != int(keep.sum()):
        raise ValueError(
            f"{column}: label join produced {len(values)} rows for {keep.sum()} windows"
        )
    return keep, values.astype(int), np.asarray(ws.subjects[keep])


def evaluate(checkpoint: Path, out: Path, *, device: str, quick: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n_repeats = 2 if quick else 10

    print("building downstream windows ...", flush=True)
    window_sets = build_all()
    label_tables = labels.build_all()

    print(f"loading encoder from {checkpoint} ...", flush=True)
    model, ref = load_encoder(checkpoint, device=device)
    print(f"  epoch {ref.epoch}, seed {ref.seed}, provenance tag {ref.tag}", flush=True)

    features: dict[str, dict[str, np.ndarray]] = {}
    for name, ws in window_sets.items():
        embedded = load_or_embed(model, ws, ref, device=device)
        features[name] = {
            "opencgm_mean": embedded["mean"],
            "opencgm_density_weighted": embedded["density_weighted"],
            "opencgm_mean_max": embedded["mean_max"],
            "clinical_metrics": baselines.build("clinical_metrics", ws),
            "raw_masked": baselines.build("raw_masked", ws),
        }

    methods = ["opencgm_mean", "clinical_metrics", "raw_masked"] + (
        [] if quick else ["opencgm_density_weighted", "opencgm_mean_max"]
    )

    fold_records, all_folds, runs = [], [], {}
    for task in labels.TASKS:
        table = label_tables[task.dataset]
        for source in SOURCES_FOR[task.dataset]:
            ws = window_sets[source]
            aligned = align(ws, table, task.name)
            if aligned is None:
                print(f"  {task.key} [{source}]: no labelled windows, skipped")
                continue
            keep, window_labels, window_subjects = aligned

            subjects, first = np.unique(window_subjects, return_index=True)
            subject_labels = window_labels[first]
            if len(np.unique(subject_labels)) < 2:
                print(f"  {task.key} [{source}]: single class, skipped")
                continue

            key = f"{task.key}[{source}]"
            folds = build_folds(key, subjects, subject_labels, n_repeats=n_repeats)
            all_folds.extend(folds)

            for method in methods:
                run = run_probe(
                    folds, features[source][method][keep], window_subjects, window_labels,
                    task=key, method=method, n_classes=task.n_classes, cfg=HEADLINE,
                )
                runs[(key, method)] = run
                fold_records.extend(run.to_records())
            summary = runs[(key, "opencgm_mean")].summary()
            print(
                f"  {key:<48} n={len(subjects):>3}  "
                f"PR-AUC {summary['pr_auc_mean']:.3f}  ROC-AUC {summary['roc_auc_mean']:.3f}",
                flush=True,
            )

    pd.DataFrame(fold_records).to_csv(out / "fold_scores.csv", index=False)
    summaries = [r.summary() for r in runs.values()]
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)

    comparisons = []
    for (key, method), run in runs.items():
        if not method.startswith("opencgm"):
            continue
        for baseline in ("clinical_metrics", "raw_masked"):
            other = runs.get((key, baseline))
            if other is None:
                continue
            for metric in ("pr_auc", "roc_auc", "macro_f1"):
                a = np.array([getattr(r, metric) for r in run.results])
                b = np.array([getattr(r, metric) for r in other.results])
                usable = ~(np.isnan(a) | np.isnan(b))
                if usable.sum() < 2:
                    continue
                comparisons.append(compare(
                    a[usable], b[usable], task=key, metric=metric,
                    method=method, baseline=baseline,
                ))
    comparisons = holm_adjust(comparisons)
    pd.DataFrame([c.__dict__ for c in comparisons]).to_csv(out / "comparisons.csv", index=False)

    # §19.5 macro-averaged comparison. Held in its own family and Holm-adjusted separately: it is
    # the headline number, and burying it in the per-task family would cost it power for no reason.
    macro = []
    for weighting, by_dataset in (("per_entry", False), ("per_dataset", True)):
        for method in methods:
            if not method.startswith("opencgm"):
                continue
            for baseline in ("clinical_metrics", "raw_masked"):
                for metric in ("pr_auc", "roc_auc", "macro_f1"):
                    a, b, _ = paired_macro(
                        runs, method, baseline, metric, by_dataset=by_dataset
                    )
                    if len(a) < 2:
                        continue
                    c = compare(a, b, task=f"MACRO[{weighting}]", metric=metric,
                                method=method, baseline=baseline)
                    # The wider interval. See `task_bootstrap_ci`: the Nadeau-Bengio one
                    # above counts fold noise only and is reported as descriptive.
                    low, high = task_bootstrap_ci(
                        runs, method, baseline, metric, by_dataset=by_dataset
                    )
                    macro.append(replace(c, bootstrap_ci_low=low, bootstrap_ci_high=high))
    macro = holm_adjust(macro)
    pd.DataFrame([c.__dict__ for c in macro]).to_csv(out / "macro_comparisons.csv", index=False)

    (out / "run_record.json").write_text(json.dumps({
        "encoder": ref.to_dict(),
        "fold_manifest": fold_manifest(all_folds),
        "n_repeats": n_repeats,
        "methods": methods,
        "probe": HEADLINE.__dict__,
        "windows": {k: v.summary() for k, v in window_sets.items()},
    }, indent=2, default=str))

    print(f"\nwrote {out}/fold_scores.csv, summary.csv, comparisons.csv, run_record.json")
    headline = [c for c in comparisons
                if c.method == "opencgm_mean" and c.metric == "roc_auc"]
    wins = sum(c.mean_difference > 0 for c in headline)
    significant = sum(c.significant and c.mean_difference > 0 for c in headline)
    print(f"opencgm_mean vs baselines, ROC-AUC: {wins}/{len(headline)} ahead, "
          f"{significant} significant after Holm")
    print("\nmacro-averaged difference. The task-bootstrap interval is the one to quote: the")
    print("fold-paired p-value counts fold noise only, because each task draws its own folds.")
    for c in macro:
        if c.method == "opencgm_mean" and c.metric == "roc_auc":
            crosses = c.bootstrap_ci_low <= 0 <= c.bootstrap_ci_high
            verdict = "spans zero" if crosses else "excludes zero"
            print(f"  {c.task} vs {c.baseline:<17} {c.mean_difference:+.4f}  "
                  f"task-bootstrap [{c.bootstrap_ci_low:+.4f}, {c.bootstrap_ci_high:+.4f}] "
                  f"{verdict}   (fold-paired p={c.p_holm:.3f}, descriptive)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true",
                    help="2 repeats and headline pooling only, for a preview")
    args = ap.parse_args()
    out = args.out or Path("reports/eval") / args.checkpoint.parent.name
    evaluate(args.checkpoint, out, device=args.device, quick=args.quick)


if __name__ == "__main__":
    main()
