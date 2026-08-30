"""§19.7 few-shot: k labelled subjects per class, fit probe, score held-out subjects.

A probe fitted on k=1 subject per class is asking how much the encoder's daily embedding carries
of the phenotype on its own. The paper reports k=1, 5, 10, 20 per class. We run the same grid.

Subject-disjoint folds (so the test subjects are never in the training pool). For each k:
  * for each repeat (10), for each fold (5), sample k training subjects per class from the
    training pool (with replacement if the class has fewer than k subjects),
  * fit the probe on those subjects only,
  * score the test subjects of that fold.

Output: `reports/eval/fewshot_seed43/macro_comparisons.csv` and per-task CSV, plus a curve CSV.

Single seed (43) — the paper does it single-seed, and the variance is dominated by the per-fold
probe, not by the encoder. The encoder itself is fixed across this evaluation.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from opencgm_stateevent.eval import baselines, labels
from opencgm_stateevent.eval.embed import embed as encode_windows
from opencgm_stateevent.eval.probe import HEADLINE, FoldResult, ProbeRun, make_pipeline
from opencgm_stateevent.eval.splits import build_folds
from opencgm_stateevent.eval.windows import build_all
from opencgm_stateevent.infer import Analyser

K_VALUES = (1, 5, 10, 20)
N_REPEATS = 10
N_FOLDS = 5


def _fit_score(
    features: np.ndarray,
    window_subjects: np.ndarray,
    window_labels: np.ndarray,
    fold,
    *,
    k: int,
    n_classes: int,
    seed: int,
) -> FoldResult:
    """Sample k train subjects per class from the fold's training pool, fit, score test."""
    train_mask = np.isin(window_subjects, list(fold.train_subjects))
    test_mask = np.isin(window_subjects, list(fold.test_subjects))
    train_subjs = np.unique(window_subjects[train_mask])
    train_labels = np.array([
        window_labels[train_mask][window_subjects[train_mask] == s][0]
        for s in train_subjs
    ])

    rng = np.random.default_rng(seed)
    keep = []
    for cls in range(n_classes):
        pool = train_subjs[train_labels == cls]
        if len(pool) == 0:
            continue
        n_take = min(k, len(pool))
        # sample with replacement if pool < k (paper does this for tiny cohorts)
        replace = n_take > len(pool)
        chosen = rng.choice(pool, size=n_take, replace=replace)
        keep.extend(chosen.tolist())
    keep_set = set(keep)

    sub_mask = np.array([s in keep_set for s in window_subjects])
    final_train = train_mask & sub_mask

    result = FoldResult(
        task=fold.task, method=f"fewshot_k{k}", repeat=fold.repeat, fold=fold.fold,
        n_train=int(final_train.sum()), n_test=int(test_mask.sum()),
        n_test_subjects=len(fold.test_subjects),
        note=f"k={k} per class",
    )
    if result.n_train == 0 or result.n_test == 0:
        return result
    y_train = window_labels[final_train]
    if len(np.unique(y_train)) < 2:
        result.note = "single-class few-shot"
        return result

    pipeline = make_pipeline(HEADLINE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pipeline.fit(features[final_train], y_train)
            proba = pipeline.predict_proba(features[test_mask])
        except Exception as exc:
            result.note = f"fit failed: {exc}"
            return result
    y_test = window_labels[test_mask]
    pred = pipeline.predict(features[test_mask])

    from opencgm_stateevent.eval.probe import _scores
    scores = _scores(y_test, proba, pred, n_classes=n_classes)
    result.roc_auc = scores.get("roc_auc")
    result.pr_auc = scores.get("pr_auc")
    result.macro_f1 = scores.get("macro_f1")
    return result


SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/eval/fewshot_seed43"))
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading analyser ...", flush=True)
    analyser = Analyser.load(args.checkpoint, device="cpu")
    model = analyser.model
    print(f"  encoder ep {analyser.ref.epoch}, seed {analyser.ref.seed}", flush=True)

    print("building windows ...", flush=True)
    window_sets = build_all()
    label_tables = labels.build_all()

    print("embedding windows ...", flush=True)
    features: dict[str, dict[str, np.ndarray]] = {}
    for name, ws in window_sets.items():
        embs = encode_windows(model, ws, device="cpu", poolings=("mean",))
        features[name] = {
            "opencgm_mean": embs["mean"],
            "raw_masked": baselines.build("raw_masked", ws),
        }

    methods = ["opencgm_mean", "raw_masked"]

    # Per-task k-shots
    fold_records = []
    curve_records = []  # one row per (task, method, k) with mean ROC
    for task in labels.TASKS:
        table = label_tables[task.dataset]
        for source in SOURCES_FOR[task.dataset]:
            ws = window_sets[source]
            # align as in scripts/evaluate.py
            import sys
            sys.path.insert(0, "scripts")
            from evaluate import align
            aligned = align(ws, table, task.name)
            if aligned is None:
                continue
            keep, window_labels, window_subjects = aligned
            subjects, first = np.unique(window_subjects, return_index=True)
            subject_labels = window_labels[first]
            if len(np.unique(subject_labels)) < 2:
                continue

            key = f"{task.key}[{source}]"
            folds = build_folds(key, subjects, subject_labels,
                                n_repeats=N_REPEATS, n_folds=N_FOLDS)

            for method in methods:
                feat = features[source][method][keep]
                for k in K_VALUES:
                    run = ProbeRun(task=key, method=method, config=HEADLINE)
                    for fold in folds:
                        seed_base = (
                            args.seed * 1000
                            + abs(hash((key, method, k, fold.repeat, fold.fold))) % (2**31)
                        )
                        run.results.append(_fit_score(
                            feat, window_subjects, window_labels, fold,
                            k=k, n_classes=task.n_classes, seed=seed_base,
                        ))
                    for r in run.results:
                        fold_records.append(asdict(r))
                    valid = [r for r in run.results if r.roc_auc is not None]
                    if valid:
                        roc = float(np.mean([r.roc_auc for r in valid]))
                        pr = float(np.mean([r.pr_auc for r in valid]))
                        curve_records.append({
                            "task": key, "method": method, "k": k,
                            "n_folds_used": len(valid),
                            "roc_auc_mean": roc,
                            "pr_auc_mean": pr,
                        })
                    else:
                        roc = pr = float("nan")
                    print(f"  {key:<48} {method:<14} k={k:>2}  "
                          f"ROC {roc:.3f}  PR {pr:.3f}", flush=True)

    df = pd.DataFrame(fold_records)
    df.to_csv(args.out / "fold_scores.csv", index=False)
    pd.DataFrame(curve_records).to_csv(args.out / "curve.csv", index=False)

    # Macro per k
    rows = []
    for k in K_VALUES:
        for method in methods:
            vals = [r["roc_auc_mean"] for r in curve_records
                    if r["k"] == k and r["method"] == method and r["roc_auc_mean"] is not None]
            if not vals:
                continue
            arr = np.array(vals)
            rows.append({
                "k": k, "method": method, "n_tasks": len(arr),
                "roc_auc_mean": arr.mean(),
                "roc_auc_sd": arr.std(ddof=1) if len(arr) > 1 else 0.0,
                "pr_auc_mean": np.mean([r["pr_auc_mean"] for r in curve_records
                                         if r["k"] == k and r["method"] == method]),
            })
    pd.DataFrame(rows).to_csv(args.out / "macro_curve.csv", index=False)

    (args.out / "run_record.json").write_text(json.dumps({
        "encoder": analyser.ref.to_dict(),
        "k_values": list(K_VALUES),
        "n_repeats": N_REPEATS, "n_folds": N_FOLDS,
        "methods": methods,
    }, indent=2, default=str))

    print(f"\nwrote {args.out}/fold_scores.csv, curve.csv, macro_curve.csv, run_record.json")


if __name__ == "__main__":
    main()
