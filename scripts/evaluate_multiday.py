"""§19.9 multiday: pool N consecutive 24h windows per subject, score downstream.

The paper's §19.9 asks: does aggregating more days into one embedding help downstream phenotype
prediction? Two natural questions:

  1. **Pure-pool**: average N adjacent 24h embeddings into one N-day embedding, fit the same
     probe on top. This is the linear benefit of more context.
  2. **Sequence-pool**: build a longer sequence of N*288 5-min positions, encode it whole, then
     mean-pool the resulting contextual tokens. This is what `analyse_stream` does at the
     application layer.

This script does (1) — pure-pool — because (2) would require retraining the encoder on
N*288 position sequences. The paper's encoder was trained on 288-position windows; using longer
inputs at inference time without retraining would test something other than what the paper
claims to test.

Procedure per task:
  * for each subject with at least N 24h windows, build an "N-day" embedding by averaging the
    first N consecutive windows (sessions are time-ordered, not random, so consecutive = same
    recording period),
  * use the same probe, fitted per (subject, task), scored on the same held-out subjects,
  * repeat for N in {1, 2, 3, 5, 7} days.

Single seed (43). The encoder is fixed, so seed variance is dominated by per-fold probe noise.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "scripts")

from evaluate import align

from opencgm_stateevent.eval import labels
from opencgm_stateevent.eval.embed import embed as encode_windows
from opencgm_stateevent.eval.probe import HEADLINE, FoldResult, _scores, make_pipeline
from opencgm_stateevent.eval.splits import build_folds
from opencgm_stateevent.eval.windows import build_all
from opencgm_stateevent.infer import Analyser

N_DAYS = (1, 2, 3, 5, 7)
N_REPEATS = 5
N_FOLDS = 5

SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}


def _aggregate_per_subject(
    features: np.ndarray,
    window_subjects: np.ndarray,
    window_sessions: np.ndarray,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each subject with >= n_days windows, build one n_day embedding by mean-pooling the
    first n_days consecutive windows (in session-time order). Returns (aggregated_features,
    aggregated_subjects) — each row is one (subject, n_days-window) pair.
    """
    rows = []
    out_subjects = []
    for subj in np.unique(window_subjects):
        idx = np.where(window_subjects == subj)[0]
        if len(idx) < n_days:
            continue
        # sort by session — windows are stored in session order
        idx_sorted = idx[np.argsort(window_sessions[idx])]
        # take the first n_days, mean-pool
        chunk = features[idx_sorted[:n_days]]
        rows.append(chunk.mean(axis=0))
        out_subjects.append(subj)
    if not rows:
        empty = np.empty((0, features.shape[1]), dtype=features.dtype)
        empty_subj = np.array([], dtype=window_subjects.dtype)
        return empty, empty_subj
    return np.stack(rows), np.array(out_subjects)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/eval/multiday_seed43"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading analyser ...", flush=True)
    analyser = Analyser.load(args.checkpoint, device="cpu")
    model = analyser.model

    print("building windows + embeddings ...", flush=True)
    window_sets = build_all()
    label_tables = labels.build_all()
    daily_features: dict[str, np.ndarray] = {}
    for name, ws in window_sets.items():
        daily_features[name] = encode_windows(model, ws, device="cpu", poolings=("mean",))["mean"]

    fold_records = []
    curve_records = []
    for task in labels.TASKS:
        table = label_tables[task.dataset]
        for source in SOURCES_FOR[task.dataset]:
            ws = window_sets[source]
            aligned = align(ws, table, task.name)
            if aligned is None:
                continue
            keep, window_labels, window_subjects = aligned
            feats_d = daily_features[source][keep]
            # Aggregate per n_days
            for n_days in N_DAYS:
                agg_feats, agg_subjects = _aggregate_per_subject(
                    feats_d, window_subjects, ws.sessions[keep], n_days,
                )
                if len(agg_subjects) < 10:
                    continue
                # Build per-subject labels (first-window label is the subject-level label)
                subj_label = {}
                for s, lab in zip(window_subjects, window_labels, strict=True):
                    if s not in subj_label:
                        subj_label[s] = lab
                labels_for_subj = np.array([subj_label[s] for s in agg_subjects])
                if len(np.unique(labels_for_subj)) < 2:
                    continue

                key = f"{task.key}[{source}]"
                folds = build_folds(key, agg_subjects, labels_for_subj,
                                    n_repeats=N_REPEATS, n_folds=N_FOLDS)
                for fold in folds:
                    train_mask = np.isin(agg_subjects, list(fold.train_subjects))
                    test_mask = np.isin(agg_subjects, list(fold.test_subjects))
                    if not train_mask.any() or not test_mask.any():
                        continue
                    y_train = labels_for_subj[train_mask]
                    if len(np.unique(y_train)) < 2:
                        continue
                    pipeline = make_pipeline(HEADLINE)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            pipeline.fit(agg_feats[train_mask], y_train)
                            proba = pipeline.predict_proba(agg_feats[test_mask])
                            pred = pipeline.predict(agg_feats[test_mask])
                        except Exception:
                            continue
                    scores = _scores(labels_for_subj[test_mask], proba, pred,
                                      n_classes=task.n_classes)
                    fr = FoldResult(
                        task=key, method=f"multiday_n{n_days}",
                        repeat=fold.repeat, fold=fold.fold,
                        n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
                        n_test_subjects=len(fold.test_subjects),
                        pr_auc=scores.get("pr_auc"),
                        roc_auc=scores.get("roc_auc"),
                        macro_f1=scores.get("macro_f1"),
                    )
                    fold_records.append(asdict(fr))
                    valid = [r for r in fold_records if r.get("method") == f"multiday_n{n_days}"
                             and r.get("task") == key and r.get("roc_auc") is not None]
                    if valid:
                        roc = float(np.mean([r["roc_auc"] for r in valid]))
                        pr = float(np.mean([r["pr_auc"] for r in valid]))
                        curve_records.append({
                            "task": key, "n_days": n_days,
                            "n_subjects_aggregated": len(agg_subjects),
                            "roc_auc_mean": roc,
                            "pr_auc_mean": pr,
                        })
                print(f"  {key:<48} n={n_days}d  "
                      f"subj={len(agg_subjects):>3}  "
                      f"ROC {roc:.3f}  PR {pr:.3f}", flush=True)

    pd.DataFrame(fold_records).to_csv(args.out / "fold_scores.csv", index=False)
    pd.DataFrame(curve_records).to_csv(args.out / "curve.csv", index=False)

    # Macro per n_days
    rows = []
    for n_days in N_DAYS:
        vals = [r["roc_auc_mean"] for r in curve_records
                if r["n_days"] == n_days and r["roc_auc_mean"] is not None]
        if not vals:
            continue
        arr = np.array(vals)
        rows.append({
            "n_days": n_days,
            "n_tasks": len(arr),
            "roc_auc_mean": float(arr.mean()),
            "roc_auc_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "pr_auc_mean": float(np.mean([r["pr_auc_mean"] for r in curve_records
                                          if r["n_days"] == n_days])),
        })
    pd.DataFrame(rows).to_csv(args.out / "macro_curve.csv", index=False)

    (args.out / "run_record.json").write_text(json.dumps({
        "encoder": analyser.ref.to_dict(),
        "n_days": list(N_DAYS),
        "n_repeats": N_REPEATS,
        "n_folds": N_FOLDS,
    }, indent=2, default=str))

    print(f"\nwrote {args.out}/fold_scores.csv, curve.csv, macro_curve.csv, run_record.json")


if __name__ == "__main__":
    main()
