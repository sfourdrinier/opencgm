"""§19.8 cross-dataset: train a probe on one cohort's labelled subjects, score a different cohort.

The question: how much of what the encoder learned transfers across populations? A probe fitted
on cgmacros-Dexcom's insulin-resistance labels tested on stanford's insulin-resistance labels
asks the encoder to embed both cohorts into a space where the *same* linear hyperplane separates
IR+ from IR-. If the encoder learned something population-specific, the hyperplane won't
transfer; if it learned a physiological signal, it will.

Tasks with >=2 cohorts sharing the same label name:

  diabetes_risk:      cgmacros_dexcom/libre, hall, stanford  (mixed: 3-class vs 2-class)
  insulin_resistance: cgmacros_dexcom/libre, hall, stanford, shanghai_t2dm
  hyperlipidemia:     cgmacros_dexcom/libre, hall, shanghai_t2dm

Class-set mismatches (cgmacros:diabetes_risk is 3-class, hall/stanford are 2-class) are
reported and skipped, not silently zeroed.

The probe is the same logistic regression used in `evaluate.py` (ProbeConfig HEADLINE), fitted
once on all labelled subjects of the source cohort, scored once on all labelled subjects of the
target cohort. No resampling; this is a single transfer score, not a cross-validated one.

This is *not* a fair test of the encoder's robustness to distribution shift on its own -- the
probes are cohort-specific linear scalings on top of the encoder -- but it does tell us whether
the *encoder* has produced an embedding where a single per-task linear decision is portable.

Single seed (43). Variance is dominated by the cohort size, not the encoder seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "scripts")

from evaluate import align

from opencgm_stateevent.eval import labels
from opencgm_stateevent.eval.embed import embed as encode_windows
from opencgm_stateevent.eval.probe import HEADLINE, _scores, make_pipeline
from opencgm_stateevent.eval.windows import build_all
from opencgm_stateevent.infer import Analyser

SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}


def _fit_on_src_score_on_tgt(
    src_x, src_y, src_subj, src_subjects_uniq,
    tgt_x, tgt_y, tgt_subj, n_classes,
):
    """Fit a probe on src, score on tgt. Returns a scores dict or None."""
    train_mask = np.isin(src_subj, src_subjects_uniq)
    if not train_mask.any():
        return None
    y_train = src_y[train_mask]
    if len(np.unique(y_train)) < 2:
        return None
    # Target may collapse to a single class in practice; AUC is undefined there.
    if len(np.unique(tgt_y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "macro_f1": float("nan"),
                "note": "single-class target"}
    pipeline = make_pipeline(HEADLINE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pipeline.fit(src_x[train_mask], y_train)
            proba = pipeline.predict_proba(tgt_x)
            pred = pipeline.predict(tgt_x)
        except Exception:
            return None
        try:
            return _scores(tgt_y, proba, pred, n_classes=n_classes)
        except ValueError:
            # sklearn: y_true / proba class-count mismatch (e.g. target collapsed after fit)
            return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                    "macro_f1": float("nan"), "note": "score mismatch"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/eval/cross_dataset_seed43"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading analyser ...", flush=True)
    analyser = Analyser.load(args.checkpoint, device="cpu")
    model = analyser.model

    print("building windows + embeddings ...", flush=True)
    window_sets = build_all()
    label_tables = labels.build_all()
    features: dict[str, np.ndarray] = {}
    for name, ws in window_sets.items():
        features[name] = encode_windows(model, ws, device="cpu", poolings=("mean",))["mean"]

    # Per task: list of (cohort, embeddings, labels, subjects, unique_subjects, subject_labels)
    aligned_per_task: dict[str, list[tuple]] = {}
    for task in labels.TASKS:
        table = label_tables[task.dataset]
        for source in SOURCES_FOR[task.dataset]:
            ws = window_sets[source]
            aligned = align(ws, table, task.name)
            if aligned is None:
                continue
            keep, window_labels, window_subjects = aligned
            subjects, first = np.unique(window_subjects, return_index=True)
            subject_labels = window_labels[first]
            if len(np.unique(subject_labels)) < 2:
                continue
            aligned_per_task.setdefault(task.name, []).append(
                (source, features[source][keep], window_labels,
                 window_subjects, subjects, subject_labels)
            )

    rows = []
    for task_name, cohort_list in aligned_per_task.items():
        if len(cohort_list) < 2:
            continue
        for i, cohort_i in enumerate(cohort_list):
            for j, cohort_j in enumerate(cohort_list):
                if i == j:
                    continue
                src_name, src_x, src_y, src_subj, src_subj_uniq, _ = cohort_i
                tgt_name, tgt_x, tgt_y, tgt_subj, tgt_subj_uniq, _ = cohort_j
                # Class-set must match -- cgmacros:diabetes_risk is 3-class, hall/stanford are 2.
                src_classes = set(np.unique(src_y).tolist())
                tgt_classes = set(np.unique(tgt_y).tolist())
                if src_classes != tgt_classes:
                    print(
                        f"  {task_name:<24} {src_name} -> {tgt_name:<18}  "
                        f"(class-set mismatch: {src_classes} vs {tgt_classes})",
                        flush=True,
                    )
                    continue
                task_obj = next(t for t in labels.TASKS if t.name == task_name)
                scores = _fit_on_src_score_on_tgt(
                    src_x, src_y, src_subj, src_subj_uniq,
                    tgt_x, tgt_y, tgt_subj, task_obj.n_classes,
                )
                if scores is None:
                    print(
                        f"  {task_name:<24} {src_name} -> {tgt_name:<18}  (skipped)",
                        flush=True,
                    )
                    continue
                rows.append({
                    "task": task_name,
                    "source_cohort": src_name,
                    "target_cohort": tgt_name,
                    "n_source_subjects": len(src_subj_uniq),
                    "n_target_subjects": len(tgt_subj_uniq),
                    "roc_auc": scores.get("roc_auc"),
                    "pr_auc": scores.get("pr_auc"),
                    "macro_f1": scores.get("macro_f1"),
                })
                print(
                    f"  {task_name:<24} {src_name} -> {tgt_name:<18}  "
                    f"ROC {scores.get('roc_auc', float('nan')):.3f}  "
                    f"PR {scores.get('pr_auc', float('nan')):.3f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "transfers.csv", index=False)

    summary = []
    for task_name, g in df.groupby("task"):
        summary.append({
            "task": task_name,
            "n_transfers": len(g),
            "roc_auc_mean": g["roc_auc"].mean(),
            "roc_auc_sd": g["roc_auc"].std(ddof=1) if len(g) > 1 else 0.0,
            "pr_auc_mean": g["pr_auc"].mean(),
            "pr_auc_sd": g["pr_auc"].std(ddof=1) if len(g) > 1 else 0.0,
        })
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)

    (args.out / "run_record.json").write_text(json.dumps({
        "encoder": analyser.ref.to_dict(),
        "n_transfers": len(df),
        "tasks": list(df["task"].unique()) if len(df) else [],
    }, indent=2, default=str))

    print(f"\nwrote {args.out}/transfers.csv, summary.csv, run_record.json")
    if summary:
        sdf = pd.DataFrame(summary)
        print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
