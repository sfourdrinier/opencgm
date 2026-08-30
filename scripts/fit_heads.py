"""Fit one deployable probe head per dataset-task, each carrying its measured reliability.

Evaluation fits a probe inside every cross-validation fold and throws it away; nothing in the
repository has ever produced a head you could actually run on a new person. This does.

Two things are produced together and must stay together:

  * the head -- a logistic regression on the frozen 128-dimension daily embedding, fitted on
    every labelled subject of that dataset-task, which is the most data available and therefore
    the best head we can ship;
  * its reliability -- the subject-grouped cross-validated ROC-AUC with a spread, which is the
    measured statement of how well that head does on people it has never seen.

The head fitted on all subjects cannot be scored on those same subjects, so the reliability comes
from the cross-validation and the head comes from the full fit. Shipping the first number without
the second would be indefensible: `hall:glucotype` and `cgmacros:hyperlipidemia` produce
identical-looking probabilities and one of them is worth 0.89 while the other is below chance.

    uv run python scripts/fit_heads.py --checkpoint runs/rawstats120/ckpt_last.pt
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from opencgm_stateevent.eval import labels
from opencgm_stateevent.eval.embed import load_encoder, load_or_embed
from opencgm_stateevent.eval.probe import HEADLINE, make_pipeline, run_probe
from opencgm_stateevent.eval.splits import build_folds
from opencgm_stateevent.eval.windows import build_all

SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}

#: Below this cross-validated ROC-AUC a head is kept but flagged: it carries no usable signal, and
#: a probability from it should be displayed as "no signal detected in our cohort" rather than as
#: a percentage. 0.55 is a judgement call, recorded here rather than buried in a UI.
SIGNAL_FLOOR = 0.55


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/heads.pkl"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    window_sets = build_all()
    label_tables = labels.build_all()
    model, ref = load_encoder(args.checkpoint, device=args.device)

    features = {}
    for name, ws in window_sets.items():
        features[name] = load_or_embed(model, ws, ref, device=args.device)["mean"]

    heads: dict[str, dict] = {}
    for task in labels.TASKS:
        table = label_tables[task.dataset]
        for source in SOURCES_FOR[task.dataset]:
            ws = window_sets[source]
            series = table.dropna(subset=[task.name]).set_index("entry")[task.name]
            keys = ws.entries if ws.entries is not None else ws.subjects
            keep = np.array([k in series.index for k in keys])
            if not keep.any():
                continue

            x = features[source][keep]
            y = series.loc[list(keys[keep])].to_numpy().astype(int)
            subjects = np.asarray(ws.subjects[keep])
            unique, first = np.unique(subjects, return_index=True)
            if len(np.unique(y[first])) < 2:
                continue

            key = f"{task.key}[{source}]"
            folds = build_folds(key, unique, y[first], n_repeats=args.repeats)
            run = run_probe(folds, x, subjects, y, task=key, method="opencgm_mean",
                            n_classes=task.n_classes, cfg=HEADLINE)
            window_auc = run.scores("roc_auc")
            subject_auc = run.scores("roc_auc_subject")

            pipeline = make_pipeline(HEADLINE)
            pipeline.fit(x, y)

            # A head learned from 15-minute Libre windows (coverage ~0.31 on a 5-minute grid)
            # sees nothing resembling a dense Dexcom day (~0.84). Applied across that gap it
            # returns confident nonsense -- measured: the same subject-day scored 1% by the
            # Dexcom head and 100% by the Libre head for the identical question. The band is
            # recorded so inference can refuse rather than extrapolate.
            coverage = ws.mask[keep].mean(axis=1)
            heads[key] = {
                "pipeline": pipeline,
                "coverage_p05": float(np.percentile(coverage, 5)),
                "coverage_p95": float(np.percentile(coverage, 95)),
                "coverage_median": float(np.median(coverage)),
                "task": task.key,
                "dataset": task.dataset,
                "source": source,
                "n_classes": task.n_classes,
                "n_subjects": len(unique),
                "n_windows": int(keep.sum()),
                "class_balance": np.bincount(y[first]).tolist(),
                "roc_auc": float(window_auc.mean()) if len(window_auc) else float("nan"),
                "roc_auc_sd": (float(window_auc.std(ddof=1))
                               if len(window_auc) > 1 else float("nan")),
                "roc_auc_subject": float(subject_auc.mean()) if len(subject_auc) else float("nan"),
                "n_folds": len(window_auc),
                "has_signal": bool(len(window_auc) and window_auc.mean() >= SIGNAL_FLOOR),
            }
            flag = "" if heads[key]["has_signal"] else "   <- below signal floor"
            print(f"  {key:<48} n={heads[key]['n_subjects']:>3}  "
                  f"ROC-AUC {heads[key]['roc_auc']:.3f} "
                  f"(subject {heads[key]['roc_auc_subject']:.3f}){flag}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as fh:
        pickle.dump({"encoder": ref.to_dict(), "heads": heads,
                     "signal_floor": SIGNAL_FLOOR, "n_repeats": args.repeats}, fh)

    summary = {k: {m: v[m] for m in
                   ("roc_auc", "roc_auc_sd", "roc_auc_subject", "n_subjects", "n_windows",
                    "class_balance", "n_folds", "has_signal",
                    "coverage_p05", "coverage_p95", "coverage_median")}
               for k, v in heads.items()}
    args.out.with_suffix(".json").write_text(
        json.dumps({"encoder": ref.to_dict(), "signal_floor": SIGNAL_FLOOR,
                    "heads": summary}, indent=2, default=str)
    )
    usable = sum(h["has_signal"] for h in heads.values())
    print(f"\nwrote {args.out} and {args.out.with_suffix('.json')}")
    print(f"{len(heads)} heads, {usable} above the {SIGNAL_FLOOR} signal floor")


if __name__ == "__main__":
    main()
