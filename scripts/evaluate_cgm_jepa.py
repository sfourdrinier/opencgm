"""Evaluate the CGM-JEPA comparator on the same downstream benchmark as our own model.

The point is the head-to-head: subject-disjoint 5-fold x 10 repeats, identical folds to the GlucoFM
evaluations that already live in `reports/eval/seed*_ep120_full/`, so the paired comparison is
structural rather than correlational.

Two things deliberately diverge from `scripts/evaluate.py`:

  * **The encoder.** `load_encoder_cgm_jepa` reads the comparator's own checkpoint and embeds
    pre-projection, mean-pooled, as the authors specify (`models/encoder.py:84`, appendix B).
  * **The pooling.** The authors produce a 96-dim embedding from a 24-token sequence; we expose it
    as the single `cgm_jepa` method rather than three poolings, because the comparator is what it
    is. Reading the 96-dim embedding into the same 256-dim `mean_max` slot would be a different
    representation and would muddy the head-to-head.

`raw_masked` and `clinical_metrics` baselines still come from `opencgm_stateevent.eval.baselines`,
so they share the exact fold structure with every other GlucoFM eval. That is what makes the
paired comparison meaningful.
"""

from __future__ import annotations

import argparse
import hashlib

# `scripts/` has no __init__.py by design; import the shared helpers directly from the module
# file. Adding an __init__.py would put scripts on sys.path and risk import shadowing elsewhere.
import importlib.util
import json
import sys as _sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from opencgm_stateevent.baselines.cgm_jepa import CGMJEPA, PATCH_SIZE, PATCHES
from opencgm_stateevent.eval import baselines, labels
from opencgm_stateevent.eval.embed import EncoderRef
from opencgm_stateevent.eval.probe import HEADLINE, run_probe
from opencgm_stateevent.eval.splits import build_folds, fold_manifest
from opencgm_stateevent.eval.stats import compare, holm_adjust
from opencgm_stateevent.eval.windows import build_all
from opencgm_stateevent.train.dataset import interpolate_dense


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_eval_mod = _load("scripts_evaluate", "scripts/evaluate.py")
paired_macro = _eval_mod.paired_macro
task_bootstrap_ci = _eval_mod.task_bootstrap_ci
align = _eval_mod.align

SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}


def load_encoder_cgm_jepa(checkpoint: Path, device: str) -> tuple[CGMJEPA, EncoderRef]:
    """Load the comparator's frozen online encoder.

    The official release ships both an online and an EMA target; the *online* branch is what
    appendix B specifies for downstream use, and it is what we train and save. Loading both and
    choosing the wrong one would silently regress the comparison; this is the single most
    substituted-encoder risk in the whole baseline.
    """
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CGMJEPA()
    if "state_dict" in state:
        model.encoder.load_state_dict(state["state_dict"])
    elif "model" in state:
        model.encoder.load_state_dict({
            k.removeprefix("encoder."): v for k, v in state["model"].items()
            if k.startswith("encoder.")
        })
    else:
        model.encoder.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    encoded = model.encoder.state_dict()
    digest = hashlib.sha256()
    for k in sorted(encoded):
        digest.update(k.encode())
        digest.update(encoded[k].detach().cpu().numpy().tobytes())
    config = state.get("config", {})
    return model, EncoderRef(
        checkpoint=checkpoint,
        weights_sha256=digest.hexdigest(),
        epoch=int(config.get("epochs", -1)),
        seed=int(config.get("seed", -1)),
        git_sha="unknown",  # the comparator's training didn't use our git tracker
        architecture=json.dumps({
            "model": "cgm_jepa",
            "embed_dim": config.get("embed_dim", 96),
            "n_layers": config.get("n_layers", 3),
            "predictor_dim": config.get("predictor_dim", 48),
        }, sort_keys=True),
    )


@torch.no_grad()
def embed_cgm_jepa(model: CGMJEPA, ws, device: str) -> np.ndarray:
    """Mean-pooled 96-dim embedding per window.

    Fills gaps via `interpolate_dense` first -- that is the authors' preprocessing, and it is the
    substantive contrast with GlucoFM. The fill happens here rather than in `train.dataset` so the
    evaluation's data path matches the official comparator's without leaving a permissive fill
    sitting next to a strict model elsewhere.
    """
    out = np.zeros((len(ws), 96), dtype=np.float32)
    for start in range(0, len(ws), 256):
        stop = start + 256
        filled = np.stack([
            interpolate_dense(v.astype(np.float32, copy=False), m)[0]
            for v, m in zip(ws.values[start:stop], ws.mask[start:stop], strict=True)
        ])
        patches = filled.reshape(-1, PATCHES, PATCH_SIZE)
        tensor = torch.from_numpy(patches).to(device)
        out[start:stop] = model.embed(tensor).float().cpu().numpy()
    return out


def evaluate(checkpoint: Path, out: Path, *, device: str, quick: bool) -> None:
    n_repeats = 2 if quick else 10
    out.mkdir(parents=True, exist_ok=True)

    print("building downstream windows ...", flush=True)
    window_sets = build_all()
    label_tables = labels.build_all()

    print(f"loading CGM-JEPA encoder from {checkpoint} ...", flush=True)
    model, ref = load_encoder_cgm_jepa(checkpoint, device=device)
    print(f"  epoch {ref.epoch}, seed {ref.seed}, provenance tag {ref.tag}", flush=True)

    features: dict[str, dict[str, np.ndarray]] = {}
    for name, ws in window_sets.items():
        features[name] = {
            "cgm_jepa": embed_cgm_jepa(model, ws, device=device),
            "clinical_metrics": baselines.build("clinical_metrics", ws),
            "raw_masked": baselines.build("raw_masked", ws),
        }

    methods = ["cgm_jepa", "clinical_metrics", "raw_masked"]

    fold_records, all_folds, runs = [], [], {}
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
            summary = runs[(key, "cgm_jepa")].summary()
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
        if method != "cgm_jepa":
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

    macro = []
    for weighting, by_dataset in (("per_entry", False), ("per_dataset", True)):
        for method in methods:
            if method != "cgm_jepa":
                continue
            for baseline in ("clinical_metrics", "raw_masked"):
                for metric in ("pr_auc", "roc_auc", "macro_f1"):
                    a, b, _ = paired_macro(runs, method, baseline, metric, by_dataset=by_dataset)
                    if len(a) < 2:
                        continue
                    c = compare(a, b, task=f"MACRO[{weighting}]", metric=metric,
                                method=method, baseline=baseline)
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
    headline = [c for c in comparisons if c.method == "cgm_jepa" and c.metric == "roc_auc"]
    wins = sum(c.mean_difference > 0 for c in headline)
    significant = sum(c.significant and c.mean_difference > 0 for c in headline)
    print(f"cgm_jepa vs baselines, ROC-AUC: {wins}/{len(headline)} ahead, "
          f"{significant} significant after Holm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = args.out or Path("reports/eval") / args.checkpoint.parent.name
    evaluate(args.checkpoint, out, device=args.device, quick=args.quick)


if __name__ == "__main__":
    main()
