"""Is the downstream benchmark measuring anything? A subject-label permutation test.

Every probe number in this project rests on one assumption: that the evaluation can tell a good
representation from a bad one. Six very different training configurations all scored within
+/-0.014 of an untrained encoder, which has two possible explanations. Either none of them
learned anything, or the benchmark cannot distinguish them.

This separates the two. Subject labels are permuted *before* fold construction, so every
structural property is preserved -- same windows, same subjects, same windows-per-subject
imbalance, same class balance, same fold sizes -- and only the association between a subject and
their label is destroyed. A sound benchmark must then score at chance.

If permuted labels score near 0.5, the benchmark is sound and the models genuinely are not
learning. If they score near the real number, the benchmark is broken and every downstream figure
so far is meaningless. There is no third reading, which is what makes this worth the compute.

    uv run python scripts/permutation_test.py --checkpoint runs/fixed_seed17/ckpt_ep010.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from opencgm_stateevent.eval import baselines, labels
from opencgm_stateevent.eval.embed import load_encoder, load_or_embed
from opencgm_stateevent.eval.probe import HEADLINE, run_probe
from opencgm_stateevent.eval.splits import build_folds
from opencgm_stateevent.eval.windows import build_all

SOURCES_FOR = {
    "cgmacros": ("cgmacros_dexcom", "cgmacros_libre"),
    "hall": ("hall",),
    "stanford": ("stanford",),
    "shanghai_t2dm": ("shanghai_t2dm",),
}


def score_once(
    features_by_source: dict[str, dict[str, np.ndarray]],
    window_sets,
    label_tables,
    *,
    method: str,
    permute_seed: int | None,
    n_repeats: int,
) -> float:
    """Mean ROC-AUC across every dataset-task, optionally with labels permuted per subject."""
    scores = []
    for task in labels.TASKS:
        table = label_tables[task.dataset]
        for source in SOURCES_FOR[task.dataset]:
            ws = window_sets[source]
            series = table.dropna(subset=[task.name]).set_index("entry")[task.name]
            keys = ws.entries if ws.entries is not None else ws.subjects
            keep = np.array([k in series.index for k in keys])
            if not keep.any():
                continue

            window_subjects = np.asarray(ws.subjects[keep])
            window_labels = series.loc[list(keys[keep])].to_numpy().astype(int)
            subjects, first = np.unique(window_subjects, return_index=True)
            subject_labels = window_labels[first]
            if len(np.unique(subject_labels)) < 2:
                continue

            if permute_seed is not None:
                # Permute the subject -> label map, then rebroadcast to windows. Structure is
                # untouched; only the association is destroyed.
                # SHA-256, not hash(): Python randomises string hashes per process unless
                # PYTHONHASHSEED is fixed, which would make this test unreproducible.
                digest = hashlib.sha256(task.key.encode()).digest()
                rng = np.random.default_rng(
                    [permute_seed, int.from_bytes(digest[:4], "big")]
                )
                shuffled = subject_labels[rng.permutation(len(subject_labels))]
                lookup = dict(zip(subjects, shuffled, strict=True))
                subject_labels = shuffled
                window_labels = np.array([lookup[s] for s in window_subjects])

            key = f"{task.key}[{source}]"
            folds = build_folds(key, subjects, subject_labels, n_repeats=n_repeats)
            run = run_probe(
                folds, features_by_source[source][method][keep], window_subjects, window_labels,
                task=key, method=method, n_classes=task.n_classes, cfg=HEADLINE,
            )
            values = run.scores("roc_auc")
            if len(values):
                scores.append(float(values.mean()))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--permutations", type=int, default=200,
                    help="p is floored at 1/(B+1), so B=10 cannot go below 0.091")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("reports/eval/permutation_test.json"))
    args = ap.parse_args()

    window_sets = build_all()
    label_tables = labels.build_all()
    model, ref = load_encoder(args.checkpoint, device=args.device)

    features = {}
    for name, ws in window_sets.items():
        embedded = load_or_embed(model, ws, ref, device=args.device)
        features[name] = {
            "opencgm_mean": embedded["mean"],
            "clinical_metrics": baselines.build("clinical_metrics", ws),
            "raw_masked": baselines.build("raw_masked", ws),
        }

    results = {}
    for method in ("opencgm_mean", "clinical_metrics", "raw_masked"):
        real = score_once(features, window_sets, label_tables, method=method,
                          permute_seed=None, n_repeats=args.repeats)
        null = [
            score_once(features, window_sets, label_tables, method=method,
                       permute_seed=p, n_repeats=args.repeats)
            for p in range(args.permutations)
        ]
        null = np.array(null)
        # Finite-sample one-sided empirical p, (1 + #{perm >= real}) / (B + 1). The raw fraction
        # can report 0.000, which is not supportable from B permutations: with B=10 the smallest
        # attainable value is 1/11 = 0.091. The floor is a property of the estimator, not of the
        # evidence, and quoting 0.000 overstates it.
        p_value = float((1 + int((null >= real).sum())) / (len(null) + 1))
        results[method] = {
            "real": real, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_min": float(null.min()), "null_max": float(null.max()),
            "p_value": p_value, "permutations": len(null),
        }
        print(
            f"{method:<18} real {real:.4f}   permuted {null.mean():.4f} "
            f"+/- {null.std(ddof=1):.4f}  (range {null.min():.4f}-{null.max():.4f})  "
            f"p={p_value:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "encoder": ref.to_dict(), "results": results},
        indent=2, default=str,
    ))
    print(f"\nwrote {args.out}")
    verdict = results["opencgm_mean"]
    if verdict["null_mean"] > 0.56:
        print("WARNING: permuted labels score well above chance. The benchmark is leaking.")
    else:
        print("Permuted labels score at chance: the benchmark discriminates as intended.")


if __name__ == "__main__":
    main()
