"""Two-hour postprandial response from a frozen representation. Paper §4.3, blueprint §19.10.

    uv run python scripts/ppgr.py --checkpoint runs_5090/rawstats120/ckpt_ep040.pt

Reports MAE for the trajectory and for the three derived endpoints, per sensor and as the
equal-weight two-sensor mean the paper uses, across a cumulative context ladder:

    nutrition only -> + subject context -> + one hour of pre-meal CGM -> + the representation

The ladder is the point. A head given carbohydrate grams and a person's BMI already predicts a
good deal, and the only question worth asking of a foundation model here is what it adds *on top
of* that. Reporting the final number alone would credit the encoder with everything the meal log
already knew.

Two controls sit underneath: predicting no change at all, and predicting each subject's own mean
response. The second is the one that matters -- a model that cannot beat "this person usually
rises 50 mg/dL" has learned nothing about the meal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from opencgm_stateevent.eval import ppgr
from opencgm_stateevent.eval.embed import load_encoder
from opencgm_stateevent.eval.splits import repeat_seed

#: Cumulative context stages. Each adds a block to the previous one. §4.3 Table 4.
STAGES = ("nutrition", "+subject", "+recent_cgm", "+representation")

#: "the same MLP with two hidden layers" (§4.3). Widths are unpublished.
HIDDEN = (128, 64)
N_FOLDS = 5
N_INITS = 10


def features(dataset: ppgr.PPGRDataset, embedding: np.ndarray, stage: str) -> np.ndarray:
    blocks = [dataset.nutrition]
    if stage in ("+subject", "+recent_cgm", "+representation"):
        blocks.append(dataset.subject_context)
    if stage in ("+recent_cgm", "+representation"):
        # The hour before the meal, plus the value the target is measured against. Without the
        # baseline the head cannot know whether a rise of 40 starts from 90 or from 180.
        blocks.append(dataset.premeal)
        blocks.append(dataset.baseline_glucose.reshape(-1, 1))
    if stage == "+representation":
        blocks.append(embedding)
    return np.hstack(blocks)


def subject_folds(subjects: np.ndarray, n_folds: int = N_FOLDS) -> list[np.ndarray]:
    """Subject-disjoint folds. Every event of a person lands on one side. §4.3."""
    unique = np.unique(subjects)
    rng = np.random.default_rng(repeat_seed("ppgr", 0))
    order = rng.permutation(len(unique))
    assignment = {unique[i]: k % n_folds for k, i in enumerate(order)}
    return [np.array([assignment[s] == f for s in subjects]) for f in range(n_folds)]


def embed_windows(model, dataset: ppgr.PPGRDataset, device: str) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(dataset), 256):
            stop = start + 256
            out.append(model.encode(
                torch.from_numpy(dataset.window_values[start:stop]).to(device),
                torch.from_numpy(dataset.window_mask[start:stop]).to(device),
                torch.from_numpy(dataset.circadian[start:stop]).to(device),
            ).contextual_tokens.mean(dim=1).float().cpu().numpy())
    return np.vstack(out)


def evaluate_stage(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, cadence: int) -> dict:
    """Five subject-disjoint folds, ten head initialisations each. §4.3."""
    predictions = np.zeros((N_INITS, *y.shape))
    for test in subject_folds(subjects):
        train = ~test
        scaler = StandardScaler().fit(x[train])
        a, b = scaler.transform(x[train]), scaler.transform(x[test])
        for init in range(N_INITS):
            head = MLPRegressor(
                hidden_layer_sizes=HIDDEN, max_iter=600, early_stopping=True,
                n_iter_no_change=20, random_state=init, learning_rate_init=1e-3,
            )
            head.fit(a, y[train])
            predictions[init][test] = head.predict(b)

    per_init = [ppgr.score(p, y, cadence) for p in predictions]
    return {
        key: float(np.mean([s[key] for s in per_init]))
        for key in per_init[0]
    } | {
        f"{key}_sd": float(np.std([s[key] for s in per_init], ddof=1))
        for key in per_init[0]
    }


def controls(dataset: ppgr.PPGRDataset, cadence: int) -> dict:
    """No-change, and each subject's own mean response estimated out of fold."""
    y = dataset.target
    out = {"no_change": ppgr.score(np.zeros_like(y), y, cadence)}

    subject_mean = np.zeros_like(y)
    for test in subject_folds(dataset.subjects):
        train = ~test
        # A held-out subject has no history in the training split, so the fallback is the
        # training population's mean curve; a subject present in training gets their own.
        population = y[train].mean(axis=0)
        for subject in np.unique(dataset.subjects[test]):
            rows = dataset.subjects == subject
            history = rows & train
            subject_mean[rows & test] = (
                y[history].mean(axis=0) if history.any() else population
            )
    out["subject_mean_curve"] = ppgr.score(subject_mean, y, cadence)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("reports/eval/ppgr.json"))
    args = ap.parse_args()

    print("building matched meal events ...", flush=True)
    datasets = ppgr.build_matched()
    model, ref = load_encoder(args.checkpoint, device=args.device)

    results: dict[str, dict] = {}
    for sensor, dataset in datasets.items():
        cadence = ppgr.SENSORS[sensor]["cadence"]
        print(f"\n=== {sensor}: {len(dataset)} matched events, "
              f"{len(set(dataset.subjects))} subjects, {dataset.target.shape[1]} outputs "
              f"at {cadence} min ===", flush=True)

        embedding = embed_windows(model, dataset, args.device)
        results[sensor] = {"n_events": len(dataset),
                           "n_subjects": len(set(dataset.subjects)),
                           "controls": controls(dataset, cadence), "stages": {}}

        for name, scores in results[sensor]["controls"].items():
            print(f"  {'control: ' + name:<24} trajectory {scores['trajectory_mae']:6.2f}  "
                  f"iAUC {scores['iauc_mae']:6.2f}  peak {scores['peak_rise_mae']:6.2f}  "
                  f"peak-time {scores['peak_time_mae']:6.1f}")

        for stage in STAGES:
            x = features(dataset, embedding, stage)
            scores = evaluate_stage(x, dataset.target, dataset.subjects, cadence)
            results[sensor]["stages"][stage] = scores | {"n_features": int(x.shape[1])}
            print(f"  {stage:<24} trajectory {scores['trajectory_mae']:6.2f} "
                  f"+/-{scores['trajectory_mae_sd']:.2f}  "
                  f"iAUC {scores['iauc_mae']:6.2f}  peak {scores['peak_rise_mae']:6.2f}  "
                  f"peak-time {scores['peak_time_mae']:6.1f}   ({x.shape[1]} features)",
                  flush=True)

    print("\nequal-weight two-sensor mean, as §4.3 reports it")
    both = {}
    for stage in STAGES:
        both[stage] = {
            metric: float(np.mean([results[s]["stages"][stage][metric] for s in datasets]))
            for metric in ("trajectory_mae", "iauc_mae", "peak_rise_mae", "peak_time_mae")
        }
        m = both[stage]
        print(f"  {stage:<24} trajectory {m['trajectory_mae']:6.2f}  iAUC {m['iauc_mae']:6.2f}  "
              f"peak {m['peak_rise_mae']:6.2f}  peak-time {m['peak_time_mae']:6.1f}")

    gain = (both["+recent_cgm"]["trajectory_mae"] - both["+representation"]["trajectory_mae"])
    print(f"\nthe representation changes trajectory MAE by {-gain:+.2f} mg/dL over the same head "
          f"without it ({gain / both['+recent_cgm']['trajectory_mae']:+.1%})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"encoder": ref.to_dict(), "per_sensor": results, "two_sensor_mean": both},
        indent=2, default=str,
    ))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
