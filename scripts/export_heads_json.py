"""Export the fitted heads bundle to a JSON file consumable from TypeScript.

`artifacts/heads.pkl` ships as a pickle of sklearn `Pipeline` objects — fine for Python, useless
in a browser. This script flattens every fitted head into a JSON object the Next.js demo (and
any TypeScript consumer) can read.

The output schema mirrors what the consumer needs to apply each head to a 128-d embedding:

    {
      "encoder":  { ... provenance ... },   // same as the existing heads.json
      "signal_floor": 0.55,
      "n_repeats": 10,
      "heads": {
        "<task>[<source>]": {
          "task": "...",
          "dataset": "...",
          "source": "...",
          "n_classes": N,
          "intermediate": ["scale"],          // tells the consumer the order of stages
          "scale":   {"mean": [..128..], "scale": [..128..]},   // from StandardScaler
          "classifier": {
            "coef":      [[..128..], ...],    // [K, 128] from LogReg
            "intercept": [..K..],
            "classes":   [..K..]
          },
          "reliability": {
            "roc_auc": ...,
            "roc_auc_sd": ...,
            "roc_auc_subject": ...,
            "n_subjects": ...,
            "n_windows": ...,
            "n_folds": ...,
            "has_signal": bool
          },
          "applicability": {
            "coverage_p05": ..., "coverage_p95": ..., "coverage_median": ...
          },
          "class_balance": [...]
        }
      }
    """

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def pipeline_to_dict(pipeline: Any) -> dict[str, Any]:
    """Flatten a fitted sklearn Pipeline (scale → classifier) into a JSON-safe dict.

    The consumer applies the steps in order:

        x_scaled = (x - scale.mean) / scale.scale
        logits   = x_scaled @ coef.T + intercept          (coef is [K, 128])
        proba    = softmax(logits)                 (binary: sigmoid; multi: argmax-of-softmax)
    """
    steps = {name: pipeline.named_steps[name] for name in pipeline.named_steps}
    out: dict[str, Any] = {"intermediate": list(pipeline.named_steps.keys())}

    if "scale" in steps:
        scaler = steps["scale"]
        out["scale"] = {
            "mean": np.asarray(scaler.mean_).tolist(),
            "scale": np.asarray(scaler.scale_).tolist(),
        }

    clf = steps["classifier"]
    out["classifier"] = {
        "coef": np.asarray(clf.coef_).tolist(),
        "intercept": np.asarray(clf.intercept_).tolist(),
        "classes": [int(c) for c in np.asarray(clf.classes_).tolist()],
    }
    return out


def head_to_dict(head: dict[str, Any]) -> dict[str, Any]:
    """Flatten one fitted head's metadata + pipeline into a single consumer-ready dict."""
    out = pipeline_to_dict(head["pipeline"])
    out.update({
        "task": head["task"],
        "dataset": head["dataset"],
        "source": head["source"],
        "n_classes": int(head["n_classes"]),
        "reliability": {
            "roc_auc": head.get("roc_auc"),
            "roc_auc_sd": head.get("roc_auc_sd"),
            "roc_auc_subject": head.get("roc_auc_subject"),
            "n_subjects": int(head.get("n_subjects", 0)),
            "n_windows": int(head.get("n_windows", 0)),
            "n_folds": int(head.get("n_folds", 0)),
            "has_signal": bool(head.get("has_signal", False)),
        },
        "applicability": {
            "coverage_p05": float(head["coverage_p05"]),
            "coverage_p95": float(head["coverage_p95"]),
            "coverage_median": float(head["coverage_median"]),
        },
        "class_balance": list(head.get("class_balance", [])),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("artifacts/heads.pkl"),
        help="Input heads bundle pickle.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/glucofm_heads.json"),
        help="Output JSON path.",
    )
    args = ap.parse_args()

    if not args.in_path.exists():
        raise FileNotFoundError(
            f"heads bundle not found: {args.in_path}; run `just fit-heads` first."
        )

    print(f"loading {args.in_path}")
    with args.in_path.open("rb") as fh:
        bundle = pickle.load(fh)

    encoder_meta = bundle.get("encoder", {})
    signal_floor = bundle.get("signal_floor", 0.55)
    n_repeats = bundle.get("n_repeats", 10)

    heads_in = bundle.get("heads", {})
    heads_out: dict[str, dict[str, Any]] = {}
    for key, head in heads_in.items():
        heads_out[key] = head_to_dict(head)

    payload = {
        "encoder": encoder_meta,
        "signal_floor": signal_floor,
        "n_repeats": n_repeats,
        "heads": heads_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    print(f"  {len(heads_out)} heads, "
          f"{sum(1 for h in heads_out.values() if h['reliability']['has_signal'])} above the "
          f"{signal_floor} signal floor")


if __name__ == "__main__":
    main()
