#!/usr/bin/env python3
"""Score the training corpus with every probe, and keep the score distribution.

The probe heads are unregularised logistic classifiers fitted in 128 dimensions on a few
hundred days. Their outputs saturate: most inputs come back near 0 or 1, including inputs
drawn from the cohort a head was fitted on, while held-out discrimination is only 0.64-0.88
ROC-AUC. So the raw score cannot be shown as a probability, and the demo currently shows
only its direction.

A rank is different. "This day scores higher than 82% of the corpus days this probe has seen"
is a statement the head can actually support, because ranking is exactly what ROC-AUC
measures. This script builds the reference the rank is computed against: a sample of corpus
windows, encoded, scored by every head, and reduced to percentile breakpoints.

Only breakpoints are published -- 101 numbers per head, not the embeddings. That keeps the
file small and means no participant window leaves the machine.

    uv run python scripts/build_reference_distribution.py --sample 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "canonical" / "windows"
ENCODER = REPO_ROOT / "artifacts" / "glucofm_encoder.onnx"
HEADS = REPO_ROOT / "artifacts" / "glucofm_heads.json"
OUT = REPO_ROOT / "artifacts" / "glucofm_reference.json"


def head_scores(head: dict, emb: np.ndarray) -> np.ndarray:
    """Positive-class (or winning-class) score for every row of `emb`."""
    mu = np.asarray(head["scale"]["mean"], dtype=np.float64)
    sd = np.asarray(head["scale"]["scale"], dtype=np.float64)
    w = np.asarray(head["classifier"]["coef"], dtype=np.float64)
    b = np.asarray(head["classifier"]["intercept"], dtype=np.float64)
    z = ((emb - mu) / sd) @ w.T + b
    if w.shape[0] == 1:
        return 1.0 / (1.0 + np.exp(-z[:, 0]))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=20_000, help="corpus windows to encode")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    values_path = CACHE / "strict_seed17.values.npy"
    if not values_path.exists():
        print(f"error: {values_path} missing; run `just build-windows`")
        return 2

    values = np.load(values_path, mmap_mode="r")
    mask = np.load(CACHE / "strict_seed17.mask.npy", mmap_mode="r")
    meta = json.loads((CACHE / "strict_seed17.meta.json").read_text())
    circ = np.asarray(meta["circadian_start"], dtype=np.int64)
    datasets = np.asarray(meta["dataset"])

    rng = np.random.default_rng(args.seed)
    n = min(args.sample, values.shape[0])
    idx = np.sort(rng.choice(values.shape[0], n, replace=False))
    print(f"encoding {n:,} of {values.shape[0]:,} corpus windows")

    sess = ort.InferenceSession(str(ENCODER), providers=["CPUExecutionProvider"])
    embeddings = np.empty((n, 128), dtype=np.float32)
    for start in range(0, n, args.batch):
        sl = idx[start : start + args.batch]
        embeddings[start : start + len(sl)] = sess.run(
            ["embedding"],
            {
                "values": np.asarray(values[sl], dtype=np.float32),
                "mask": np.asarray(mask[sl], dtype=np.float32),
                "circadian_start": circ[sl],
            },
        )[0]
        if start % (args.batch * 10) == 0:
            print(f"  {start:,}/{n:,}")

    bundle = json.loads(HEADS.read_text())
    percentiles = np.arange(0, 101)
    out: dict[str, object] = {
        "n_reference_windows": int(n),
        "sampled_from": int(values.shape[0]),
        "seed": args.seed,
        "cohort_mix": {
            str(k): int(v)
            for k, v in zip(*np.unique(datasets[idx], return_counts=True), strict=True)
        },
        "note": (
            "Percentile breakpoints of each probe's score over a random sample of the "
            "pretraining corpus. A day's rank against these is the statistic these heads "
            "support; their raw score is not calibrated and is not a probability."
        ),
        "heads": {},
    }
    for key, head in bundle["heads"].items():
        if not head["reliability"]["has_signal"]:
            continue
        scores = head_scores(head, embeddings.astype(np.float64))
        out["heads"][key] = {
            "breakpoints": [round(float(v), 6) for v in np.percentile(scores, percentiles)],
            "median": round(float(np.median(scores)), 6),
        }

    OUT.write_text(json.dumps(out))
    kb = OUT.stat().st_size / 1024
    print(f"\nwrote {OUT} ({kb:.0f} KB) — {len(out['heads'])} heads over {n:,} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
