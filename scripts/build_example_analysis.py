#!/usr/bin/env python3
"""Produce the worked example the website shows on /example.

The site could describe what the model returns but never showed it end to end on a real
recording, so a reader had no way to judge whether the output was worth anything. This runs
the full pipeline over a real multi-day CGM export and writes everything the page needs.

The output is a single JSON file. Swapping the example recording, or removing it, is a
one-file change with no code edit -- which matters, because the file contains a real person's
glucose readings and that decision should stay easy to reverse.

    uv run python scripts/build_example_analysis.py --csv <export.csv> --days 5

The CSV is read with the same column detection the browser uses. Nothing about the person is
carried through: no name, no device serial, no sensor id, no absolute dates. Days are labelled
"Day 1", "Day 2" and timestamps are reduced to a time of day.

Days are cut at LOCAL midnight, not UTC midnight, and `--timezone` says which local. This is
not only a labelling question: the encoder takes a `circadian_start` index saying where the
window begins in the 24-hour cycle, and a window cut at UTC midnight for someone seven hours
west starts at 17:00 their time. Passing 0 for that window tells the model the day starts at
midnight when it does not, and the time-of-day embedding is load-bearing (removing it costs
0.015 ROC). Cutting at local midnight makes `circadian_start = 0` true.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
ENCODER = REPO_ROOT / "artifacts" / "glucofm_encoder.onnx"
STREAMS = REPO_ROOT / "artifacts" / "glucofm_encoder_streams.onnx"
HEADS = REPO_ROOT / "artifacts" / "glucofm_heads.json"
REFERENCE = REPO_ROOT / "artifacts" / "glucofm_reference.json"
OUT = REPO_ROOT / "web" / "public" / "data" / "example-analysis.json"

N = 288
STEP = timedelta(minutes=5)
ISO_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def read_readings(path: Path) -> list[tuple[datetime, float]]:
    """Pick the timestamp and glucose columns by inspecting values, as the browser does."""
    rows = list(csvmod.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    header = list(rows[0])

    def looks_like_datetime(col: str) -> bool:
        vals = [r[col] for r in rows[:25] if r.get(col)]
        ok = 0
        for v in vals:
            v = v.strip()
            if re.fullmatch(r"-?\d+(\.\d+)?", v):
                continue  # a counter or an epoch, not a wall clock
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
                ok += 1
            except ValueError:
                pass
        return bool(vals) and ok >= max(1, int(len(vals) * 0.8))

    tcol = next((c for c in header if looks_like_datetime(c)), None)
    gcol = next((c for c in header if "glucose" in c.lower() and "mg" in c.lower()), None)
    gcol = gcol or next((c for c in header if "glucose" in c.lower()), None)
    if not tcol or not gcol:
        raise SystemExit(f"could not find timestamp/glucose columns in {header}")

    state_col = next((c for c in header if c.lower() == "algorithm_state"), None)
    out = []
    for r in rows:
        if state_col and "warmup" in (r.get(state_col) or ""):
            continue
        try:
            ts = datetime.fromisoformat(r[tcol].replace("Z", "+00:00"))
            g = float(r[gcol])
        except (ValueError, TypeError):
            continue
        if 20 <= g <= 600:
            out.append((ts, g))
    out.sort(key=lambda x: x[0])
    return out


def gridify(readings, start):
    buckets = defaultdict(list)
    for ts, g in readings:
        idx = int((ts - start) / STEP)
        if 0 <= idx < N:
            buckets[idx].append(g)
    values = np.zeros(N, dtype=np.float32)
    mask = np.zeros(N, dtype=np.float32)
    for i, vs in buckets.items():
        values[i] = float(np.median(vs))
        mask[i] = 1.0
    return values, mask


def clinical(values, mask):
    obs = values[mask > 0]
    if not len(obs):
        return None
    mean, sd = float(obs.mean()), float(obs.std())
    frac = lambda lo, hi: float(((obs >= lo) & (obs < hi)).mean())  # noqa: E731
    return {
        "n_observed": len(obs),
        "mean_mg_dl": round(mean, 1),
        "sd_mg_dl": round(sd, 1),
        "coefficient_of_variation": round(sd / mean, 4),
        "gmi_percent": round(3.31 + 0.02392 * mean, 2),
        "time_below_70": round(frac(0, 70), 4),
        "time_in_range_70_180": round(frac(70, 180), 4),
        "time_above_180": round(frac(180, 1000), 4),
        "min_mg_dl": round(float(obs.min()), 1),
        "max_mg_dl": round(float(obs.max()), 1),
    }


def head_score(head, emb):
    mu = np.asarray(head["scale"]["mean"])
    sd = np.asarray(head["scale"]["scale"])
    w = np.asarray(head["classifier"]["coef"])
    b = np.asarray(head["classifier"]["intercept"])
    z = ((emb - mu) / sd) @ w.T + b
    if w.shape[0] == 1:
        return float(1 / (1 + np.exp(-z[0])))
    e = np.exp(z - z.max())
    return float((e / e.sum()).max())


def percentile(breakpoints, score):
    b = np.asarray(breakpoints)
    if score <= b[0]:
        return 0.0
    if score >= b[-1]:
        return 100.0
    i = int(np.searchsorted(b, score) - 1)
    span = b[i + 1] - b[i]
    return float(i + ((score - b[i]) / span if span > 0 else 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument(
        "--timezone",
        default="America/Los_Angeles",
        help="IANA zone the recording was made in; days are cut at midnight in this zone",
    )
    args = ap.parse_args()

    tz = ZoneInfo(args.timezone)
    readings = [(ts.astimezone(tz), g) for ts, g in read_readings(args.csv)]
    print(f"{len(readings):,} readings in {args.timezone}")
    print(f"  {readings[0][0]:%Y-%m-%d %H:%M} -> {readings[-1][0]:%Y-%m-%d %H:%M}")

    starts = sorted({r[0].replace(hour=0, minute=0, second=0, microsecond=0) for r in readings})
    grids = []
    for s in starts:
        v, m = gridify(readings, s)
        cov = float(m.mean())
        if cov >= args.min_coverage:
            grids.append((s, v, m, cov))
    grids = sorted(grids, key=lambda g: -g[3])[: args.days]
    grids.sort(key=lambda g: g[0])
    if not grids:
        raise SystemExit("no day cleared the coverage threshold")

    enc = ort.InferenceSession(str(ENCODER), providers=["CPUExecutionProvider"])
    strm = ort.InferenceSession(str(STREAMS), providers=["CPUExecutionProvider"])
    heads = json.loads(HEADS.read_text())["heads"]
    ref = json.loads(REFERENCE.read_text())

    days = []
    embeddings = []
    for n, (start, v, m, cov) in enumerate(grids, start=1):
        # Windows are cut at local midnight, so the circadian index is genuinely 0.
        circ = np.array([0], dtype=np.int64)
        feed = {"values": v[None], "mask": m[None], "circadian_start": circ}
        emb = enc.run(["embedding"], feed)[0][0]
        out = strm.run(["state_signal", "event_signal"], feed)
        embeddings.append(emb)

        probes = []
        for key, h in heads.items():
            if not h["reliability"]["has_signal"]:
                continue
            lo, hi = h["applicability"]["coverage_p05"], h["applicability"]["coverage_p95"]
            if not (lo - 0.15 <= cov <= hi + 0.15):
                continue
            s = head_score(h, emb)
            rh = ref["heads"].get(key)
            probes.append({
                "key": key,
                "task": h["task"],
                "cohort": h["dataset"],
                "score": round(s, 6),
                "percentile": round(percentile(rh["breakpoints"], s), 1) if rh else None,
                "roc_auc": round(h["reliability"]["roc_auc"], 3),
                "n_subjects": h["reliability"]["n_subjects"],
                # Every 5th percentile of this probe's score over the corpus sample. Enough
                # to draw where the 20,000 reference days actually pile up, which is what
                # makes "the raw score is not a probability" visible rather than asserted.
                "corpus_deciles": (
                    [round(float(v), 6) for v in rh["breakpoints"][::5]] if rh else None
                ),
            })
        probes.sort(key=lambda p: -p["roc_auc"])

        days.append({
            "label": f"Day {n}",
            "weekday": start.strftime("%A"),
            "coverage": round(cov, 4),
            "values_mg_dl": [round(float(x), 1) if m[i] else None for i, x in enumerate(v)],
            "mask": [int(x) for x in m],
            "state": [round(float(x), 4) for x in out[0][0]],
            "event": [round(float(x), 4) for x in out[1][0]],
            "embedding": [round(float(x), 4) for x in emb],
            "clinical": clinical(v, m),
            "probes": probes,
        })
        print(f"  Day {n}: coverage {cov:.1%}, {len(probes)} probes applicable")

    stacked = np.stack(embeddings)
    norm = stacked / (np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-9)
    similarity = np.round(norm @ norm.T, 4).tolist()

    below_floor = sorted(
        (
            {
                "key": k,
                "task": h["task"],
                "cohort": h["dataset"],
                "roc_auc": round(h["reliability"]["roc_auc"], 3),
            }
            for k, h in heads.items()
            if not h["reliability"]["has_signal"]
        ),
        key=lambda x: x["roc_auc"],
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": (
            "One real multi-day CGM export, run through the released encoder. De-identified: "
            "no name, device, sensor id or calendar date is carried through."
        ),
        "source": "a consenting adult volunteer, Dexcom G6",
        "times_are_local": True,
        "n_days": len(days),
        # The funnel from shipped classifiers to the rows a reader sees. Without these the
        # page shows five rows and a reader reasonably concludes there are five classifiers.
        "heads_shipped": len(heads),
        "heads_with_signal": sum(1 for h in heads.values() if h["reliability"]["has_signal"]),
        "heads_below_floor": below_floor,
        "signal_floor": json.loads(HEADS.read_text())["signal_floor"],
        "days": days,
        "similarity": similarity,
        "encoder": json.loads(
            (REPO_ROOT / "artifacts" / "glucofm_encoder.onnx.meta.json").read_text()
        ),
        "reference_windows": ref["n_reference_windows"],
    }))
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
