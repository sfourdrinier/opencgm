"""Analyse a CGM export. The command-line face of `opencgm_stateevent.infer`.

    uv run python scripts/analyse.py --csv my_dexcom_export.csv
    uv run python scripts/analyse.py --csv export.csv --days 14 --json report.json

Column names are sniffed rather than required, because every vendor exports something different
and a tool that only accepts one schema will be wrong for the next sensor. Dexcom Clarity, Libre
View and a plain `timestamp,glucose` file all work. If sniffing fails it says which columns it
found instead of guessing, since guessing a glucose column wrong produces a complete, plausible,
entirely fictional report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from opencgm_stateevent.infer import Analyser, similarity

#: Substrings, lower-cased, in vendor order of specificity. Dexcom Clarity writes
#: "Glucose Value (mg/dL)"; Libre View writes "Historic Glucose mg/dL"; exports from our own
#: readers write "glucose_mg_dl".
TIME_HINTS = ("timestamp (yyyy-mm-ddthh:mm:ss)", "device timestamp", "timestamp", "time", "date")
GLUCOSE_HINTS = (
    "glucose value (mg/dl)", "historic glucose mg/dl", "glucose_mg_dl",
    "glucose", "sgv", "value", "mg/dl",
)


def sniff(columns: list[str], hints: tuple[str, ...], what: str) -> str:
    lowered = {c.lower().strip(): c for c in columns}
    for hint in hints:
        for low, original in lowered.items():
            if hint in low:
                return original
    raise SystemExit(
        f"could not find a {what} column. Columns present: {list(columns)}\n"
        f"Pass --{what}-column explicitly rather than letting this guess."
    )


def read_csv(path: Path, time_column: str | None, glucose_column: str | None):
    frame = pd.read_csv(path)
    time_column = time_column or sniff(list(frame.columns), TIME_HINTS, "time")
    glucose_column = glucose_column or sniff(list(frame.columns), GLUCOSE_HINTS, "glucose")
    print(f"reading time from {time_column!r}, glucose from {glucose_column!r}", file=sys.stderr)

    times = pd.to_datetime(frame[time_column], errors="coerce", format="mixed")
    values = pd.to_numeric(frame[glucose_column], errors="coerce")
    usable = times.notna() & values.notna()
    dropped = int((~usable).sum())
    if dropped:
        # Dexcom exports interleave calibration and event rows with no glucose value. Dropping
        # them is correct; doing it silently is not, because a schema change looks identical.
        print(f"skipped {dropped} rows with no usable timestamp or glucose", file=sys.stderr)

    readings = [
        (t.to_pydatetime(), float(v))
        for t, v in zip(times[usable], values[usable], strict=True)
    ]
    if not readings:
        raise SystemExit("no usable readings found")
    return readings


def render(report, index: int | None = None) -> str:
    m = report.metrics
    head = f"=== {report.start:%Y-%m-%d %H:%M} ==="
    if index is not None:
        head = f"=== day {index}: {report.start:%a %d %b} ==="
    lines = [head]
    for warning in report.warnings:
        lines.append(f"  ! {warning}")
    lines += [
        f"  coverage            {m.coverage:6.1%}  ({m.n_observed}/288 five-minute positions)",
        f"  average glucose     {m.mean_glucose:6.1f} mg/dL   "
        f"(GMI {m.glucose_management_indicator:.1f}%)",
        f"  time in range       {m.time_in_range:6.1%}  70-180 mg/dL",
        f"  time below 70       {m.time_below_70:6.1%}  (below 54: {m.time_below_54:.1%})",
        f"  time above 180      {m.time_above_180:6.1%}  (above 250: {m.time_above_250:.1%})",
        f"  variability (CV)    {m.coefficient_of_variation:6.1%}  "
        f"{'stable' if m.variability_is_stable else 'above the 36% stability threshold'}",
        f"  range               {m.min_glucose:.0f} - {m.max_glucose:.0f} mg/dL",
        f"  overnight mean      {m.overnight_mean:6.1f} mg/dL   (dawn rise {m.dawn_rise:+.1f})",
        f"  longest steady run  {m.longest_stable_hours:6.1f} h",
    ]
    if report.phenotypes:
        usable = [p for p in report.phenotypes if p.has_signal and p.applicable]
        lines.append("")
        lines.append(f"  research signals ({len(usable)} of {len(report.phenotypes)} above the "
                     f"signal floor; all are population associations, not diagnoses):")
        for p in report.phenotypes:
            if p.has_signal and p.applicable:
                lines.append(f"    {p.task:<44} {p.probability:5.0%}   "
                             f"reliability {p.reliability:.2f} +/- {p.reliability_sd:.2f} "
                             f"(n={p.n_subjects_learned_from})")
        wrong_sensor = [p for p in report.phenotypes if not p.applicable]
        if wrong_sensor:
            lines.append(f"    ({len(wrong_sensor)} measures were learned from a different "
                         f"sampling rate and do not apply to this recording: "
                         f"{', '.join(p.task for p in wrong_sensor)})")
        hidden = [p for p in report.phenotypes if p.applicable and not p.has_signal]
        if hidden:
            lines.append(f"    ({len(hidden)} measures showed no reliable signal in our cohort "
                         f"and are not scored: {', '.join(p.task for p in hidden)})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/rawstats120/ckpt_last.pt"))
    ap.add_argument("--heads", type=Path, default=Path("artifacts/heads.pkl"))
    ap.add_argument("--days", type=int, default=1, help="how many recent 24h windows to report")
    ap.add_argument("--time-column", default=None)
    ap.add_argument("--glucose-column", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    readings = read_csv(args.csv, args.time_column, args.glucose_column)
    span = max(t for t, _ in readings) - min(t for t, _ in readings)
    print(f"{len(readings):,} readings spanning {span.days} days "
          f"{span.seconds // 3600} hours\n", file=sys.stderr)

    analyser = Analyser.load(args.checkpoint, heads=args.heads, device=args.device)
    reports = (
        analyser.analyse_stream(readings, days=args.days) if args.days > 1
        else [analyser.analyse_day(readings)]
    )
    if not reports:
        raise SystemExit("no window had enough readings to analyse")

    for i, report in enumerate(reports, start=1):
        print(render(report, i if len(reports) > 1 else None))
        print()

    if len(reports) > 1:
        s = similarity(reports)
        latest = s[-1, :-1]
        most, least = int(np.argmax(latest)), int(np.argmin(latest))
        print("day-to-day similarity (cosine, in embedding space)")
        print(f"  today is most like day {most + 1} ({latest[most]:.3f}) "
              f"and least like day {least + 1} ({latest[least]:.3f})")
        print(f"  mean similarity of today to the previous {len(latest)} days: "
              f"{latest.mean():.3f}")

    if args.json:
        args.json.write_text(
            "[\n" + ",\n".join(r.to_json() for r in reports) + "\n]"
            if len(reports) > 1 else reports[0].to_json()
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
