"""Per-dataset binning and reconciliation audit. Blueprint §9.3.

Regenerates the evidence behind DECISIONS D004. Run:

    uv run python -m opencgm_stateevent.data.audit
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ..provenance import RunRecord
from .readers import Reading, read_big_ideas, read_colas, read_hall, read_shanghai, read_stanford
from .timestamps import BinningRule, assign_index, grid_offsets, segment_readings

RAW = Path(os.environ.get("OPENCGM_RAW_ROOT", "data/raw"))

#: Where the Shanghai archive was unpacked. It is read from here rather than from `data/raw`
#: because the zip's filenames are GBK-encoded and have to be extracted with `unzip -O GBK`
#: before anything can read them. Override with OPENCGM_EXTRACTED_ROOT; the default is a
#: sibling of the raw tree so a fresh checkout works without setting anything.
EXTRACTED = Path(os.environ.get("OPENCGM_EXTRACTED_ROOT", str(RAW.parent / "extracted")))

STANFORD_DIR = "stanford/github_94d647944c001dfa34492fd593b6e4804c1f45c4"

#: Paper targets. BIG IDEAs/Stanford/Colas from §6.1; Shanghai is pretrain 12,414 +
#: downstream 15,634 because we read the whole T2DM cohort at once.
PAPER: dict[str, tuple[int, int]] = {
    "big_ideas": (16, 3017),
    "stanford": (56, 36332),
    "shanghai_t2dm": (109, 28048),
    "colas": (206, 9544),
    "hall": (56, 7090),
}


def sources() -> dict[str, Any]:
    return {
        "big_ideas": lambda: read_big_ideas(RAW / "big_ideas/1.1.2"),
        "stanford": lambda: read_stanford(RAW / STANFORD_DIR),
        "shanghai_t2dm": lambda: read_shanghai(EXTRACTED / "shanghai", "T2DM"),
        "colas": lambda: read_colas(RAW / "colas/plos_pone_0225817"),
        "hall": lambda: read_hall(RAW / "hall/plos_pbio_2005143"),
    }


def _by_session(readings: Iterator[Reading]) -> tuple[dict[str, list[datetime]], set[str]]:
    per: dict[str, list[datetime]] = defaultdict(list)
    subjects: set[str] = set()
    for r in readings:
        per[r.session_id].append(r.local_datetime)
        subjects.add(r.source_subject_id)
    for ts in per.values():
        ts.sort()
    return per, subjects


def audit_dataset(name: str, readings: Iterator[Reading]) -> dict[str, Any]:
    per, subjects = _by_session(readings)
    n = sum(len(v) for v in per.values())
    hours = 0.0
    segments = 0
    stats = {
        r: {"kept": 0, "collisions": 0, "abs_err": 0.0, "max_err": 0.0} for r in BinningRule
    }

    for ts in per.values():
        for s, e in segment_readings(ts):
            segments += 1
            hours += (ts[e - 1] - ts[s]).total_seconds() / 3600.0
            t0 = ts[s]
            for rule in BinningRule:
                seen: set[int] = set()
                st = stats[rule]
                for t in ts[s:e]:
                    u = grid_offsets(t, t0)
                    idx = assign_index(u, rule)
                    err = abs(u - idx) * 5.0
                    st["abs_err"] += err
                    st["max_err"] = max(st["max_err"], err)
                    if idx in seen:
                        st["collisions"] += 1
                    else:
                        seen.add(idx)
                        st["kept"] += 1

    out: dict[str, Any] = {
        "dataset": name,
        "subjects": len(subjects),
        "sessions": len(per),
        "readings": n,
        "segments": segments,
        "hours": round(hours),
        "rules": {},
    }
    for rule, st in stats.items():
        out["rules"][rule.value] = {
            "kept": st["kept"],
            "collisions": st["collisions"],
            "lost_pct": round(100 * st["collisions"] / n, 2) if n else 0.0,
            "mean_abs_err_min": round(st["abs_err"] / n, 3) if n else 0.0,
            "max_abs_err_min": round(st["max_err"], 2),
        }
    if name in PAPER:
        exp_rec, exp_h = PAPER[name]
        out["paper"] = {
            "records": exp_rec,
            "hours": exp_h,
            "hours_delta_pct": round(100 * (hours - exp_h) / exp_h, 1),
        }
    return out


def main() -> None:
    rec = RunRecord(command="data audit")
    results = []
    header = (
        f"{'dataset':<15}{'subj':>5}{'sess':>6}{'readings':>10}{'hours':>8}"
        f"{'Δpaper':>8}   {'rule':<8}{'collide':>8}{'lost':>7}{'err(min)':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, fn in sources().items():
        r = audit_dataset(name, fn())
        results.append(r)
        delta = f"{r['paper']['hours_delta_pct']:+.1f}%" if "paper" in r else "-"
        for i, (rule, st) in enumerate(r["rules"].items()):
            lead = (
                f"{name:<15}{r['subjects']:>5}{r['sessions']:>6}"
                f"{r['readings']:>10}{r['hours']:>8}{delta:>8}"
                if i == 0
                else " " * 52
            )
            print(
                f"{lead}   {rule:<8}{st['collisions']:>8}"
                f"{st['lost_pct']:>6.2f}%{st['mean_abs_err_min']:>9.2f}"
            )
    rec.finish(datasets=len(results), results=results).write()
    print("\nDecision: nearest, all datasets. See DECISIONS.md D004 and reports/binning_audit.md")


if __name__ == "__main__":
    main()
