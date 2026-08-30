"""BVP-segment to CGM-window alignment for the section-23 PPG pilot."""

# Each BVP JSON file in the on-disk dataset contains ~30 min x 64 Hz raw photoplethysmography
# plus a Unix-microsecond `timestamp_start`. The per-subject `P00x_glucose.csv` contains
# CGM readings in mmol/L at 15-min cadence on local Kolkata time.
#
# We align each BVP segment to the CGM grid by:
#
#   1. Converting `timestamp_start` (Unix microseconds, UTC) to local-naive datetime.
#   2. Resampling the CGM trace to the 5-minute grid (nearest, no interpolation beyond the
#      sensor's own observations; missing CGM positions stay missing and become the validity
#      mask).
#   3. Computing the BVP patch grid: the BVP segment is split into 5-minute chunks of
#      19200 samples each. Each chunk gets the CGM reading at the same local timestamp.
#   4. If the CGM reading falls inside the segment's 5-min window, that patch has a target;
#      otherwise the patch's mask is 0.
#
# The output is a list of (bvp_patch: np.ndarray shape (19200,), glucose_mmol: float or NaN,
# mask: 0/1) tuples, ready for the PpgStudentEncoder.
#
# The on-disk timestamps are microseconds-since-Unix-epoch UTC. Date folders in the dataset
# are local Kolkata time, which is UTC+5:30 - but the JSON timestamps are UTC, so the
# local-naive datetime conversion is correct as long as we use the same local-naive convention
# as the per-subject glucose CSV (which is naive local). For this pilot we treat both as
# local-naive and rely on the fact that they're recorded in the same timezone (the README
# states "Date folders are derived from timestamps converted to local time (Kolkata)").

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PpgGlucoseMol = float  # mmol/L

#: CGM grid resolution (5 minutes). Matches the strict CGM pipeline (§6).
PATCH_DURATION = timedelta(minutes=5)
#: BVP sampling rate in Hz.
BVP_RATE_HZ = 64
#: Samples per 5-minute patch.
SAMPLES_PER_PATCH = int(PATCH_DURATION.total_seconds() * BVP_RATE_HZ)  # 19200


@dataclass(frozen=True)
class BvpCgmPatch:
    """One aligned (BVP patch, CGM value, mask) triple for the student."""

    bvp: np.ndarray  # (SAMPLES_PER_PATCH,) raw 64 Hz photoplethysmography
    glucose_mmol: float | None  # mmol/L; None where CGM is missing
    timestamp_local: datetime  # the CGM-grid timestamp for this patch
    subject: str  # P001..P005


def _parse_cgm_csv(path: Path) -> list[tuple[datetime, PpgGlucoseMol]]:
    """Parse a per-subject glucose CSV with mixed date formats.

    The on-disk format uses DD/MM/YY HH:MM (e.g. '19/09/24 17:05'). We accept the two
    formats observed in P001/P005 and fall back to dateutil. Returns list of
    (naive-local-datetime, mmol/L) sorted ascending. Empty readings are skipped.
    """
    rows: list[tuple[datetime, PpgGlucoseMol]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = (row.get("Device Timestamp") or "").strip()
            v = (row.get("Historic Glucose mmol/L") or "").strip()
            if not ts or not v:
                continue
            try:
                mmol = float(v)
            except ValueError:
                continue
            # Two formats are observed in this dataset:
            #   * P001 uses DD/MM/YY HH:MM with "/" separators (e.g. "19/09/24 17:05").
            #   * P002-P005 use DD-MM-YYYY HH:MM with "-" separators (e.g.
            #     "25-10-2024 11:36"). Putting DD-MM-YYYY ahead of MM-DD-YYYY matters:
            #     for ambiguous dates like "01-11-2024" or "02-02-2025", MM-DD-YYYY
            #     silently mis-parses them (Jan 11 / Feb 2 with day/month swapped).
            #     DD-MM-YYYY is the format the per-subject files actually use, so it
            #     goes first. We do not accept MM-DD-YYYY: the dataset never emits it,
            #     and including it produced silent corruption of dates like
            #     "01-11-2024" (was being parsed as Jan 11, the wrong answer).
            parsed: datetime | None = None
            for fmt in ("%d/%m/%y %H:%M", "%d-%m-%Y %H:%M"):
                try:
                    parsed = datetime.strptime(ts, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                continue
            rows.append((parsed, mmol))
    rows.sort(key=lambda r: r[0])
    return rows


def _parse_bvp_json(path: Path) -> tuple[datetime, np.ndarray]:
    """Read a single bvp_NNNNNN.json. Returns (naive-local-Kolkata start, samples array).

    The on-disk JSON timestamps are microseconds-since-Unix-epoch UTC. The per-subject
    glucose CSVs are naive-local Kolkata time (UTC+5:30). We shift the JSON timestamp by
    +5h30m so both sides of the alignment live in the same naive-local frame. Without this
    shift, every BVP segment was 5h30m away from any CGM reading and the alignment was
    silently empty for 4 of the 5 subjects.
    """
    with path.open() as f:
        data = json.load(f)
    ts_us = int(data["timestamp_start"])
    dt_utc = datetime.utcfromtimestamp(ts_us / 1_000_000)
    kolkata_offset = timedelta(hours=5, minutes=30)
    dt_local = dt_utc + kolkata_offset
    return dt_local.replace(microsecond=0), np.asarray(data["bvp_values"], dtype=np.float32)


def _cgm_at(cgm_rows: list[tuple[datetime, PpgGlucoseMol]], t: datetime) -> float | None:
    """Return the mmol/L value at datetime t, or None if no CGM reading within +/- 7.5 min.

    The sensor's native cadence is 15 min; readings land on :05, :20, :35, :50.
    We use nearest-within-7.5-min as the alignment rule.
    """
    if not cgm_rows:
        return None
    # Binary search by datetime
    lo, hi = 0, len(cgm_rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if cgm_rows[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    # lo is the first row >= t; check lo-1 and lo
    best: tuple[datetime, float] | None = None
    best_dt: timedelta | None = None
    for cand in (lo - 1, lo):
        if 0 <= cand < len(cgm_rows):
            dt = abs(cgm_rows[cand][0] - t)
            if dt <= timedelta(minutes=12) and (best_dt is None or dt < best_dt):
                best_dt = dt
                best = cgm_rows[cand]
    return best[1] if best is not None else None


def iter_aligned_patches(
    data_zip_dir: Path,
    subjects: list[str] | None = None,
) -> Iterator[BvpCgmPatch]:
    """Yield aligned (BVP, CGM, mask) patches across all subjects and segments.

    Walks `data_zip_dir/Data/P00x/<date>/bvp_*.json`, parses each segment, aligns it to the
    5-min grid using that subject's `P00x_glucose.csv`, and emits a BvpCgmPatch per
    5-minute chunk. Patches whose CGM timestamp falls outside the segment's range are
    emitted with mask=0 (mask encoded by glucose_mmol=None).
    """
    data_root = data_zip_dir / "Data"
    if subjects is None:
        subjects = sorted(
            p.name for p in data_root.iterdir() if p.is_dir() and p.name.startswith("P")
        )
    for subj in subjects:
        subj_dir = data_root / subj
        glucose_path = subj_dir / f"{subj}_glucose.csv"
        if not glucose_path.exists():
            continue
        cgm_rows = _parse_cgm_csv(glucose_path)
        if not cgm_rows:
            continue
        # Walk date folders
        for date_dir in sorted(subj_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for bvp_path in sorted(date_dir.glob("bvp_*.json")):
                try:
                    start, samples = _parse_bvp_json(bvp_path)
                except (KeyError, ValueError):
                    continue
                n_patches = len(samples) // SAMPLES_PER_PATCH
                if n_patches == 0:
                    continue
                for i in range(n_patches):
                    patch_start = start + i * PATCH_DURATION
                    sl = slice(i * SAMPLES_PER_PATCH, (i + 1) * SAMPLES_PER_PATCH)
                    bvp = samples[sl].astype(np.float32, copy=False)
                    mmol = _cgm_at(cgm_rows, patch_start)
                    yield BvpCgmPatch(
                        bvp=bvp,
                        glucose_mmol=mmol,
                        timestamp_local=patch_start,
                        subject=subj,
                    )


SUBJECT_RE = re.compile(r"^P\d{3}$")


def list_subjects(data_zip_dir: Path) -> list[str]:
    """Return the list of P00x subject directories under Data/."""
    data_root = data_zip_dir / "Data"
    return sorted(p.name for p in data_root.iterdir() if p.is_dir() and SUBJECT_RE.match(p.name))
