"""Grid assignment and the floor-vs-nearest binning question.

Blueprint §9.2 fixes the arithmetic. §9.3 leaves open which datasets use floor and which use
nearest-index rounding: the paper states both are used but never publishes the mapping. This
module implements both and measures which one the source timestamps actually support, so
DECISIONS.md D004 is settled by evidence rather than preference.

The tie rule is defined explicitly here because Python's ``round`` is banker's rounding, which
would silently send exact .5 residuals to the even index. Blueprint §9.3 warns about this.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

GRID_MINUTES = 5
SEQUENCE_LENGTH = 288  # 24 h / 5 min
PATCHES = 24
STEPS_PER_PATCH = 12


class BinningRule(StrEnum):
    FLOOR = "floor"
    NEAREST = "nearest"


def grid_offsets(t: datetime, t0: datetime) -> float:
    """Position of ``t`` in 5-minute units relative to ``t0``. May be fractional."""
    return (t - t0).total_seconds() / (GRID_MINUTES * 60.0)


def assign_index(u: float, rule: BinningRule) -> int:
    """Map a fractional grid offset to an integer grid index.

    ``NEAREST`` rounds half **away from zero**, not to even. For the non-negative offsets this
    project produces, that means exactly .5 rounds up. Stated explicitly so the behaviour is a
    decision rather than an artefact of the language.
    """
    if rule is BinningRule.FLOOR:
        return math.floor(u)
    return math.floor(u + 0.5)


def circadian_start_index(t1: datetime) -> int:
    """Absolute 5-minute index of a window's first timestamp. Blueprint §9.2 — PAPER_EXACT.

    ``s = floor((60*hour + minute) / 5)``. Note this uses local wall-clock time; absolute
    time-of-day is a model input, so the caller must supply local time, not UTC.
    """
    return (60 * t1.hour + t1.minute) // GRID_MINUTES


def absolute_index(start_index: int, j: int) -> int:
    """``a_j = (s + j) mod 288``. Blueprint §9.2 — PAPER_EXACT."""
    return (start_index + j) % SEQUENCE_LENGTH


def segment_readings(
    timestamps: list[datetime], max_internal_gap_minutes: int = 60
) -> list[tuple[int, int]]:
    """Split into continuous segments at gaps **strictly greater than** one hour.

    Blueprint §9.1 — PAPER_EXACT. The boundary matters: a gap of exactly 60 minutes stays
    internal, because the paper says gaps *longer than* one hour create boundaries.

    Returns half-open ``[start, end)`` index ranges into ``timestamps``.
    """
    if not timestamps:
        return []
    bound = timedelta(minutes=max_internal_gap_minutes)
    segments: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > bound:
            segments.append((start, i))
            start = i
    segments.append((start, len(timestamps)))
    return segments


@dataclass
class BinningAudit:
    """Evidence for the floor-vs-nearest decision on one dataset. Blueprint §9.3."""

    dataset_id: str
    n_readings: int = 0
    n_segments: int = 0
    #: Seconds past the minute in the raw source timestamps. A constant non-zero value means
    #: the device stamps at a fixed offset, which is what makes the rule choice bite.
    second_offsets: dict[int, int] = field(default_factory=dict)
    #: Histogram of the fractional part of the grid offset, in 10 bins over [0,1).
    residual_hist: list[int] = field(default_factory=lambda: [0] * 10)
    #: Readings whose assigned index differs between the two rules.
    n_rule_disagreements: int = 0
    #: Two real readings landing on one grid position (blueprint §9.2: average them).
    collisions_floor: int = 0
    collisions_nearest: int = 0
    recovered_cadence_minutes: float | None = None

    @property
    def disagreement_fraction(self) -> float:
        return self.n_rule_disagreements / self.n_readings if self.n_readings else 0.0

    def summary(self) -> str:
        offs = sorted(self.second_offsets.items(), key=lambda kv: -kv[1])[:3]
        off_s = ", ".join(f":{s:02d}s x{n}" for s, n in offs)
        return (
            f"{self.dataset_id:<14} n={self.n_readings:>7}  seg={self.n_segments:>5}  "
            f"cadence={self.recovered_cadence_minutes or 0:.1f}m  "
            f"disagree={self.disagreement_fraction:6.2%}  "
            f"collide floor={self.collisions_floor} near={self.collisions_nearest}  [{off_s}]"
        )


def audit_binning(dataset_id: str, timestamps: list[datetime]) -> BinningAudit:
    """Measure how the two rules behave on real source timestamps.

    Runs per continuous segment, because grid offsets are defined relative to the first
    reading of a window and windows never cross a segment boundary (§9.1).
    """
    audit = BinningAudit(dataset_id=dataset_id, n_readings=len(timestamps))
    if not timestamps:
        return audit

    for t in timestamps:
        audit.second_offsets[t.second] = audit.second_offsets.get(t.second, 0) + 1

    deltas = [
        (b - a).total_seconds() / 60.0
        for a, b in itertools.pairwise(timestamps)
        if 0 < (b - a).total_seconds() <= 3600
    ]
    if deltas:
        audit.recovered_cadence_minutes = sorted(deltas)[len(deltas) // 2]

    segments = segment_readings(timestamps)
    audit.n_segments = len(segments)
    for start, end in segments:
        t0 = timestamps[start]
        seen_floor: set[int] = set()
        seen_near: set[int] = set()
        for t in timestamps[start:end]:
            u = grid_offsets(t, t0)
            frac = u - math.floor(u)
            audit.residual_hist[min(int(frac * 10), 9)] += 1
            i_f = assign_index(u, BinningRule.FLOOR)
            i_n = assign_index(u, BinningRule.NEAREST)
            if i_f != i_n:
                audit.n_rule_disagreements += 1
            if i_f in seen_floor:
                audit.collisions_floor += 1
            seen_floor.add(i_f)
            if i_n in seen_near:
                audit.collisions_nearest += 1
            seen_near.add(i_n)
    return audit
