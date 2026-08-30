"""Mapping a continuous segment onto the 288-position 5-minute grid. Blueprint §9.2.

The single most important property here: **the mask is authoritative, the values are not**.
Absent grid positions carry a fill value only because tensors need a number there. Nothing
downstream may read a value whose mask is zero, and `tests/golden/test_grid.py` proves that
changing the fill value cannot alter any valid output (§10.2).

There is no interpolation. There is no imputation. A 15-minute source stays sparse on a
5-minute grid, with two-thirds of its positions masked off, and that sparsity is the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .timestamps import (
    GRID_MINUTES,
    PATCHES,
    SEQUENCE_LENGTH,
    STEPS_PER_PATCH,
    BinningRule,
    assign_index,
    circadian_start_index,
    grid_offsets,
)

WINDOW = timedelta(hours=24)


@dataclass
class Window:
    """One 24-hour aligned window. Blueprint §8.4."""

    window_id: str
    dataset_id: str
    canonical_subject_id: str
    biological_person_id: str
    session_id: str
    segment_index: int
    start_local: datetime
    circadian_start_index: int
    values: np.ndarray  # float32[288]; meaningless where mask is False
    mask: np.ndarray  # bool[288]; the physical observation mask, authoritative
    #: True when the source supplied a real calendar date (false for Colas, see readers).
    date_is_real: bool = True

    @property
    def n_observed(self) -> int:
        return int(self.mask.sum())

    @property
    def coverage(self) -> float:
        return self.n_observed / SEQUENCE_LENGTH

    def patch_density(self) -> np.ndarray:
        """Observed fraction per one-hour patch. Blueprint §11.2 — PAPER_EXACT."""
        return self.mask.reshape(PATCHES, STEPS_PER_PATCH).mean(axis=1).astype(np.float32)

    def absolute_indices(self) -> np.ndarray:
        """``a_j = (s + j) mod 288`` for every position. Blueprint §9.2."""
        return (self.circadian_start_index + np.arange(SEQUENCE_LENGTH)) % SEQUENCE_LENGTH


def build_window(
    timestamps: list[datetime],
    values: list[float],
    start: datetime,
    *,
    rule: BinningRule = BinningRule.NEAREST,
    fill_value: float = 0.0,
    window_id: str = "",
    dataset_id: str = "",
    canonical_subject_id: str = "",
    biological_person_id: str = "",
    session_id: str = "",
    segment_index: int = 0,
    date_is_real: bool = True,
) -> Window:
    """Place readings in ``[start, start+24h)`` onto the grid.

    Readings mapping to the same grid position are averaged (§9.2). Positions with no reading
    keep ``fill_value`` and mask False. Indices outside ``[0, 287]`` are excluded rather than
    clipped, since clipping would fabricate an observation at the boundary.
    """
    values_out = np.full(SEQUENCE_LENGTH, fill_value, dtype=np.float32)
    mask_out = np.zeros(SEQUENCE_LENGTH, dtype=bool)

    sums = np.zeros(SEQUENCE_LENGTH, dtype=np.float64)
    counts = np.zeros(SEQUENCE_LENGTH, dtype=np.int32)
    end = start + WINDOW
    for t, v in zip(timestamps, values, strict=True):
        if t < start or t >= end:
            continue
        idx = assign_index(grid_offsets(t, start), rule)
        if 0 <= idx < SEQUENCE_LENGTH:
            sums[idx] += v
            counts[idx] += 1

    observed = counts > 0
    values_out[observed] = (sums[observed] / counts[observed]).astype(np.float32)
    mask_out[observed] = True

    return Window(
        window_id=window_id,
        dataset_id=dataset_id,
        canonical_subject_id=canonical_subject_id,
        biological_person_id=biological_person_id,
        session_id=session_id,
        segment_index=segment_index,
        start_local=start,
        circadian_start_index=circadian_start_index(start),
        values=values_out,
        mask=mask_out,
        date_is_real=date_is_real,
    )


def legal_starts(
    segment_start: datetime, segment_end: datetime, *, resolution_minutes: int = GRID_MINUTES
) -> list[datetime]:
    """Every 5-minute start whose full 24-hour window fits inside the segment.

    Blueprint §9.4: a window must never cross a segment boundary (§9.1), so the last legal
    start is ``segment_end - 24h``. A segment shorter than 24 hours yields no legal starts,
    which is why Colas — whose recordings run about two days — contributes far fewer windows
    per subject than its reading count suggests.
    """
    span = segment_end - segment_start
    if span < WINDOW:
        return []
    step = timedelta(minutes=resolution_minutes)
    n = int((span - WINDOW).total_seconds() // step.total_seconds()) + 1
    return [segment_start + i * step for i in range(n)]


def non_overlapping_starts(segment_start: datetime, segment_end: datetime) -> list[datetime]:
    """Consecutive 24-hour windows with no overlap. Blueprint §9.5 — PAPER_EXACT (downstream)."""
    out: list[datetime] = []
    t = segment_start
    while t + WINDOW <= segment_end:
        out.append(t)
        t += WINDOW
    return out
