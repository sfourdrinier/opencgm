"""Overlapping pretraining-window sampling. Blueprint §9.4.

The paper says windows are sampled at random from each segment with a fixed seed and a
per-segment "coverage ratio" between 20% and 80%, but never defines coverage ratio. The two
readings produce materially different corpus sizes, so this is not a detail: it sets how much
data the model actually sees.

Both are implemented here and compared in reports/window_sampler.md. DECISIONS D005 records
the choice. Neither is presented as the paper's method.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

import numpy as np

from .grid import WINDOW, legal_starts

COVERAGE_MIN = 0.20
COVERAGE_MAX = 0.80


class SamplerCandidate(StrEnum):
    #: Coverage = fraction of legal start positions selected. Blueprint's default reading.
    LEGAL_START_FRACTION = "legal_start_fraction"
    #: Coverage = fraction of the segment's timeline covered by the union of chosen windows.
    UNION_TIMELINE = "union_timeline"


def stable_rng(global_seed: int, *parts: str) -> np.random.Generator:
    """Deterministic per-segment RNG. Blueprint §9.4 step 4, §17.3.

    Derived by hashing the identity parts rather than advancing a shared stream, so a segment's
    sample does not depend on how many segments were processed before it. That is what makes
    the manifest reproducible under a different worker count or corpus order.
    """
    key = "|".join((str(global_seed), *parts)).encode()
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class SegmentSample:
    session_id: str
    segment_index: int
    n_legal_starts: int
    coverage_ratio: float
    starts: tuple[datetime, ...]

    @property
    def n_windows(self) -> int:
        return len(self.starts)


def sample_segment(
    session_id: str,
    segment_index: int,
    segment_start: datetime,
    segment_end: datetime,
    *,
    candidate: SamplerCandidate = SamplerCandidate.LEGAL_START_FRACTION,
    global_seed: int = 17,
    dataset_id: str = "",
) -> SegmentSample | None:
    """Sample overlapping 24-hour windows from one continuous segment.

    Returns None when the segment is shorter than 24 hours and admits no legal window.
    Starts are sorted before returning so the serialized manifest is order-stable (§9.4 step 5).
    """
    starts = legal_starts(segment_start, segment_end)
    if not starts:
        return None

    rng = stable_rng(global_seed, dataset_id, session_id, str(segment_index))
    coverage = float(rng.uniform(COVERAGE_MIN, COVERAGE_MAX))

    if candidate is SamplerCandidate.LEGAL_START_FRACTION:
        n = max(1, round(coverage * len(starts)))
        chosen_idx = rng.choice(len(starts), size=min(n, len(starts)), replace=False)
        chosen = sorted(starts[int(i)] for i in chosen_idx)
    else:
        chosen = _union_coverage(starts, segment_start, segment_end, coverage, rng)

    return SegmentSample(
        session_id=session_id,
        segment_index=segment_index,
        n_legal_starts=len(starts),
        coverage_ratio=coverage,
        starts=tuple(chosen),
    )


def _union_coverage(
    starts: Sequence[datetime],
    segment_start: datetime,
    segment_end: datetime,
    target: float,
    rng: np.random.Generator,
) -> list[datetime]:
    """Candidate B: add windows until their union covers ``target`` of the segment.

    Coverage is measured in 5-minute units over the segment's own span. Because a single
    24-hour window already covers the whole of a 24-hour segment, this candidate collapses to
    one window for short segments regardless of the sampled ratio — which is exactly the
    behavioural difference from Candidate A that the comparison report quantifies.
    """
    total_units = max(1, int((segment_end - segment_start).total_seconds() // 300))
    window_units = int(WINDOW.total_seconds() // 300)
    covered = np.zeros(total_units, dtype=bool)
    order = rng.permutation(len(starts))
    chosen: list[datetime] = []
    for i in order:
        s = starts[int(i)]
        lo = int((s - segment_start).total_seconds() // 300)
        covered[lo : lo + window_units] = True
        chosen.append(s)
        if covered.mean() >= target:
            break
    return sorted(chosen)


def sample_sessions(
    segments: Sequence[tuple[str, int, datetime, datetime, str]],
    *,
    candidate: SamplerCandidate = SamplerCandidate.LEGAL_START_FRACTION,
    global_seed: int = 17,
) -> list[SegmentSample]:
    """Sample every segment. Input tuples are (session_id, index, start, end, dataset_id)."""
    out = []
    for session_id, idx, start, end, dataset_id in segments:
        s = sample_segment(
            session_id,
            idx,
            start,
            end,
            candidate=candidate,
            global_seed=global_seed,
            dataset_id=dataset_id,
        )
        if s is not None:
            out.append(s)
    return out


def summarize(samples: Sequence[SegmentSample]) -> dict[str, float | int]:
    if not samples:
        return {"segments": 0, "windows": 0, "mean_windows_per_segment": 0.0}
    counts = [s.n_windows for s in samples]
    legal = [s.n_legal_starts for s in samples]
    return {
        "segments": len(samples),
        "windows": int(sum(counts)),
        "legal_starts_total": int(sum(legal)),
        "mean_windows_per_segment": round(float(np.mean(counts)), 2),
        "median_windows_per_segment": int(np.median(counts)),
        "max_windows_per_segment": int(max(counts)),
        "selected_fraction_of_legal": round(sum(counts) / max(1, sum(legal)), 4),
    }


def hours_covered(samples: Sequence[SegmentSample]) -> float:
    """Total window-hours, counting overlap. This is what an epoch actually costs."""
    return sum(s.n_windows * 24.0 for s in samples)


def window_span(start: datetime) -> tuple[datetime, datetime]:
    return start, start + timedelta(hours=24)
