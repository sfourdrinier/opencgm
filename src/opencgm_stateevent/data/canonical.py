"""Canonicalization: raw readings to the canonical schema. Blueprint §8.2.

Every reading keeps a path back to the source file and row it came from, and every row dropped
along the way is counted. Silent loss is the failure mode this layer exists to prevent: a
pipeline that quietly discards 6% of a cohort still trains, and still reports plausible losses.

No interpolation happens here or anywhere downstream. Absent grid positions are absent; the
physical observation mask carries that fact (§9.2).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .readers import Reading

#: Physiologically impossible readings. CGM devices clamp to roughly 40-400 mg/dL; values
#: outside this are sentinels or corruption, not measurements. Recorded, never silently kept.
MIN_PLAUSIBLE_MG_DL = 10.0
MAX_PLAUSIBLE_MG_DL = 600.0


@dataclass
class SessionReport:
    """Row accounting for one session. Blueprint §21.1 requires source row accounting."""

    session_id: str
    dataset_id: str
    source_subject_id: str
    biological_person_id: str
    raw_rows: int = 0
    dropped_exact_duplicate: int = 0
    averaged_conflicting: int = 0
    dropped_implausible: int = 0
    valid_rows: int = 0
    observed_hours: float = 0.0
    n_segments: int = 0
    date_is_real: bool = True

    def check(self) -> None:
        """Every raw row must be accounted for. This is the anti-silent-loss invariant."""
        accounted = (
            self.valid_rows
            + self.dropped_exact_duplicate
            + self.dropped_implausible
            + self.averaged_conflicting
        )
        if accounted != self.raw_rows:
            raise ValueError(
                f"{self.session_id}: {self.raw_rows} raw rows but {accounted} accounted for"
            )


@dataclass
class CanonicalSession:
    """One continuous recording session on the canonical schema."""

    session_id: str
    dataset_id: str
    source_subject_id: str
    canonical_subject_id: str
    biological_person_id: str
    timestamps: list[datetime] = field(default_factory=list)
    values_mg_dl: list[float] = field(default_factory=list)
    report: SessionReport | None = None

    def __len__(self) -> int:
        return len(self.timestamps)


def split_identity(r: Reading) -> tuple[str, str]:
    """(biological_person_id, canonical_subject_id) for a reading.

    Shanghai encodes repeat visits as separate workbooks named ``{person}_{visit}_{date}``, so
    the same human appears under several session ids. Blueprint §6.4 requires person and visit
    identity to be tracked separately, or a person-disjoint split is impossible to build later.
    """
    person = f"{r.dataset_id}/person_{r.source_subject_id}"
    return person, f"{r.dataset_id}/{r.source_subject_id}"


def canonicalize(
    readings: Iterable[Reading], *, dedup_exact: bool = True
) -> Iterator[CanonicalSession]:
    """Group readings into sessions, deduplicate, and account for every dropped row.

    Deduplication follows DECISIONS D007: exact duplicates (same timestamp and value within a
    session) are dropped; same-timestamp readings with differing values are averaged and
    counted separately. Both are applied only on an exact timestamp match — readings merely
    close in time are left for the binning rule to place.
    """
    by_session: dict[str, list[Reading]] = defaultdict(list)
    for r in readings:
        by_session[r.session_id].append(r)

    for session_id, rows in by_session.items():
        first = rows[0]
        person, canonical = split_identity(first)
        rep = SessionReport(
            session_id=session_id,
            dataset_id=first.dataset_id,
            source_subject_id=first.source_subject_id,
            biological_person_id=person,
            raw_rows=len(rows),
            date_is_real=first.date_is_real,
        )

        by_ts: dict[datetime, list[float]] = defaultdict(list)
        for r in rows:
            if not (MIN_PLAUSIBLE_MG_DL <= r.glucose_mg_dl <= MAX_PLAUSIBLE_MG_DL):
                rep.dropped_implausible += 1
                continue
            by_ts[r.local_datetime].append(r.glucose_mg_dl)

        timestamps: list[datetime] = []
        values: list[float] = []
        for ts in sorted(by_ts):
            vals = by_ts[ts]
            if len(vals) > 1:
                if len(set(vals)) == 1 and dedup_exact:
                    rep.dropped_exact_duplicate += len(vals) - 1
                    value = vals[0]
                else:
                    rep.averaged_conflicting += len(vals) - 1
                    value = sum(vals) / len(vals)
            else:
                value = vals[0]
            timestamps.append(ts)
            values.append(value)

        rep.valid_rows = len(timestamps)
        rep.check()

        yield CanonicalSession(
            session_id=session_id,
            dataset_id=first.dataset_id,
            source_subject_id=first.source_subject_id,
            canonical_subject_id=canonical,
            biological_person_id=person,
            timestamps=timestamps,
            values_mg_dl=values,
            report=rep,
        )


def reconcile(sessions: Iterable[CanonicalSession]) -> dict[str, object]:
    """Aggregate row accounting across sessions, for the PR 2 acceptance gate."""
    from .timestamps import segment_readings

    out = {
        "sessions": 0,
        "subjects": set(),
        "people": set(),
        "raw_rows": 0,
        "valid_rows": 0,
        "dropped_exact_duplicate": 0,
        "averaged_conflicting": 0,
        "dropped_implausible": 0,
        "segments": 0,
        "observed_hours": 0.0,
    }
    for s in sessions:
        r = s.report
        assert r is not None
        out["sessions"] += 1
        out["subjects"].add(s.canonical_subject_id)
        out["people"].add(s.biological_person_id)
        for key in (
            "raw_rows",
            "valid_rows",
            "dropped_exact_duplicate",
            "averaged_conflicting",
            "dropped_implausible",
        ):
            out[key] += getattr(r, key)
        segs = segment_readings(s.timestamps)
        out["segments"] += len(segs)
        out["observed_hours"] += sum(
            (s.timestamps[e - 1] - s.timestamps[b]).total_seconds() / 3600.0 for b, e in segs
        )
    out["subjects"] = len(out["subjects"])
    out["people"] = len(out["people"])
    out["observed_hours"] = round(out["observed_hours"])
    return out


def report_rows(sessions: Iterable[CanonicalSession]) -> list[dict[str, object]]:
    return [asdict(s.report) for s in sessions if s.report is not None]
