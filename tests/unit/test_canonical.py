"""Canonicalization invariants.

The failure this guards against is silent loss: a pipeline that quietly drops rows still
trains and still reports plausible losses. Blueprint §21.1 requires source row accounting.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from opencgm_stateevent.data.canonical import (
    MAX_PLAUSIBLE_MG_DL,
    canonicalize,
    reconcile,
)
from opencgm_stateevent.data.readers import Reading

T0 = datetime(2024, 1, 1, 12, 0)


def mk(i: int, value: float, *, session: str = "ds/s1", subject: str = "s1", minutes: int = 5):
    return Reading("ds", subject, session, T0 + timedelta(minutes=minutes * i), value, "f.csv", i)


def test_every_raw_row_is_accounted_for():
    """The core anti-silent-loss invariant; canonicalize() raises if it fails."""
    readings = [mk(i, 100 + i) for i in range(10)]
    (s,) = list(canonicalize(readings))
    r = s.report
    assert r.raw_rows == 10
    assert r.valid_rows == 10
    r.check()


def test_exact_duplicates_are_dropped_not_averaged():
    """DECISIONS D007: identical value at an identical timestamp carries no information."""
    a = mk(0, 120.0)
    b = Reading("ds", "s1", "ds/s1", a.local_datetime, 120.0, "f.csv", 1)
    (s,) = list(canonicalize([a, b]))
    assert len(s) == 1
    assert s.values_mg_dl == [120.0]
    assert s.report.dropped_exact_duplicate == 1
    assert s.report.averaged_conflicting == 0


def test_conflicting_values_are_averaged_and_counted_separately():
    """§9.2 averages collisions, but the count must stay visible for the audit."""
    a = mk(0, 100.0)
    b = Reading("ds", "s1", "ds/s1", a.local_datetime, 140.0, "f.csv", 1)
    (s,) = list(canonicalize([a, b]))
    assert s.values_mg_dl == [120.0]
    assert s.report.averaged_conflicting == 1
    assert s.report.dropped_exact_duplicate == 0


def test_implausible_values_are_dropped_and_counted():
    readings = [mk(0, 100.0), mk(1, MAX_PLAUSIBLE_MG_DL + 1), mk(2, -5.0)]
    (s,) = list(canonicalize(readings))
    assert len(s) == 1
    assert s.report.dropped_implausible == 2
    s.report.check()


def test_readings_are_sorted_by_time():
    """Sources are not guaranteed ordered; segmentation depends on order."""
    readings = [mk(3, 103.0), mk(1, 101.0), mk(2, 102.0)]
    (s,) = list(canonicalize(readings))
    assert s.timestamps == sorted(s.timestamps)


def test_sessions_are_never_merged():
    """Blueprint §21.1: no cross-session merges. A window may not span two sessions."""
    readings = [mk(0, 100.0, session="ds/a"), mk(0, 110.0, session="ds/b")]
    sessions = {s.session_id for s in canonicalize(readings)}
    assert sessions == {"ds/a", "ds/b"}


def test_repeat_visits_share_a_biological_person():
    """Shanghai encodes repeat visits as separate files; §6.4 needs person identity preserved.

    Without this, a person-disjoint split is impossible and leakage tests cannot pass.
    """
    v1 = Reading("shanghai_t2dm", "2001", "shanghai_t2dm/2001/visit_0", T0, 120.0, "a.xlsx", 0)
    v2 = Reading("shanghai_t2dm", "2001", "shanghai_t2dm/2001/visit_1", T0, 130.0, "b.xlsx", 0)
    sessions = list(canonicalize([v1, v2]))
    assert len(sessions) == 2
    assert len({s.biological_person_id for s in sessions}) == 1


def test_reconcile_counts_people_separately_from_sessions():
    v1 = Reading("ds", "p1", "ds/p1/v0", T0, 120.0, "a", 0)
    v2 = Reading("ds", "p1", "ds/p1/v1", T0, 130.0, "b", 0)
    v3 = Reading("ds", "p2", "ds/p2/v0", T0, 140.0, "c", 0)
    r = reconcile(canonicalize([v1, v2, v3]))
    assert r["sessions"] == 3
    assert r["people"] == 2


def test_no_value_is_invented():
    """There is no interpolation anywhere. Output length never exceeds distinct input times."""
    readings = [mk(i, 100.0 + i) for i in range(5)]
    readings.append(mk(20, 150.0))  # a 75-minute gap; must not be filled
    (s,) = list(canonicalize(readings))
    assert len(s) == 6


@pytest.mark.parametrize("value", [39.0, 400.0, 100.0])
def test_clinically_plausible_values_are_kept(value):
    (s,) = list(canonicalize([mk(0, value)]))
    assert s.values_mg_dl == [value]
