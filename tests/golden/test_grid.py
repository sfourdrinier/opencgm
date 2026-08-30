"""Grid and mask invariants. Blueprint §9.2, §9.5, §10.2.

The fill-value invariance test (§10.2) is the important one. If it ever fails, some downstream
computation is reading values at masked positions, which means the model is training on
fabricated zeros and nothing else in the pipeline will say so.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta

import numpy as np
import pytest

from opencgm_stateevent.data.grid import (
    build_window,
    legal_starts,
    non_overlapping_starts,
)
from opencgm_stateevent.data.timestamps import PATCHES, SEQUENCE_LENGTH

T0 = datetime(2024, 3, 1, 0, 0)


def dense_day(cadence_minutes: int = 5, value: float = 100.0):
    n = (24 * 60) // cadence_minutes
    ts = [T0 + timedelta(minutes=cadence_minutes * i) for i in range(n)]
    vs = [value + i * 0.1 for i in range(n)]
    return ts, vs


def test_full_day_at_five_minutes_fills_every_position():
    ts, vs = dense_day(5)
    w = build_window(ts, vs, T0)
    assert w.mask.all()
    assert w.n_observed == SEQUENCE_LENGTH
    assert w.coverage == 1.0


def test_fifteen_minute_source_stays_sparse():
    """Blueprint §21.1: 15-minute sources must remain sparse on the 5-minute grid.

    Exactly one position in three observed. If this ever fills in, something interpolated.
    """
    ts, vs = dense_day(15)
    w = build_window(ts, vs, T0)
    assert w.n_observed == 96
    assert w.coverage == pytest.approx(1 / 3)
    assert w.patch_density() == pytest.approx(np.full(PATCHES, 4 / 12, dtype=np.float32))


def test_no_interpolation_across_a_gap():
    """A hole in the source stays a hole. Nothing fills it."""
    ts, vs = dense_day(5)
    keep = [(t, v) for t, v in zip(ts, vs, strict=True) if not (6 <= t.hour < 9)]
    w = build_window([t for t, _ in keep], [v for _, v in keep], T0)
    assert w.n_observed == SEQUENCE_LENGTH - 36
    assert not w.mask[72:108].any()


@pytest.mark.parametrize("fill", [0.0, -999.0, 1e6, float("nan")])
def test_fill_value_cannot_affect_valid_outputs(fill):
    """Blueprint §10.2 fill-value invariance, at the grid layer.

    Values at observed positions, the mask, and patch density must all be identical no matter
    what number sits in the unobserved slots.
    """
    ts, vs = dense_day(15)
    ref = build_window(ts, vs, T0, fill_value=0.0)
    alt = build_window(ts, vs, T0, fill_value=fill)
    assert np.array_equal(ref.mask, alt.mask)
    assert ref.values[ref.mask] == pytest.approx(alt.values[alt.mask])
    assert ref.patch_density() == pytest.approx(alt.patch_density())


def test_colliding_readings_are_averaged():
    """Blueprint §9.2: multiple real readings on one position are averaged, not overwritten."""
    ts = [T0, T0 + timedelta(seconds=30), T0 + timedelta(minutes=5)]
    vs = [100.0, 200.0, 50.0]
    w = build_window(ts, vs, T0)
    assert w.values[0] == pytest.approx(150.0)
    assert w.values[1] == pytest.approx(50.0)
    assert w.n_observed == 2


def test_readings_outside_the_window_are_excluded_not_clipped():
    """Clipping would fabricate an observation at the boundary."""
    ts = [T0 - timedelta(minutes=5), T0, T0 + timedelta(hours=24)]
    vs = [1.0, 100.0, 999.0]
    w = build_window(ts, vs, T0)
    assert w.n_observed == 1
    assert w.values[0] == pytest.approx(100.0)
    assert 999.0 not in set(w.values.tolist())


def test_window_end_is_exclusive():
    """[start, start+24h). The 288th position belongs to the next window."""
    w = build_window([T0 + timedelta(hours=24)], [123.0], T0)
    assert w.n_observed == 0


def test_patch_density_shape_and_range():
    ts, vs = dense_day(5)
    d = build_window(ts, vs, T0).patch_density()
    assert d.shape == (PATCHES,)
    assert ((d >= 0) & (d <= 1)).all()


def test_empty_window_is_finite_and_unobserved():
    """Blueprint §9.6: reject zero-observation windows, but never produce NaN in doing so."""
    w = build_window([], [], T0)
    assert w.n_observed == 0
    assert np.isfinite(w.values).all()
    assert w.patch_density().sum() == 0.0


def test_circadian_index_tracks_window_start():
    assert build_window([], [], datetime(2024, 3, 1, 0, 0)).circadian_start_index == 0
    assert build_window([], [], datetime(2024, 3, 1, 12, 0)).circadian_start_index == 144
    w = build_window([], [], datetime(2024, 3, 1, 18, 30))
    assert w.circadian_start_index == 222
    assert w.absolute_indices()[0] == 222
    assert w.absolute_indices()[66] == 0  # wraps at midnight


# --- window enumeration, blueprint §9.4 and §9.5 ---------------------------------------


def test_no_legal_start_in_a_segment_shorter_than_24h():
    """Why Colas, whose recordings run ~2 days, yields few windows per subject."""
    assert legal_starts(T0, T0 + timedelta(hours=23, minutes=55)) == []


def test_exactly_one_legal_start_at_exactly_24h():
    assert legal_starts(T0, T0 + timedelta(hours=24)) == [T0]


def test_legal_starts_step_by_five_minutes_and_all_fit():
    starts = legal_starts(T0, T0 + timedelta(hours=25))
    assert len(starts) == 13  # 60 min / 5 min + 1
    assert starts[1] - starts[0] == timedelta(minutes=5)
    assert starts[-1] + timedelta(hours=24) <= T0 + timedelta(hours=25)


def test_non_overlapping_starts_do_not_overlap():
    starts = non_overlapping_starts(T0, T0 + timedelta(days=3, hours=5))
    assert len(starts) == 3
    for a, b in itertools.pairwise(starts):
        assert b - a == timedelta(hours=24)
