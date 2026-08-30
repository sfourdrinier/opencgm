"""Golden tests for grid assignment and segmentation.

Blueprint sections 9.1 to 9.3 as executable invariants. They exist so that a future
change to binning or segmentation fails loudly, instead of quietly shifting every
window by a step.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from opencgm_stateevent.data.timestamps import (
    PATCHES,
    SEQUENCE_LENGTH,
    STEPS_PER_PATCH,
    BinningRule,
    absolute_index,
    assign_index,
    circadian_start_index,
    grid_offsets,
    segment_readings,
)


def test_grid_geometry_is_paper_exact():
    assert SEQUENCE_LENGTH == 288
    assert PATCHES * STEPS_PER_PATCH == SEQUENCE_LENGTH


# --- segmentation boundary, blueprint §9.1 ---------------------------------------------
# The paper says gaps *longer than* one hour create a boundary, so exactly 60 minutes must
# stay internal. Off-by-one here would resegment the entire corpus.


@pytest.mark.parametrize(
    ("gap_minutes", "expected_segments"),
    [(5, 1), (59, 1), (60, 1), (60.5, 2), (61, 2), (600, 2)],
)
def test_segment_boundary_is_strictly_greater_than_one_hour(gap_minutes, expected_segments):
    t0 = datetime(2024, 1, 1, 12, 0)
    ts = [t0, t0 + timedelta(minutes=gap_minutes)]
    assert len(segment_readings(ts)) == expected_segments


def test_segments_are_contiguous_and_cover_every_reading():
    t0 = datetime(2024, 1, 1)
    ts = [
        t0,
        t0 + timedelta(minutes=5),
        t0 + timedelta(hours=3),
        t0 + timedelta(hours=3, minutes=5),
    ]
    segs = segment_readings(ts)
    assert segs == [(0, 2), (2, 4)]
    assert sum(e - s for s, e in segs) == len(ts)


def test_empty_input_yields_no_segments():
    assert segment_readings([]) == []


# --- binning rules, blueprint §9.3 ------------------------------------------------------


@pytest.mark.parametrize(
    ("u", "floor_idx", "nearest_idx"),
    [
        (0.0, 0, 0),
        (0.49, 0, 0),
        (0.5, 0, 1),  # exact tie rounds away from zero, never to even
        (0.51, 0, 1),
        (1.5, 1, 2),  # banker's rounding would give 2 here too, but...
        (2.5, 2, 3),  # ...banker's rounding would give 2. This is the guard.
        (3.999, 3, 4),
    ],
)
def test_assign_index_rules(u, floor_idx, nearest_idx):
    assert assign_index(u, BinningRule.FLOOR) == floor_idx
    assert assign_index(u, BinningRule.NEAREST) == nearest_idx


def test_nearest_is_not_bankers_rounding():
    """Blueprint §9.3 explicitly warns against relying on Python's round()."""
    assert round(2.5) == 2  # what we must NOT do
    assert assign_index(2.5, BinningRule.NEAREST) == 3


def test_nearest_error_never_exceeds_half_a_step():
    """The defining property of nearest, and the basis of DECISIONS D004."""
    for i in range(2000):
        u = i / 97.0
        err = abs(u - assign_index(u, BinningRule.NEAREST))
        assert err <= 0.5 + 1e-12


def test_floor_and_nearest_agree_on_exact_grid_multiples():
    """Why the rule choice is provably irrelevant for Shanghai's on-the-minute 15-min data."""
    for k in range(300):
        assert assign_index(float(k), BinningRule.FLOOR) == assign_index(
            float(k), BinningRule.NEAREST
        )


# --- circadian phase, blueprint §9.2 ----------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(0, 0, 0), (0, 4, 0), (0, 5, 1), (12, 0, 144), (23, 55, 287), (23, 59, 287)],
)
def test_circadian_start_index(hour, minute, expected):
    assert circadian_start_index(datetime(2024, 6, 1, hour, minute)) == expected


def test_absolute_index_wraps_at_288():
    assert absolute_index(287, 1) == 0
    assert absolute_index(287, 2) == 1
    assert absolute_index(0, 288) == 0


def test_grid_offsets_are_in_five_minute_units():
    t0 = datetime(2024, 1, 1, 0, 0)
    assert grid_offsets(t0 + timedelta(minutes=5), t0) == pytest.approx(1.0)
    assert grid_offsets(t0 + timedelta(minutes=150), t0) == pytest.approx(30.0)
    # A 24-hour window is exactly one full grid.
    assert grid_offsets(t0 + timedelta(hours=24), t0) == pytest.approx(SEQUENCE_LENGTH)
