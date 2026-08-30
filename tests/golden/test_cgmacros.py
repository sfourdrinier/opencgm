"""Recovering CGMacros' native observations from its interpolated release.

CGMacros 1.0.0 publishes a 1-minute grid in which both CGM columns have been linearly
interpolated from their native cadence. Blueprint §19.10 asks for the "original raw Dexcom and
Libre exports, not its interpolated CGM columns", but the release contains no such export.

Reading the published column would silently break the invariant the whole project rests on: the
physical observation mask is authoritative. A Dexcom day would report 288 observations when 288/5
were measured and the rest manufactured, and every density weight downstream would be wrong.

Linear interpolation is recoverable exactly, and these tests check that it is rather than
assuming it. The decisive one is the round trip: re-interpolating the recovered points must
reproduce the published file. That can only hold if the recovered points are the originals.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from opencgm_stateevent.data.readers import (
    CGMACROS_NATIVE_MINUTES,
    read_cgmacros,
    recover_observations,
)

ROOT = Path("data/raw/cgmacros")
ARCHIVE = ROOT / "1.0.0/CGMacros_dateshifted365.zip"
available = pytest.mark.skipif(not ARCHIVE.exists(), reason="CGMacros archive not present")


# --- the primitive, on synthetic data -------------------------------------------------------


def test_recovers_the_knots_of_an_exactly_linear_interpolation():
    knots_t = [0.0, 5.0, 10.0, 15.0]
    knots_v = [100.0, 118.0, 90.0, 95.0]
    minutes = [float(i) for i in range(16)]
    values = list(np.interp(minutes, knots_t, knots_v))
    assert [minutes[i] for i in recover_observations(minutes, values)] == knots_t


def test_a_straight_line_keeps_only_its_endpoints():
    """Collinear knots are genuinely unrecoverable, so the answer is the endpoints.

    A sensor reporting a perfectly constant slope across three readings is indistinguishable from
    two readings interpolated. This under-counts observations, which is the safe direction: it
    can only make density look lower than it is, never higher.
    """
    minutes = [float(i) for i in range(11)]
    values = [100.0 + i for i in range(11)]
    assert recover_observations(minutes, values) == [0, 10]


def test_handles_a_series_too_short_to_have_a_slope():
    assert recover_observations([0.0, 1.0], [90.0, 91.0]) == [0, 1]


def test_ignores_missing_values():
    minutes = [0.0, 1.0, 2.0, 3.0, 4.0]
    values = [100.0, None, 110.0, None, 100.0]
    assert recover_observations(minutes, values) == [0, 2, 4]


# --- the real archive -----------------------------------------------------------------------


def _series(subject: str, column: str):
    import csv
    import io
    from datetime import datetime

    with zipfile.ZipFile(ARCHIVE) as z:
        name = f"CGMacros/CGMacros-{subject}/CGMacros-{subject}.csv"
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(name), "utf-8-sig")))
    stamps = [datetime.fromisoformat(r["Timestamp"].strip()) for r in rows]
    origin = stamps[0]
    minutes = [(t - origin).total_seconds() / 60.0 for t in stamps]
    values = [float(r[column]) if r[column] not in ("", "NaN") else None for r in rows]
    return minutes, values


@available
@pytest.mark.parametrize("subject", ["001", "017", "033"])
@pytest.mark.parametrize("column", list(CGMACROS_NATIVE_MINUTES))
def test_reinterpolating_the_recovered_points_reproduces_the_published_file(subject, column):
    """The decisive check. Only the true observations can regenerate the file."""
    minutes, values = _series(subject, column)
    keep = recover_observations(minutes, values)
    t = np.asarray(minutes)
    v = np.asarray([np.nan if x is None else x for x in values])
    rebuilt = np.interp(t, t[keep], v[keep])
    observed = ~np.isnan(v)
    assert np.max(np.abs(rebuilt[observed] - v[observed])) < 1e-9


@available
@pytest.mark.parametrize("subject", ["001", "017", "033"])
@pytest.mark.parametrize(("column", "native"), list(CGMACROS_NATIVE_MINUTES.items()))
def test_recovered_timestamps_land_on_the_sensors_native_grid(subject, column, native):
    """Independent confirmation: the recovered points are spaced as the sensor samples."""
    minutes, values = _series(subject, column)
    gaps = np.diff(np.asarray(minutes)[recover_observations(minutes, values)])
    on_grid = np.mean(np.abs(gaps / native - np.round(gaps / native)) < 1e-6)
    assert on_grid > 0.99
    assert np.median(gaps) == pytest.approx(native)


@available
def test_the_published_column_really_is_interpolated():
    """Control. If CGMacros ever ships raw exports, this fails and the workaround can go.

    A native 5-minute Dexcom series on a 1-minute grid would be mostly missing. This one is
    almost fully populated, which is the interpolation.
    """
    _, values = _series("001", "Dexcom GL")
    populated = sum(v is not None for v in values) / len(values)
    assert populated > 0.9


@available
def test_the_reader_yields_far_fewer_readings_than_the_file_has_rows():
    readings = list(read_cgmacros(ROOT, "Dexcom GL"))
    subjects = {r.source_subject_id for r in readings}
    assert len(subjects) == 45  # §6.8 "45 participants"
    # 45 subjects x ~10 days x 288 five-minute readings is of order 100k, not the 660k rows the
    # 1-minute grid contains.
    assert 50_000 < len(readings) < 200_000


@available
def test_both_sensors_cover_the_same_participants():
    """§19.3 pairs Dexcom and Libre from the same person in the same split."""
    dex = {r.source_subject_id for r in read_cgmacros(ROOT, "Dexcom GL")}
    libre = {r.source_subject_id for r in read_cgmacros(ROOT, "Libre GL")}
    assert dex == libre


@available
def test_the_two_sensors_are_kept_in_separate_sessions():
    """§19.10 trains and evaluates sensors separately at native cadence."""
    dex = {r.session_id for r in read_cgmacros(ROOT, "Dexcom GL")}
    libre = {r.session_id for r in read_cgmacros(ROOT, "Libre GL")}
    assert not (dex & libre)


@available
def test_no_reading_is_flagged_as_a_reconstructed_date():
    assert all(r.date_is_real for r in read_cgmacros(ROOT, "Dexcom GL"))
