"""The application entry point. Live sensor data is messier than a curated cohort.

What is pinned here is what a real Dexcom stream does that a dataset does not: irregular arrival
times, dropouts of arbitrary length, duplicate readings, and a stream that does not begin at
midnight. Each of those has a correct answer already settled elsewhere in the codebase, and this
module has to inherit it rather than reinvent it -- above all that gaps stay gaps.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest
import torch

from opencgm_stateevent.infer import (
    HYPO_1,
    MIN_OBSERVED,
    TARGET_HIGH,
    Analyser,
    DayReport,
    PhenotypeScore,
    compute_metrics,
    similarity,
)
from opencgm_stateevent.model.model import OpenCGMStateEvent

START = datetime(2026, 3, 1, 0, 0)


def stream(n: int = 288, every: int = 5, value: float = 120.0):
    return [(START + timedelta(minutes=every * i), value) for i in range(n)]


def analyser() -> Analyser:
    model = OpenCGMStateEvent().eval()

    class Ref:
        weights_sha256 = "x"

    return Analyser(model, Ref(), heads=None, device="cpu")


def test_gaps_stay_gaps():
    """The mask must show the dropout. Interpolating it would be invisible and wrong."""
    readings = [r for r in stream() if not (100 <= (r[0] - START).seconds / 300 < 150)]
    _, mask, _, _ = analyser().window_from(readings, start=START)
    assert mask.sum() == len(readings)
    assert not mask[100:150].any(), "a sensor dropout was filled in"


def test_window_defaults_to_the_last_24_hours():
    """A live app wants the most recent day, not a calendar day."""
    readings = stream(n=576)  # two days
    _, mask, _, start = analyser().window_from(readings)
    assert start > START, "window did not advance to the recent end of the stream"
    assert mask.sum() == 288


def test_readings_need_not_be_sorted_or_aligned():
    """Real streams arrive out of order and off-grid."""
    readings = stream(n=48)
    shuffled = [(t + timedelta(seconds=37), v) for t, v in readings][::-1]
    _, mask, _, _ = analyser().window_from(shuffled, start=START)
    assert mask.sum() == 48


def test_metrics_ignore_unobserved_positions_entirely():
    """The fill value must not reach any statistic. Same invariant as §10.2 on the grid."""
    values = np.full(288, 120.0, dtype=np.float32)
    mask = np.zeros(288, dtype=bool)
    mask[:100] = True
    values[100:] = -999.0  # poison; nothing may read it

    m = compute_metrics(values, mask)
    assert m.n_observed == 100
    assert m.mean_glucose == pytest.approx(120.0)
    assert m.min_glucose == pytest.approx(120.0)
    assert m.time_in_range == pytest.approx(1.0)


def test_time_in_range_uses_the_standard_thresholds():
    values = np.concatenate([
        np.full(72, 60.0),    # below 70
        np.full(144, 120.0),  # in range
        np.full(72, 200.0),   # above 180
    ]).astype(np.float32)
    mask = np.ones(288, dtype=bool)
    m = compute_metrics(values, mask)
    assert m.time_in_range == pytest.approx(0.5)
    assert m.time_below_70 == pytest.approx(0.25)
    assert m.time_above_180 == pytest.approx(0.25)
    assert HYPO_1 < 120.0 < TARGET_HIGH


def test_sparse_day_is_warned_about_not_silently_scored():
    report = analyser().analyse_day(stream(n=10, every=5), start=START)
    assert report.warnings, "a 10-reading day produced no warning"
    assert str(MIN_OBSERVED) in report.warnings[0] or "unreliable" in report.warnings[0]


def test_report_serialises_and_carries_the_phrasing():
    report = analyser().analyse_day(stream(), start=START)
    assert len(report.embedding) == 128
    payload = report.to_json()
    assert '"time_in_range"' in payload
    assert '"variability_is_stable"' in payload


def test_a_head_below_the_signal_floor_refuses_to_be_read_as_a_percentage():
    weak = PhenotypeScore(
        task="cgmacros:hyperlipidemia", dataset="cgmacros", probability=0.71,
        predicted_class=1, reliability=0.41, reliability_sd=0.1,
        reliability_subject_level=0.43, n_subjects_learned_from=45, has_signal=False,
    )
    text = weak.population_phrasing
    assert "No reliable signal" in text
    assert "71%" not in text, "a below-chance head presented its output as a percentage"

    strong = PhenotypeScore(
        task="hall:glucotype", dataset="hall", probability=0.71, predicted_class=1,
        reliability=0.89, reliability_sd=0.05, reliability_subject_level=0.91,
        n_subjects_learned_from=56, has_signal=True,
    )
    assert "71%" in strong.population_phrasing
    assert "56 subjects" in strong.population_phrasing
    assert "you have" not in strong.population_phrasing.lower(), "phrased as a diagnosis"


def test_similarity_is_one_on_the_diagonal_and_symmetric():
    reports = [
        DayReport(start=START, metrics=compute_metrics(
            np.full(288, 120.0, dtype=np.float32), np.ones(288, dtype=bool)
        ), embedding=list(e))
        for e in (np.eye(3) + 0.1)
    ]
    s = similarity(reports)
    assert np.allclose(np.diag(s), 1.0)
    assert np.allclose(s, s.T)


def test_stream_splits_into_non_overlapping_days():
    reports = analyser().analyse_stream(stream(n=288 * 3), days=3)
    assert len(reports) == 3
    starts = [r.start for r in reports]
    assert starts == sorted(starts), "days out of order"
    for earlier, later in pairwise(starts):
        assert (later - earlier) == timedelta(hours=24)


def test_a_head_refuses_a_day_whose_sampling_rate_it_never_saw():
    """The failure this pins was observed on real data, not imagined.

    The Libre heads are fitted on 15-minute recordings, ~31% dense on a 5-minute grid; the Dexcom
    heads on ~84%. Applied across that gap the Libre head scored a real subject-day at 100% for
    the same question its Dexcom counterpart scored at 1%. Both were confident and one of them
    had to be nonsense.
    """
    from opencgm_stateevent.infer import _applicable

    libre = {"coverage_p05": 0.288, "coverage_p95": 0.323}
    dexcom = {"coverage_p05": 0.736, "coverage_p95": 0.903}

    ok, note = _applicable(libre, 0.31)
    assert ok and note == ""

    ok, note = _applicable(libre, 0.84)
    assert not ok, "a 15-minute head accepted a 5-minute day"
    assert "sampling rates differ" in note

    assert _applicable(dexcom, 0.84)[0]
    assert not _applicable(dexcom, 0.31)[0], "a 5-minute head accepted a 15-minute day"

    # An ordinary sensor dropout must not trip the guard.
    assert _applicable(dexcom, 0.70)[0], "a normal gappy Dexcom day was rejected"


def test_a_head_without_a_recorded_band_is_not_silently_rejected():
    """Heads fitted before the band existed still work; they just cannot be checked."""
    from opencgm_stateevent.infer import _applicable

    assert _applicable({"roc_auc": 0.8}, 0.5) == (True, "")


def test_an_inapplicable_score_never_renders_as_a_percentage():
    inapplicable = PhenotypeScore(
        task="cgmacros:insulin_resistance[cgmacros_libre]", dataset="cgmacros",
        probability=float("nan"), predicted_class=-1, reliability=0.86, reliability_sd=0.07,
        reliability_subject_level=0.93, n_subjects_learned_from=44, has_signal=True,
        applicable=False, applicability_note="sampling rates differ too much",
    )
    text = inapplicable.population_phrasing
    assert "not applicable" in text
    assert "%" not in text.split("not applicable")[0]


def test_an_mmol_per_litre_export_is_rejected_not_analysed():
    """4-12 mmol/L is 72-216 mg/dL. Read as mg/dL it describes fatal hypoglycaemia.

    Every other guard passes on such a stream -- the coverage is fine, the timestamps are fine --
    so nothing else would catch it, and D019 makes it worse rather than better because the model
    now reads absolute level instead of normalising it away.
    """
    from opencgm_stateevent.infer import check_units

    rng = np.random.default_rng(0)
    check_units(rng.normal(120, 25, 500))  # ordinary mg/dL: fine

    with pytest.raises(ValueError, match="mmol/L"):
        check_units(rng.normal(6.7, 1.4, 500))  # the same person, in mmol/L

    with pytest.raises(ValueError, match=r"not survivable|outside"):
        check_units(np.full(500, 5.5))


def test_overnight_and_dawn_follow_the_clock_not_the_window_start():
    """A live window ends at the latest reading, so it starts at an arbitrary hour.

    Grid positions 0-71 are only 00:00-06:00 when the window begins at midnight. For a report
    ending at 14:00 they are the previous afternoon, and labelling that "overnight" is false.
    """
    values = np.full(288, 200.0, dtype=np.float32)
    values[:72] = 90.0  # the genuinely overnight stretch, at clock positions 0-71
    mask = np.ones(288, dtype=bool)

    aligned = compute_metrics(values, mask, circadian_start=0)
    assert aligned.overnight_mean == pytest.approx(90.0)

    # Same day, but the window starts at 12:00 so the overnight hours sit late in the array.
    rolled = np.roll(values, -144)
    shifted = compute_metrics(rolled, mask, circadian_start=144)
    assert shifted.overnight_mean == pytest.approx(90.0), (
        "overnight was read off grid position instead of clock time"
    )


def test_heads_fitted_on_a_different_architecture_are_refused(tmp_path):
    """The D019 failure again: identical weights, different front end, different embedding."""
    import pickle

    from opencgm_stateevent.model.model import OpenCGMStateEvent

    model = OpenCGMStateEvent(raw_statistics=True).eval()
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "epoch": 1,
                "config": {"seed": 17, "raw_statistics": True}}, checkpoint)

    heads = tmp_path / "heads.pkl"
    with heads.open("wb") as fh:
        pickle.dump({"encoder": {"architecture": '{"raw_statistics": false}',
                                 "dtype": "float32", "backend": "pytorch"}, "heads": {}}, fh)

    with pytest.raises(ValueError, match="different encoder"):
        Analyser.load(checkpoint, heads=heads, device="cpu")
