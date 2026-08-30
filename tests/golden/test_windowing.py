"""Window sampler invariants. Blueprint §9.4.

Determinism is the property that matters most here. The window manifest is frozen before the
headline run (§17.4), and five seeds must be comparable, so the same identity and seed have to
produce the same windows regardless of processing order or worker count.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from opencgm_stateevent.data.windowing import (
    COVERAGE_MAX,
    COVERAGE_MIN,
    SamplerCandidate,
    sample_segment,
    stable_rng,
    summarize,
)

T0 = datetime(2024, 5, 1, 0, 0)


def seg(days: float, **kw):
    return sample_segment("ds/s1", 0, T0, T0 + timedelta(days=days), dataset_id="ds", **kw)


def test_segment_shorter_than_24h_yields_nothing():
    assert seg(0.99) is None


def test_sampling_is_deterministic_for_the_same_identity_and_seed():
    a = seg(5, global_seed=17)
    b = seg(5, global_seed=17)
    assert a.starts == b.starts
    assert a.coverage_ratio == pytest.approx(b.coverage_ratio)


def test_different_seeds_give_different_samples():
    assert seg(5, global_seed=17).starts != seg(5, global_seed=29).starts


def test_sampling_does_not_depend_on_processing_order():
    """Hash-derived streams, not a shared advancing RNG. Blueprint §9.4 step 4.

    A shared stream would make a segment's sample depend on how many segments preceded it,
    which silently breaks manifest reproducibility under a different worker count.
    """
    direct = sample_segment("ds/s2", 3, T0, T0 + timedelta(days=4), dataset_id="ds")
    for other in range(50):  # advance nothing shared
        sample_segment(f"ds/other{other}", 0, T0, T0 + timedelta(days=2), dataset_id="ds")
    again = sample_segment("ds/s2", 3, T0, T0 + timedelta(days=4), dataset_id="ds")
    assert direct.starts == again.starts


def test_distinct_segments_of_one_session_sample_independently():
    a = sample_segment("ds/s1", 0, T0, T0 + timedelta(days=4), dataset_id="ds")
    b = sample_segment("ds/s1", 1, T0, T0 + timedelta(days=4), dataset_id="ds")
    assert a.starts != b.starts


def test_coverage_ratio_stays_in_the_paper_range():
    """20-80% is PAPER_EXACT; only its meaning is inferred (D005)."""
    for i in range(200):
        s = sample_segment(f"ds/s{i}", 0, T0, T0 + timedelta(days=6), dataset_id="ds")
        assert COVERAGE_MIN <= s.coverage_ratio <= COVERAGE_MAX


def test_starts_are_sorted_and_unique():
    """Blueprint §9.4 step 5: sort before serializing, so the manifest is order-stable."""
    s = seg(7)
    assert list(s.starts) == sorted(s.starts)
    assert len(set(s.starts)) == len(s.starts)


def test_every_sampled_window_fits_inside_the_segment():
    """A window may never cross a segment boundary (§9.1)."""
    end = T0 + timedelta(days=6)
    s = seg(6)
    for start in s.starts:
        assert start >= T0
        assert start + timedelta(hours=24) <= end


def test_at_least_one_window_when_any_is_legal():
    """max(1, ...) in §9.4 step 3: a low ratio must not silently drop a usable segment."""
    for i in range(100):
        s = sample_segment(f"ds/x{i}", 0, T0, T0 + timedelta(days=1), dataset_id="ds")
        assert s is not None and s.n_windows >= 1


def test_candidate_a_selects_far_more_than_candidate_b():
    """The 135x gap behind DECISIONS D005, in miniature."""
    a = seg(10, candidate=SamplerCandidate.LEGAL_START_FRACTION)
    b = seg(10, candidate=SamplerCandidate.UNION_TIMELINE)
    assert a.n_windows > 10 * b.n_windows


def test_stable_rng_depends_on_every_part():
    base = stable_rng(17, "a", "b").random()
    assert stable_rng(17, "a", "c").random() != base
    assert stable_rng(18, "a", "b").random() != base


def test_summarize_handles_empty_input():
    assert summarize([])["windows"] == 0
