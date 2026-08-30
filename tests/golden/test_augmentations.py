"""Augmentation invariants. Blueprint §16.6.

The structural rule is the one that protects the mask contract: an augmentation may remove
observations, never invent them. If that ever breaks, the model trains on fabricated readings
and the physical mask stops meaning what the whole method assumes it means.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencgm_stateevent.data.augmentations import (
    DECIMATION_MIN_OBSERVED_EXCLUSIVE,
    OPERATIONS,
    STRUCTURAL_OPERATIONS,
    VALUE_OPERATIONS,
    augment,
    baseline_wander,
    compression_drop,
    decimation,
    disconnection,
)
from opencgm_stateevent.data.timestamps import SEQUENCE_LENGTH


def full_window(value: float = 120.0):
    return (
        np.full(SEQUENCE_LENGTH, value, dtype=np.float32),
        np.ones(SEQUENCE_LENGTH, dtype=bool),
    )


def rng(seed: int = 0):
    return np.random.default_rng(seed)


# --- determinism, blueprint §16.6 --------------------------------------------------------


def test_augmentation_is_deterministic_for_a_given_seed():
    v, m = full_window()
    a = augment(v, m, rng(7))
    b = augment(v, m, rng(7))
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.mask, b.mask)
    assert a.applied == b.applied


def test_input_arrays_are_never_mutated():
    """Canonical windows are cached and reused every epoch; mutation would corrupt the corpus."""
    v, m = full_window()
    v_ref, m_ref = v.copy(), m.copy()
    for seed in range(30):
        augment(v, m, rng(seed))
    assert np.array_equal(v, v_ref)
    assert np.array_equal(m, m_ref)


# --- the mask contract -------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_no_augmentation_ever_adds_an_observation(seed):
    """The invariant the whole mask-aware method rests on."""
    v, m = full_window()
    m[::7] = False
    before = int(m.sum())
    out = augment(v, m, rng(seed))
    assert int(out.mask.sum()) <= before
    assert not (out.mask & ~m).any()


@pytest.mark.parametrize("op", [baseline_wander, compression_drop])
def test_value_operations_leave_the_mask_untouched(op):
    v, m = full_window()
    m[10:20] = False
    _, out_mask = op(v, m, rng(3))
    assert np.array_equal(out_mask, m)


@pytest.mark.parametrize("op", [decimation, disconnection])
def test_structural_operations_leave_values_untouched(op):
    v, m = full_window()
    out_values, _ = op(v, m, rng(3))
    assert np.array_equal(out_values, v)


def test_operation_sets_partition_the_four_operations():
    assert {n for n, _, _ in OPERATIONS} == VALUE_OPERATIONS | STRUCTURAL_OPERATIONS
    assert not (VALUE_OPERATIONS & STRUCTURAL_OPERATIONS)


# --- individual operations ---------------------------------------------------------------


def test_baseline_wander_only_touches_observed_positions():
    v, m = full_window()
    m[100:150] = False
    out, _ = baseline_wander(v, m, rng(1))
    assert np.array_equal(out[~m], v[~m])
    assert not np.array_equal(out[m], v[m])


def test_baseline_wander_amplitude_stays_in_the_paper_range():
    """5-15 mg/dL is PAPER_EXACT; only the phase is inferred."""
    v, m = full_window()
    for seed in range(60):
        out, _ = baseline_wander(v, m, rng(seed))
        assert np.abs(out - v).max() <= 15.0 + 1e-4


def test_compression_drop_attenuates_a_contiguous_span():
    v, m = full_window()
    out, _ = compression_drop(v, m, rng(5))
    changed = np.flatnonzero(~np.isclose(out, v))
    assert 6 <= len(changed) <= 12
    assert np.array_equal(changed, np.arange(changed[0], changed[-1] + 1))
    assert (out[changed] < v[changed]).all()


def test_compression_drop_never_falls_below_the_minimum_multiplier():
    v, m = full_window()
    for seed in range(40):
        out, _ = compression_drop(v, m, rng(seed))
        assert (out >= v * 0.4 - 1e-4).all()


def test_decimation_is_skipped_at_or_below_200_observations():
    """Blueprint §16.4: applies only when observed count is greater than 200."""
    v, m = full_window()
    m[:] = False
    m[:DECIMATION_MIN_OBSERVED_EXCLUSIVE] = True
    _, out_mask = decimation(v, m, rng(0))
    assert np.array_equal(out_mask, m)


def test_decimation_keeps_every_third_absolute_index():
    """By absolute grid index, not by position in a compressed observation list (§16.4)."""
    v, m = full_window()
    _, out_mask = decimation(v, m, rng(2))
    kept = np.flatnonzero(out_mask)
    assert len(np.unique(kept % 3)) == 1
    assert len(kept) == 96


def test_decimation_of_a_sparse_source_keeps_only_already_observed_positions():
    v, m = full_window()
    m[1::2] = False  # 144 observed, below threshold -> no-op
    _, out_mask = decimation(v, m, rng(0))
    assert np.array_equal(out_mask, m)


@pytest.mark.parametrize("seed", range(30))
def test_disconnection_removes_between_2_and_36_positions(seed):
    """1-3 blocks of 2-12 positions; overlap allowed, so the total can be lower."""
    v, m = full_window()
    _, out_mask = disconnection(v, m, rng(seed))
    removed = int(m.sum()) - int(out_mask.sum())
    assert 2 <= removed <= 36


@pytest.mark.parametrize("seed", range(30))
def test_all_blocks_stay_inside_the_grid(seed):
    v, m = full_window()
    _, out_mask = disconnection(v, m, rng(seed))
    assert out_mask.shape == (SEQUENCE_LENGTH,)


# --- ordering and probability decay, blueprint §16.1 -------------------------------------


def test_probability_decay_compounds_across_applications():
    """Monte Carlo. With compounding decay the mean number applied stays well below the sum
    of base probabilities (0.80), because each success suppresses the rest."""
    v, m = full_window()
    counts = [len(augment(v, m, rng(s)).applied) for s in range(3000)]
    mean = float(np.mean(counts))
    assert 0.3 < mean < 0.8
    assert max(counts) <= len(OPERATIONS)


def test_operations_are_evaluated_in_random_order():
    """A fixed order would correlate which operation wins with its base probability alone."""
    v, m = full_window()
    firsts = {augment(v, m, rng(s)).applied[0] for s in range(400) if augment(v, m, rng(s)).applied}
    assert len(firsts) >= 3
