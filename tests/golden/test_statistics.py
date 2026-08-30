"""Patch statistics, checked against the NumPy reference. Blueprint §11.2-§11.5."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from opencgm_stateevent.model import reference as ref
from opencgm_stateevent.model.statistics import (
    intra_patch_difference,
    patch_density,
    patch_mean_std,
    rate_of_change,
    roc_patch_mean_std,
)

L, P, K = 288, 24, 12


def window(seed: int, p_observed: float = 0.8):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=L), rng.random(L) < p_observed


def t(x, dtype=torch.float32):
    return torch.tensor(np.asarray(x), dtype=dtype).unsqueeze(0)


# --- parity with the reference -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("p", [1.0, 0.8, 0.33, 0.05])
def test_patch_mean_std_matches_reference(seed, p):
    values, mask = window(seed, p)
    mu, sd = patch_mean_std(t(values), t(mask, torch.bool))
    ref_mu, ref_sd = ref.patch_mean_std(values, mask)
    assert mu[0].numpy() == pytest.approx(ref_mu, abs=1e-5)
    assert sd[0].numpy() == pytest.approx(ref_sd, abs=1e-5)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("p", [1.0, 0.8, 0.33, 0.05])
def test_rate_of_change_matches_reference(seed, p):
    """The vectorised lag scan must equal the reference's explicit inner loop."""
    values, mask = window(seed, p)
    roc, valid = rate_of_change(t(values), t(mask, torch.bool))
    ref_roc, ref_valid = ref.rate_of_change(values, mask)
    assert valid[0].numpy().tolist() == ref_valid.tolist()
    assert roc[0].numpy() == pytest.approx(ref_roc, abs=1e-5)


@pytest.mark.parametrize("seed", range(4))
def test_patch_density_matches_reference(seed):
    _, mask = window(seed)
    d = patch_density(t(mask, torch.bool))
    assert d[0].numpy() == pytest.approx(ref.patch_density(mask), abs=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_intra_patch_difference_matches_reference(seed):
    values, mask = window(seed)
    diff, valid = intra_patch_difference(t(values), t(mask, torch.bool))
    ref_diff, ref_valid = ref.intra_patch_difference(values, mask)
    assert diff[0].numpy() == pytest.approx(ref_diff, abs=1e-5)
    assert valid[0].numpy().tolist() == ref_valid.tolist()


# --- properties --------------------------------------------------------------------------


def test_rate_of_change_looks_back_at_most_nine_steps():
    """Blueprint §11.4: the search horizon is nine grid steps, i.e. 45 minutes."""
    mask = np.zeros(L, dtype=bool)
    mask[0] = True
    mask[10] = True  # 10 steps later: beyond the horizon
    values = np.zeros(L)
    values[10] = 5.0
    _, valid = rate_of_change(t(values), t(mask, torch.bool))
    assert not bool(valid[0, 10])


def test_rate_of_change_divides_by_the_gap():
    mask = np.zeros(L, dtype=bool)
    mask[0] = mask[4] = True
    values = np.zeros(L)
    values[4] = 8.0
    roc, valid = rate_of_change(t(values), t(mask, torch.bool))
    assert bool(valid[0, 4])
    assert roc[0, 4].item() == pytest.approx(2.0)  # 8 over 4 steps


def test_rate_of_change_takes_the_nearest_predecessor():
    mask = np.zeros(L, dtype=bool)
    mask[0] = mask[3] = mask[5] = True
    values = np.zeros(L)
    values[3] = 3.0
    values[5] = 5.0
    roc, _ = rate_of_change(t(values), t(mask, torch.bool))
    assert roc[0, 5].item() == pytest.approx(1.0)  # (5-3)/2, not (5-0)/5


def test_first_observation_has_no_valid_rate_of_change():
    mask = np.zeros(L, dtype=bool)
    mask[0] = True
    _, valid = rate_of_change(t(np.zeros(L)), t(mask, torch.bool))
    assert not bool(valid[0, 0])


def test_roc_statistics_use_valid_entries_only():
    """A position can be observed but have no predecessor; counting its zero would bias
    the event statistics toward zero exactly where the event stream matters most."""
    roc = torch.zeros(1, L)
    valid = torch.zeros(1, L, dtype=torch.bool)
    roc[0, :6] = 2.0
    valid[0, :6] = True
    mu, sd = roc_patch_mean_std(roc, valid)
    assert mu[0, 0].item() == pytest.approx(2.0)  # not 6*2/12 = 1.0
    assert sd[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_empty_patches_are_zero_and_finite():
    values, mask = window(1)
    mask[:K] = False
    mu, sd = patch_mean_std(t(values), t(mask, torch.bool))
    assert mu[0, 0].item() == 0.0
    assert sd[0, 0].item() == 0.0
    assert torch.isfinite(mu).all() and torch.isfinite(sd).all()


def test_fully_unobserved_window_produces_no_nan():
    """A single NaN here would poison the whole batch through attention."""
    mask = np.zeros(L, dtype=bool)
    values = np.zeros(L)
    mu, sd = patch_mean_std(t(values), t(mask, torch.bool))
    roc, valid = rate_of_change(t(values), t(mask, torch.bool))
    rmu, rsd = roc_patch_mean_std(roc, valid)
    for x in (mu, sd, roc, rmu, rsd):
        assert torch.isfinite(x).all()
    assert not valid.any()


def test_intra_patch_difference_never_bridges_a_patch_boundary():
    """11 differences from 12 positions; patch i's first position has no predecessor."""
    values = np.arange(L, dtype=float)
    mask = np.ones(L, dtype=bool)
    diff, valid = intra_patch_difference(t(values), t(mask, torch.bool))
    assert diff.shape == (1, P, K - 1)
    assert valid.shape == (1, P, K - 1)
    assert torch.allclose(diff, torch.ones_like(diff))  # never 12, which would bridge


def test_shapes():
    values, mask = window(2)
    v, m = t(values), t(mask, torch.bool)
    assert patch_density(m).shape == (1, P)
    assert patch_mean_std(v, m)[0].shape == (1, P)
    assert rate_of_change(v, m)[0].shape == (1, L)
    assert intra_patch_difference(v, m)[0].shape == (1, P, K - 1)


def test_batch_rows_are_independent():
    v1, m1 = window(31)
    v2, m2 = window(32)
    v = torch.tensor(np.stack([v1, v2]), dtype=torch.float32)
    m = torch.tensor(np.stack([m1, m2]))
    both = rate_of_change(v, m)[0]
    alone = rate_of_change(v[:1], m[:1])[0]
    assert torch.allclose(both[0], alone[0], atol=1e-6)
