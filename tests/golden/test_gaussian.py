"""Causal Gaussian golden tests. Blueprint §12.3.

Eight required properties, plus numerical parity against the independently written NumPy
reference. Parity is the important one: the reference was written from the equations in
float64 with explicit loops, the module was written vectorised in PyTorch, and if the two
agree to 1e-6 on random masked input then both readings of §12.1 coincide.

The causality test is the one that would otherwise fail silently. A non-causal filter trains
perfectly well and produces a model that cannot be deployed on a live stream.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from opencgm_stateevent.model import reference as ref
from opencgm_stateevent.model.causal_gaussian import (
    CausalGaussian,
    masked_instance_norm,
)

L = 288
TOL = 1e-6


def random_window(seed: int, p_observed: float = 0.8):
    rng = np.random.default_rng(seed)
    values = rng.normal(120.0, 30.0, size=L)
    mask = rng.random(L) < p_observed
    return values, mask


def as_batch(values, mask):
    return (
        torch.tensor(values, dtype=torch.float32).unsqueeze(0),
        torch.tensor(mask, dtype=torch.bool).unsqueeze(0),
    )


# --- parity with the NumPy reference -----------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_normalization_matches_reference(seed):
    values, mask = random_window(seed)
    got = masked_instance_norm(*as_batch(values, mask))[0].detach().numpy()
    assert got == pytest.approx(ref.normalize(values, mask), abs=1e-5)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("p_observed", [1.0, 0.8, 0.33, 0.1])
def test_state_matches_reference_under_random_masks(seed, p_observed):
    """Blueprint §12.3: random masks must match a high-precision reference."""
    values, mask = random_window(seed, p_observed)
    x = ref.normalize(values, mask)
    expected = ref.causal_gaussian_state(x, mask, ref.SIGMA_INIT)

    g = CausalGaussian()
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    mt = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    state, _ = g(xt, mt)
    assert state[0].detach().numpy() == pytest.approx(expected, abs=1e-5)


def test_event_is_the_masked_residual():
    values, mask = random_window(3)
    x = ref.normalize(values, mask)
    _, _, expected = ref.decompose(values, mask)
    g = CausalGaussian()
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    mt = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    _, event = g(xt, mt)
    assert event[0].detach().numpy() == pytest.approx(expected, abs=1e-5)
    assert (event[0].detach().numpy()[~mask] == 0).all()


# --- the eight required properties, §12.3 -------------------------------------------------


def test_impulse_at_t_plus_one_cannot_affect_state_at_t():
    """Strict causality. A non-causal filter trains fine and cannot be deployed live."""
    g = CausalGaussian()
    mask = torch.ones(1, L, dtype=torch.bool)
    base = torch.zeros(1, L)
    bumped = base.clone()
    bumped[0, 100] = 1000.0
    s_base, _ = g(base, mask)
    s_bumped, _ = g(bumped, mask)
    assert torch.allclose(s_base[0, :100], s_bumped[0, :100], atol=1e-6)
    assert not torch.allclose(s_base[0, 100], s_bumped[0, 100])


def test_all_observed_matches_a_direct_loop():
    values, mask = random_window(11, p_observed=1.0)
    x = ref.normalize(values, mask)
    expected = ref.causal_gaussian_state(x, mask, ref.SIGMA_INIT)
    g = CausalGaussian()
    state, _ = g(
        torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.ones(1, L, dtype=torch.bool)
    )
    assert state[0].detach().numpy() == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("fill", [0.0, -999.0, 1e6])
def test_fill_value_invariance(fill):
    """Blueprint §10.2/§12.3: the number sitting in unobserved slots cannot change any output."""
    values, mask = random_window(5, p_observed=0.5)
    a = values.copy()
    a[~mask] = 0.0
    b = values.copy()
    b[~mask] = fill
    g = CausalGaussian()
    sa, ea = g(*as_batch(ref.normalize(a, mask), mask))
    sb, eb = g(*as_batch(ref.normalize(b, mask), mask))
    assert torch.allclose(sa, sb, atol=1e-5)
    assert torch.allclose(ea, eb, atol=1e-5)


def test_sigma_initialises_to_six():
    assert CausalGaussian().sigma.item() == pytest.approx(6.0, abs=1e-5)
    assert ref.sigma_from_rho(ref.rho_from_sigma(6.0)) == pytest.approx(6.0)


def test_gradients_reach_rho():
    g = CausalGaussian()
    values, mask = random_window(2)
    x = torch.tensor(ref.normalize(values, mask), dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    state, _ = g(x, m)
    state.sum().backward()
    assert g.rho.grad is not None
    assert torch.isfinite(g.rho.grad).all()
    assert g.rho.grad.abs().item() > 0


@pytest.mark.parametrize("rho", [-50.0, -10.0, 0.0, 10.0, 50.0])
def test_sigma_never_escapes_its_bounds(rho):
    """Structural, not a penalty: no optimizer step can push sigma outside [2, 12]."""
    g = CausalGaussian()
    with torch.no_grad():
        g.rho.fill_(rho)
    assert 2.0 <= g.sigma.item() <= 12.0


def test_zero_support_positions_are_finite_zero():
    """Where nothing has been observed yet, output is exactly 0 rather than NaN."""
    g = CausalGaussian()
    mask = torch.zeros(1, L, dtype=torch.bool)
    mask[0, 200:] = True
    x = torch.zeros(1, L)
    x[0, 200:] = 1.0
    state, _ = g(x, mask)
    assert torch.isfinite(state).all()
    assert torch.allclose(state[0, :150], torch.zeros(150), atol=1e-6)


def test_all_unobserved_window_returns_zeros_not_nan():
    g = CausalGaussian()
    x = masked_instance_norm(torch.randn(1, L), torch.zeros(1, L, dtype=torch.bool))
    assert torch.isfinite(x).all()
    state, event = g(x, torch.zeros(1, L, dtype=torch.bool))
    assert torch.isfinite(state).all() and torch.isfinite(event).all()


# --- behaviour ---------------------------------------------------------------------------


def test_kernel_is_normalised_and_monotonically_decreasing():
    k = CausalGaussian().kernel()
    assert k.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (k[:-1] >= k[1:]).all()


def test_larger_sigma_smooths_more():
    """Sanity: the bandwidth must actually control smoothing, or learning it is meaningless."""
    values, mask = random_window(9, p_observed=1.0)
    x = torch.tensor(ref.normalize(values, mask), dtype=torch.float32).unsqueeze(0)
    m = torch.ones(1, L, dtype=torch.bool)
    variances = []
    for sigma in (2.5, 6.0, 11.5):
        g = CausalGaussian(sigma_init=sigma)
        state, _ = g(x, m)
        variances.append(state.var().item())
    assert variances[0] > variances[1] > variances[2]


def test_batch_rows_are_independent():
    g = CausalGaussian()
    v1, m1 = random_window(21)
    v2, m2 = random_window(22)
    x = torch.tensor(np.stack([ref.normalize(v1, m1), ref.normalize(v2, m2)]), dtype=torch.float32)
    m = torch.tensor(np.stack([m1, m2]))
    both, _ = g(x, m)
    alone, _ = g(x[:1], m[:1])
    assert torch.allclose(both[0], alone[0], atol=1e-6)
