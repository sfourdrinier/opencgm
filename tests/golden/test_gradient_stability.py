"""Gradient stability under fully unobserved patches.

Regression tests for a bug the unit suite did not catch. `torch.sqrt(0)` has an infinite
derivative, and `torch.where` differentiates the branch it does not select, so

    torch.where(has_observations, torch.sqrt(var), 0.0)

produces ``0 * inf = NaN`` in the backward pass whenever a patch has zero variance. Forward
values were correct, so every value-parity test passed. Only rho's gradient went NaN, and only
on real data, because zero-variance patches require a patch to be entirely unobserved — which
is exactly what JEPA masking does to the online branch on every step.

It surfaced in the 32-window overfit gate, not in 322 unit tests. That is the argument for
running the gate.
"""

from __future__ import annotations

import pytest
import torch

from opencgm_stateevent.model.model import OpenCGMStateEvent
from opencgm_stateevent.model.reference import PATCHES, STEPS_PER_PATCH
from opencgm_stateevent.model.statistics import (
    _safe_sqrt,
    patch_mean_std,
    roc_patch_mean_std,
)

L = 288
B = 4


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --- the primitive ------------------------------------------------------------------------


def test_safe_sqrt_is_zero_at_zero():
    var = torch.tensor([0.0, 4.0, 9.0])
    assert torch.allclose(_safe_sqrt(var), torch.tensor([0.0, 2.0, 3.0]))


def test_safe_sqrt_has_a_finite_gradient_at_zero():
    var = torch.tensor([0.0, 4.0], requires_grad=True)
    _safe_sqrt(var).sum().backward()
    assert torch.isfinite(var.grad).all()
    assert var.grad[0].item() == 0.0


def test_naive_sqrt_would_produce_nan():
    """Pins the failure mode, so the fix cannot be reverted silently."""
    var = torch.tensor([0.0, 4.0], requires_grad=True)
    torch.where(var > 0, torch.sqrt(var), torch.zeros_like(var)).sum().backward()
    assert torch.isnan(var.grad).any()


# --- the statistics -----------------------------------------------------------------------


def test_patch_mean_std_gradient_survives_an_empty_patch():
    signal = torch.randn(B, L, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, :STEPS_PER_PATCH] = False  # patch 0 fully unobserved
    mu, sd = patch_mean_std(signal, mask)
    (mu.sum() + sd.sum()).backward()
    assert torch.isfinite(signal.grad).all()


def test_patch_mean_std_gradient_survives_a_single_observation():
    """One observation gives near-zero variance, which also lands on the sqrt(0) cliff.

    Not *exactly* zero: blueprint §11.3 places epsilon in the denominator, so with n=1 the mean
    differs from the sample by about 1e-6 and the std is that order rather than 0. Faithful to
    the equation; what matters here is that the gradient stays finite.
    """
    signal = torch.randn(B, L, requires_grad=True)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, 0] = True
    _, sd = patch_mean_std(signal, mask)
    sd.sum().backward()
    assert torch.isfinite(signal.grad).all()
    assert sd[0, 0].item() == pytest.approx(0.0, abs=1e-4)


def test_roc_patch_statistics_gradient_is_finite_with_no_valid_entries():
    roc = torch.randn(B, L, requires_grad=True)
    valid = torch.zeros(B, L, dtype=torch.bool)
    mu, sd = roc_patch_mean_std(roc, valid)
    (mu.sum() + sd.sum()).backward()
    assert torch.isfinite(roc.grad).all()


# --- end to end ---------------------------------------------------------------------------


@pytest.mark.parametrize("p_observed", [1.0, 0.5, 0.33, 0.1])
def test_no_parameter_receives_a_nonfinite_gradient(p_observed):
    """The check that would have caught the bug: every parameter, on masked real-shaped data."""
    torch.manual_seed(0)
    values = torch.randn(B, L) * 30 + 120
    mask = torch.rand(B, L) < p_observed
    model = OpenCGMStateEvent()
    model(values, mask, torch.zeros(B, dtype=torch.long), gen(1)).loss.backward()
    bad = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite gradients in {bad}"


def test_rho_gradient_is_finite_when_whole_patches_are_hidden():
    """The exact configuration that failed: JEPA masking empties entire patches."""
    torch.manual_seed(0)
    values = torch.randn(B, L) * 30 + 120
    mask = torch.ones(B, L, dtype=torch.bool)
    model = OpenCGMStateEvent()
    model(values, mask, torch.zeros(B, dtype=torch.long), gen(2)).loss.backward()
    rho = model.online.gaussian.rho
    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()


def test_a_fully_unobserved_window_does_not_break_the_backward_pass():
    torch.manual_seed(0)
    values = torch.zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    model = OpenCGMStateEvent()
    out = model(values, mask, torch.zeros(B, dtype=torch.long), gen(3))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    bad = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad


def test_every_patch_count_from_the_mask_sampler_keeps_gradients_finite():
    """12-14 patches are hidden per sample; all must be safe."""
    torch.manual_seed(0)
    for seed in range(8):
        values = torch.randn(B, L) * 30 + 120
        mask = torch.rand(B, L) < 0.8
        model = OpenCGMStateEvent()
        model(values, mask, torch.zeros(B, dtype=torch.long), gen(seed)).loss.backward()
        assert torch.isfinite(model.online.gaussian.rho.grad).all()


def test_patch_statistics_shapes_unchanged_by_the_fix():
    signal = torch.randn(B, L)
    mask = torch.rand(B, L) < 0.7
    mu, sd = patch_mean_std(signal, mask)
    assert mu.shape == sd.shape == (B, PATCHES)
