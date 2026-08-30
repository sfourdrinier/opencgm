"""Pretraining objectives. Blueprint §15.

Both losses are density-weighted, and the weighting is what makes the mask-aware design pay:
a patch observed twice in twelve positions contributes a sixth as much as a fully observed one,
so sparse 15-minute sources cannot dominate the gradient simply by being present.

Denominators can legitimately be zero — a batch where no patch is both context-masked and
observed. Every loss returns a differentiable zero in that case rather than NaN, because a
single NaN would propagate through the optimizer and silently destroy the run.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-6
SMOOTH_L1_BETA = 1.0  # INFERRED_RECONSTRUCTION §15.2


def _smooth_l1_per_token(pred: Tensor, target: Tensor, beta: float = SMOOTH_L1_BETA) -> Tensor:
    """SmoothL1 averaged over the latent dimension. ``[B,P,D]`` -> ``[B,P]``."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="none").mean(dim=-1)


def masked_contextual_loss(
    predicted: Tensor,
    target: Tensor,
    context_mask: Tensor,
    density: Tensor,
    beta: float = SMOOTH_L1_BETA,
) -> Tensor:
    """L_MCR. Blueprint §15.2 — PAPER_EXACT.

        sum_i d_i c_i SmoothL1(Z_pred_i, Z_target_i) / (sum_i d_i c_i + eps)

    ``target`` must already be detached by the caller; we detach again so a mistake upstream
    cannot quietly send gradients into the EMA branch.
    """
    per_patch = _smooth_l1_per_token(predicted, target.detach(), beta)
    w = density * context_mask.to(density.dtype)
    denom = w.sum()
    if denom.item() == 0.0:
        return (predicted * 0.0).sum()
    return (w * per_patch).sum() / (denom + EPS)


def transition_weight(context_mask: Tensor, density: Tensor) -> Tensor:
    """q_i = (1 - c_i) * d_i * d_{i+1}. Blueprint §15.3 — PAPER_EXACT.

    Excludes transitions whose *starting* patch is hidden from the online branch, and
    down-weights sparse neighbours. The paper's equation has no destination-mask term and we do
    not add one in the strict path.
    """
    visible = 1.0 - context_mask.to(density.dtype)
    return visible[:, :-1] * density[:, :-1] * density[:, 1:]


def temporal_dynamics_loss(
    state_pred: Tensor,
    state_target: Tensor,
    event_pred: Tensor,
    event_target: Tensor,
    weight: Tensor,
    beta: float = SMOOTH_L1_BETA,
) -> Tensor:
    """L_TD. Blueprint §15.4 — PAPER_EXACT.

    Half the sum of the separately normalised state and event terms, so a stream with more
    valid transitions cannot dominate the other.
    """
    denom = weight.sum()
    if denom.item() == 0.0:
        return (state_pred * 0.0).sum()
    s = (weight * _smooth_l1_per_token(state_pred, state_target.detach(), beta)).sum()
    e = (weight * _smooth_l1_per_token(event_pred, event_target.detach(), beta)).sum()
    return 0.5 * (s / (denom + EPS) + e / (denom + EPS))


def total_loss(mcr: Tensor, td: Tensor, lambda_mcr: float = 1.0, lambda_td: float = 1.0) -> Tensor:
    """L = lambda_MCR * L_MCR + lambda_TD * L_TD. Blueprint §15.5 — both weights PAPER_EXACT 1.0."""
    return lambda_mcr * mcr + lambda_td * td


def sample_context_mask(
    batch: int,
    patches: int,
    generator: torch.Generator,
    *,
    ratio_min: float = 0.50,
    ratio_max: float = 0.60,
    device: torch.device | None = None,
) -> Tensor:
    """Per-sample JEPA patch mask. Blueprint §15.1 — ratio range PAPER_EXACT.

    The integer rule is ``round(ratio * 24)`` (INFERRED_RECONSTRUCTION), clamped to keep at
    least one visible and one masked patch so neither branch is ever degenerate.

    Sampling always happens on the generator's own device and is transferred afterwards, so a
    given seed produces identical masks whether training runs on CPU or GPU. Generating
    directly on CUDA would make runs unreproducible across devices.
    """
    gen_device = generator.device
    ratios = (
        torch.rand(batch, generator=generator, device=gen_device) * (ratio_max - ratio_min)
        + ratio_min
    )
    counts = torch.round(ratios * patches).long().clamp(1, patches - 1)
    scores = torch.rand(batch, patches, generator=generator, device=gen_device)
    order = scores.argsort(dim=1)
    ranks = order.argsort(dim=1)
    out = ranks < counts.unsqueeze(1)
    return out.to(device) if device is not None else out


def apply_context_mask(
    mask: Tensor, context_patch_mask: Tensor, steps_per_patch: int = 12
) -> Tensor:
    """Remove hidden patches from the branch-visible mask. Blueprint §10.1.

    This is the single operation that enforces the no-leakage ordering. The returned mask is
    what the online branch normalises, filters and computes statistics from, so hidden patches
    cannot influence any of them.
    """
    expanded = context_patch_mask.repeat_interleave(steps_per_patch, dim=1)
    return mask & ~expanded
