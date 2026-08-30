"""Patch-level statistics fed to the stream embedders. Blueprint §11.2-§11.5.

Everything here is observed-only. Empty patches produce zeros and carry their validity in the
density, rather than propagating NaN into the encoder — a single NaN here would poison the
whole batch through attention, and sparsely observed patches are common on 15-minute sources.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .reference import EPS, PATCHES, ROC_MAX_BACK, STEPS_PER_PATCH


def patch_density(mask: Tensor) -> Tensor:
    """d_i = (1/12) sum_j M_j over patch i. Blueprint §11.2 — PAPER_EXACT. [B,L] -> [B,24]."""
    b = mask.shape[0]
    return mask.float().reshape(b, PATCHES, STEPS_PER_PATCH).mean(dim=-1)


def _safe_sqrt(var: Tensor) -> Tensor:
    """sqrt that is zero, and differentiable, where the variance is zero.

    ``d/dx sqrt(x) = 1/(2 sqrt(x))`` is infinite at zero, and ``torch.where`` differentiates
    the branch it does not select, so a plain ``where(has, sqrt(var), 0)`` yields ``0 * inf =
    NaN`` in the backward pass. Substituting 1.0 *inside* the sqrt keeps every value identical
    while giving the untaken branch a finite gradient.

    This fires whenever a patch has zero or one observation: zero-variance patches are common
    in the online branch, where whole patches are hidden by JEPA masking. It produced a NaN in
    rho on the very first optimizer step and was invisible to unit tests using random masks.
    """
    positive = var > 0
    safe = torch.where(positive, var, torch.ones_like(var))
    return torch.where(positive, torch.sqrt(safe), torch.zeros_like(var))


def patch_mean_std(signal: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Observed-only mean and std per one-hour patch. Blueprint §11.3 — PAPER_EXACT.

    [B,L] -> two [B,24]. Empty patches give (0, 0).
    """
    b = signal.shape[0]
    s = signal.reshape(b, PATCHES, STEPS_PER_PATCH)
    m = mask.float().reshape(b, PATCHES, STEPS_PER_PATCH)
    n = m.sum(dim=-1)
    mu = (m * s).sum(dim=-1) / (n + EPS)
    var = (m * (s - mu.unsqueeze(-1)) ** 2).sum(dim=-1) / (n + EPS)
    has = n > 0
    return torch.where(has, mu, 0.0), torch.where(has, _safe_sqrt(var), 0.0)


def rate_of_change(
    normalized: Tensor, mask: Tensor, max_back: int = ROC_MAX_BACK
) -> tuple[Tensor, Tensor]:
    """Backward rate of change to the nearest observed predecessor. Blueprint §11.4.

        b = min{ k in [1,9] : M_{j-k} = 1 },   r_j = (X_j - X_{j-b}) / b

    Units are normalized-glucose change per five-minute grid step, matching the paper equation.
    Vectorised by scanning candidate lags in order and taking the first hit, which is
    equivalent to the reference's inner loop but avoids a Python loop over positions.

    Returns (roc, valid), both [B,L]. Invalid positions carry r = 0.
    """
    m = mask.bool()
    roc = torch.zeros_like(normalized)
    found = torch.zeros_like(m)

    for lag in range(1, max_back + 1):
        # shift right by `lag`; positions before `lag` have no predecessor at this distance
        prev_val = torch.zeros_like(normalized)
        prev_obs = torch.zeros_like(m)
        prev_val[:, lag:] = normalized[:, :-lag]
        prev_obs[:, lag:] = m[:, :-lag]

        take = m & prev_obs & ~found
        if not take.any():
            continue
        roc = torch.where(take, (normalized - prev_val) / lag, roc)
        found = found | take

    return roc, found


def masked_mean(values: Tensor, valid: Tensor, dim: int = -1) -> Tensor:
    """Mean over valid entries only; zero where nothing is valid."""
    v = valid.float()
    n = v.sum(dim=dim)
    total = (values * v).sum(dim=dim)
    return torch.where(n > 0, total / (n + EPS), torch.zeros_like(total))


def roc_patch_mean_std(roc: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Event-stream statistics over **valid rate-of-change entries only**. Blueprint §11.4.

    Distinct from `patch_mean_std`: a position can be observed yet have no valid predecessor
    within nine steps, and including its zero would bias the event statistics toward zero
    exactly in the sparse patches where the event stream matters most.
    """
    b = roc.shape[0]
    r = roc.reshape(b, PATCHES, STEPS_PER_PATCH)
    v = valid.float().reshape(b, PATCHES, STEPS_PER_PATCH)
    n = v.sum(dim=-1)
    mu = (v * r).sum(dim=-1) / (n + EPS)
    var = (v * (r - mu.unsqueeze(-1)) ** 2).sum(dim=-1) / (n + EPS)
    has = n > 0
    return torch.where(has, mu, 0.0), torch.where(has, _safe_sqrt(var), 0.0)


def intra_patch_difference(state: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Adjacent state differences inside each patch. Blueprint §11.5.

    INFERRED_RECONSTRUCTION. The paper gives a 16-dimensional trend-difference feature and a
    patch-level Diff path but never says which sequence it differences; we difference the state
    stream. Patch boundaries are never bridged, so each patch yields 11 differences from its 12
    positions. Returns [B,24,11] and its validity.
    """
    b = state.shape[0]
    s = state.reshape(b, PATCHES, STEPS_PER_PATCH)
    m = mask.bool().reshape(b, PATCHES, STEPS_PER_PATCH)
    diff = s[..., 1:] - s[..., :-1]
    valid = m[..., 1:] & m[..., :-1]
    return diff * valid.float(), valid
