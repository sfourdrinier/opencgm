"""Differentiable causal Gaussian state/event decomposition. Blueprint §12.

The learnable bandwidth is the one part of the front end that trains. It is parameterised as
``sigma = 2 + 10 * sigmoid(rho)`` so it cannot leave the paper's [2, 12] range no matter what
the optimizer does — a hard constraint rather than a penalty that can be overwhelmed.

Kernel and denominator are computed in float32 even under autocast (§12.2). The denominator is
a masked sum that can be small; computing it in half precision would quietly produce infinities
in sparsely observed patches, which is exactly where 15-minute sources live.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .reference import EPS, RADIUS, SCALE_MIN, SIGMA_INIT, SIGMA_MAX, SIGMA_MIN, rho_from_sigma


def masked_instance_norm(
    values: Tensor, mask: Tensor, *, eps: float = EPS, scale_min: float = SCALE_MIN
) -> Tensor:
    """Observed-only per-window instance normalization. Blueprint §11.1.

    ``values`` and ``mask`` are ``[B, L]``. Statistics use observed positions only and the
    output is zeroed where the mask is false, so no fill value can reach a downstream module.
    All-unobserved rows return zeros rather than NaN.
    """
    m = mask.to(values.dtype)
    n = m.sum(dim=-1, keepdim=True)
    mu = (m * values).sum(dim=-1, keepdim=True) / (n + eps)
    var = (m * (values - mu) ** 2).sum(dim=-1, keepdim=True) / (n + eps)
    scale = torch.sqrt(var + eps).clamp_min(scale_min)
    out = m * (values - mu) / scale
    return torch.where(n > 0, out, torch.zeros_like(out))


class CausalGaussian(nn.Module):
    """One-sided mask-aware Gaussian smoother with a learnable bandwidth.

    Implemented as a convolution over lags 0..R with left padding only, which is what makes it
    strictly causal: position j sees j-R..j and nothing later. `tests/golden/test_gaussian.py`
    proves an impulse at j+1 cannot move the output at j.
    """

    def __init__(
        self,
        *,
        sigma_init: float = SIGMA_INIT,
        radius: int = RADIUS,
        eps: float = EPS,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.eps = eps
        rho0 = torch.tensor(rho_from_sigma(sigma_init), dtype=torch.float32)
        self.rho = nn.Parameter(rho0, requires_grad=learnable)

    @property
    def sigma(self) -> Tensor:
        """sigma = 2 + 10 * sigmoid(rho). Structurally bounded to [2, 12]."""
        return SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * torch.sigmoid(self.rho)

    def kernel(self) -> Tensor:
        r = torch.arange(self.radius + 1, dtype=torch.float32, device=self.rho.device)
        w = torch.exp(-(r**2) / (2.0 * self.sigma.float() ** 2))
        return w / w.sum()

    def forward(self, normalized: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return (state, event) for ``[B, L]`` inputs.

        Event is the masked residual ``(X - State) * M`` (§12.1).
        """
        with torch.autocast(device_type=normalized.device.type, enabled=False):
            x = normalized.float()
            m = mask.float()
            k = self.kernel().flip(0).view(1, 1, -1)  # conv1d correlates; flip for lag order

            pad = (self.radius, 0)  # left-only padding keeps the filter causal
            num = F.conv1d(F.pad((m * x).unsqueeze(1), pad), k).squeeze(1)
            den = F.conv1d(F.pad(m.unsqueeze(1), pad), k).squeeze(1)
            state = num / (den + self.eps)

            event = (x - state) * m
        return state.to(normalized.dtype), event.to(normalized.dtype)


def decompose(
    values: Tensor, mask: Tensor, gaussian: CausalGaussian
) -> tuple[Tensor, Tensor, Tensor]:
    """normalize -> smooth -> residual. Returns (normalized, state, event)."""
    x = masked_instance_norm(values, mask)
    state, event = gaussian(x, mask)
    return x, state, event
