"""Predictor, transition heads, and the EMA target. Blueprint §14.

The transition heads take **pre-Transformer** state and event tokens and predict the EMA
target's pre-Transformer tokens at the next patch (§14.3). Using post-Transformer tokens would
let contextual attention over all 24 patches leak future information into a target that is
supposed to test one-step dynamics, which would make the temporal loss trivially satisfiable.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

from .encoder import ContextEncoder, transformer
from .stream_embedder import EVENT_TOKEN, FUSED_DIM, STATE_TOKEN

TRANSITION_HIDDEN = 256  # INFERRED_RECONSTRUCTION §14.3, kept configurable for §13.7 parity
EMA_INITIAL_MOMENTUM = 0.997  # PAPER_EXACT §14.1
EMA_FINAL_MOMENTUM = 1.0  # PAPER_EXACT §14.1


class Predictor(nn.Module):
    """One Transformer layer over the 24 contextual tokens.

    Blueprint §14.2; the single layer is PAPER_EXACT.
    """

    def __init__(self, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.transformer = transformer(1, dropout=dropout)

    def forward(self, contextual: Tensor) -> Tensor:
        return self.transformer(contextual)


class TransitionHead(nn.Module):
    """Residual next-patch prediction. Blueprint §14.3.

        S_hat_{i+1} = S_i + g([S_i, E_i, tau_i])

    Residual form means the head starts near the identity, so at initialisation the prediction
    is "the next patch looks like this one" — a sensible prior for a physiological state that
    changes slowly, and the reason the temporal loss does not dominate early training.
    """

    def __init__(self, out_dim: int, *, hidden: int = TRANSITION_HIDDEN) -> None:
        super().__init__()
        in_dim = STATE_TOKEN + EVENT_TOKEN + FUSED_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, own: Tensor, other: Tensor, temporal: Tensor) -> Tensor:
        return own + self.net(torch.cat([own, other, temporal], dim=-1))


class EMATarget(nn.Module):
    """Frozen exponential-moving-average copy of the online encoder. Blueprint §14.1.

    The copy spans the whole encoder path including the Gaussian bandwidth rho, the stream
    embedders and the fusion/time modules — not just the Transformer. The target must produce
    targets through the same transformation the online branch is learning, or the two branches
    are comparing different quantities.
    """

    def __init__(self, online: ContextEncoder) -> None:
        super().__init__()
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @staticmethod
    def momentum(global_step: int, max_steps: int) -> float:
        """Cosine schedule from 0.997 to 1.0 over training. Blueprint §14.1 — PAPER_EXACT."""
        progress = min(1.0, global_step / max(1, max_steps))
        return 1.0 - (1.0 - EMA_INITIAL_MOMENTUM) * (math.cos(math.pi * progress) + 1.0) / 2.0

    @torch.no_grad()
    def update(self, online: ContextEncoder, m: float) -> None:
        """theta_target <- m * theta_target + (1-m) * theta_online, after the optimizer step."""
        for tp, op in zip(self.target.parameters(), online.parameters(), strict=True):
            tp.mul_(m).add_(op.detach(), alpha=1.0 - m)
        for tb, ob in zip(self.target.buffers(), online.buffers(), strict=True):
            tb.copy_(ob)

    def train(self, mode: bool = True):
        """Keep the target in eval mode always.

        `torch.no_grad()` stops gradients but does not disable dropout, so a target registered as
        an ordinary child module inherits `model.train()` and becomes *stochastic*. Two identical
        forward passes then differ by an RMS of about 0.44, which is large next to the losses
        being optimised: the regression target is noise, and the online branch is rewarded for
        predicting the mean of that noise rather than the teacher's representation.

        No teacher-student implementation does this. DINO builds stochastic depth on the student
        only; I-JEPA's target encoder is never put in train mode.
        """
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)
