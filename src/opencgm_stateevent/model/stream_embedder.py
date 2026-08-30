"""State and event stream embedders. Blueprint §13.

The feature widths are PAPER_EXACT — state 64+16+48 and event 48+48+32, both concatenating to
128 and projecting to a 64-dimensional token. The internals are not: kernel sizes, MLP depth,
activations and pooling are Reference Reconstruction A from §13.2, a defensible default rather
than a claim about the authors' code. Every one is configurable so the ablations in §20 Tier 2
can move them without touching this file.

Masked mean pooling is what keeps the fill value out. A convolution with bias produces a
non-zero output even from an all-zero input, so empty patches are explicitly zeroed after the
convolution rather than trusted to come out clean (§13.3).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .reference import PATCHES, STEPS_PER_PATCH

# PAPER_EXACT feature widths, §13.1
STATE_WAVEFORM = 64
STATE_DIFF = 16
STATE_STATS = 48
STATE_TOKEN = 64
EVENT_RESIDUAL = 48
EVENT_ROC = 48
EVENT_STATS = 32
EVENT_TOKEN = 64
FUSED_DIM = 128


def masked_mean_pool(features: Tensor, mask: Tensor) -> Tensor:
    """Mean over observed positions within each patch. ``[N,C,K]`` and ``[N,K]`` -> ``[N,C]``.

    Patches with no observation return exactly zero. Without this an empty patch would emit the
    convolution's bias, which is a learned constant unrelated to any measurement.
    """
    m = mask.to(features.dtype).unsqueeze(1)  # [N,1,K]
    n = m.sum(dim=-1)  # [N,1]
    total = (features * m).sum(dim=-1)
    return torch.where(n > 0, total / n.clamp_min(1.0), torch.zeros_like(total))


class WaveformBranch(nn.Module):
    """Conv1d over a patch's 12 positions, then masked mean pooling. §13.2."""

    def __init__(self, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        """``[B,24,K]`` and ``[B,24,K]`` -> ``[B,24,C]``."""
        b, p, k = patches.shape
        x = patches.reshape(b * p, 1, k)
        h = F.gelu(self.conv(x))
        pooled = masked_mean_pool(h, mask.reshape(b * p, k))
        return pooled.reshape(b, p, -1)


class StatsBranch(nn.Module):
    """Linear projection of (mean, std) per patch. §13.2.

    Appendix C.2 ends the statistics definition with "Empty patches are zeroed by the validity
    mask". Zeroing the *inputs* does not achieve that: the projection carries a bias and the
    activation is GELU, so a fully unobserved patch emits `gelu(bias)` -- a learned, non-zero,
    input-independent vector that the Transformer cannot distinguish from a real measurement.
    The zeroing therefore has to happen after the projection, which is what `valid` does.
    """

    def __init__(self, out_features: int, *, zero_empty: bool = True) -> None:
        super().__init__()
        self.zero_empty = zero_empty
        self.proj = nn.Linear(2, out_features)

    def forward(self, mean: Tensor, std: Tensor, valid: Tensor | None = None) -> Tensor:
        out = F.gelu(self.proj(torch.stack([mean, std], dim=-1)))
        if self.zero_empty and valid is not None:
            out = out * valid.unsqueeze(-1).to(out.dtype)
        return out


class StateEmbedder(nn.Module):
    """State stream: waveform 64 + intra-patch difference 16 + statistics 48 -> token 64."""

    def __init__(self, *, zero_empty_patches: bool = True) -> None:
        super().__init__()
        self.waveform = WaveformBranch(STATE_WAVEFORM)
        self.difference = WaveformBranch(STATE_DIFF)
        self.statistics = StatsBranch(STATE_STATS, zero_empty=zero_empty_patches)
        self.project = nn.Linear(STATE_WAVEFORM + STATE_DIFF + STATE_STATS, STATE_TOKEN)
        self.norm = nn.LayerNorm(STATE_TOKEN)

    def forward(
        self,
        state: Tensor,
        mask: Tensor,
        diff: Tensor,
        diff_valid: Tensor,
        mean: Tensor,
        std: Tensor,
    ) -> Tensor:
        b = state.shape[0]
        wave = self.waveform(state.reshape(b, PATCHES, STEPS_PER_PATCH),
                             mask.reshape(b, PATCHES, STEPS_PER_PATCH))
        d = self.difference(diff, diff_valid)
        s = self.statistics(mean, std, mask.reshape(b, PATCHES, STEPS_PER_PATCH).any(dim=-1))
        return self.norm(self.project(torch.cat([wave, d, s], dim=-1)))


class EventEmbedder(nn.Module):
    """Event stream: residual 48 + rate-of-change 48 + RoC statistics 32 -> token 64."""

    def __init__(self, *, zero_empty_patches: bool = True) -> None:
        super().__init__()
        self.residual = WaveformBranch(EVENT_RESIDUAL)
        self.roc = WaveformBranch(EVENT_ROC)
        self.statistics = StatsBranch(EVENT_STATS, zero_empty=zero_empty_patches)
        self.project = nn.Linear(EVENT_RESIDUAL + EVENT_ROC + EVENT_STATS, EVENT_TOKEN)
        self.norm = nn.LayerNorm(EVENT_TOKEN)

    def forward(
        self,
        event: Tensor,
        mask: Tensor,
        roc: Tensor,
        roc_valid: Tensor,
        roc_mean: Tensor,
        roc_std: Tensor,
    ) -> Tensor:
        b = event.shape[0]
        shape = (b, PATCHES, STEPS_PER_PATCH)
        res = self.residual(event.reshape(shape), mask.reshape(shape))
        r = self.roc(roc.reshape(shape), roc_valid.reshape(shape))
        s = self.statistics(roc_mean, roc_std, roc_valid.reshape(shape).any(dim=-1))
        return self.norm(self.project(torch.cat([res, r, s], dim=-1)))


class Fusion(nn.Module):
    """concat(state 64, event 64) -> 128 physiological token. §13.2."""

    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(STATE_TOKEN + EVENT_TOKEN, FUSED_DIM)
        self.norm = nn.LayerNorm(FUSED_DIM)

    def forward(self, state_tokens: Tensor, event_tokens: Tensor) -> Tensor:
        return self.norm(self.project(torch.cat([state_tokens, event_tokens], dim=-1)))


class TimePositionEmbedding(nn.Module):
    """Learned patch position combined with circadian phase through a learnable gate. §13.4.

    ``tau_i = position_i + sigmoid(g) * time_i``, with ``g`` a 128-vector initialised to zero,
    so the gate starts at 0.5 and the model can learn how much absolute time-of-day to admit.
    Vector-versus-scalar gate and patch-start-versus-centre time are both Tier-2 ablations.
    """

    def __init__(self, dim: int = FUSED_DIM, patches: int = PATCHES,
                 *, use_circadian: bool = True) -> None:
        super().__init__()
        #: Tier-1 ablation 8. With absolute circadian phase removed the model keeps learned patch
        #: position -- it still knows a patch's ordinal place in the day -- but loses which wall
        #: clock hour that is. Dropping both would confound "no time of day" with "no ordering".
        self.use_circadian = use_circadian
        self.time_proj = nn.Linear(2, dim)
        self.position = nn.Parameter(torch.zeros(patches, dim))
        nn.init.normal_(self.position, std=0.02)
        self.gate_raw = nn.Parameter(torch.zeros(dim))

    def forward(self, circadian_start_index: Tensor) -> Tensor:
        """``[B]`` start indices -> ``[B,24,128]`` temporal embedding."""
        device = self.position.device
        patch_offset = torch.arange(PATCHES, device=device) * STEPS_PER_PATCH
        a = (circadian_start_index.to(device).unsqueeze(-1) + patch_offset) % 288
        angle = 2.0 * torch.pi * a.float() / 288.0
        circular = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        if not self.use_circadian:
            return self.position.unsqueeze(0).expand(circadian_start_index.shape[0], -1, -1)
        time = self.time_proj(circular)
        return self.position.unsqueeze(0) + torch.sigmoid(self.gate_raw) * time
