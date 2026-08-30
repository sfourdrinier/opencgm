"""PPG student encoder for the section-23 Lane-D teacher-student pilot.

This is a *small* encoder on purpose. The teacher-student framing assumes the student has to
discover glucose-relevant structure from a PPG signal at 64 Hz raw photoplethysmography. The
encoder should be expressive enough to extract pulse-rate, pulse-interval, and pulse-amplitude
features, but small enough that its representation can plausibly be aligned with the teacher's
CGM-rate representation - which is a token-per-5-minute-grid-position embedding.

The encoder downsamples 64 Hz to 1 Hz using three strided 1D convolutions, then per-patch
mean-pools to produce one student token per 5-minute CGM timestep. Two heads project to
(a) the teacher's 256-dim latent space for alignment and (b) per-token glucose values.

The student is **NOT** a GlucoFM model. It is a tiny 1D-conv stack designed for the pilot only.
Tag: PROPOSED_EXTENSION (D023).

Architecture:
  - Input: (batch, channels=1, samples=19200) at 64 Hz, i.e. 5 minutes of raw BVP.
  - Stem: Conv1d(1, 32, kernel=64, stride=32) -> 32 x 600 (1 Hz effective, receptive field 1s).
  - Block 1: Conv1d(32, 32, k=7, s=2, padding=3) + GELU -> 32 x 300.
  - Block 2: Conv1d(32, 64, k=7, s=2, padding=3) + GELU -> 64 x 150.
  - Block 3: Conv1d(64, 64, k=7, s=2, padding=3) + GELU -> 64 x 75.
  - Block 4: Conv1d(64, 64, k=5, s=3, padding=2) + GELU -> 64 x 25.
  - Per-patch mean -> 64-dim feature per 5-minute window.

Output: (batch, 64) - student per-token features for one 5-minute patch. For a 24-hour window
with 288 patches, the encoder is applied 288 times to produce (batch * 288, 64), reshaped to
(batch, 288, 64).

Trainable parameters: ~73K. This is ~7% of the GlucoFM teacher size, which is intentional - the
pilot measures whether a small student can align with a larger teacher, not whether the student
outperforms it on raw capacity.
"""

from __future__ import annotations

import torch
from torch import nn


class PpgStudentEncoder(nn.Module):
    """A small 1D-conv encoder mapping 64 Hz raw BVP to 64-dim per-5-min-token features."""

    PATCH_LEN = 19200  # 5 minutes x 60 s/min x 64 Hz
    FEATURE_DIM = 64

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=3, padding=2),
            nn.GELU(),
        )

    def forward(self, bvp: torch.Tensor) -> torch.Tensor:
        """Encode a batch of 5-minute BVP patches.

        Args:
            bvp: shape (batch, PATCH_LEN) raw 64 Hz photoplethysmography in arbitrary units.

        Returns:
            features: shape (batch, FEATURE_DIM) - one token per 5-minute patch.
        """
        if bvp.dim() != 2 or bvp.shape[-1] != self.PATCH_LEN:
            raise ValueError(
                f"expected (batch, {self.PATCH_LEN}) raw 64 Hz BVP, got {tuple(bvp.shape)}"
            )
        x = bvp.unsqueeze(1)  # (batch, 1, PATCH_LEN)
        x = self.stem(x)
        x = self.blocks(x)
        return x.mean(dim=-1)  # (batch, FEATURE_DIM)


def count_parameters(module: nn.Module) -> int:
    """Trainable parameter count, excluding buffers and frozen modules."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
