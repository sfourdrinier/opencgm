"""PPG pilot heads (D023)."""

# Two heads sit on top of the PpgStudentEncoder.
#
# 1. **TeacherLatentHead** projects the 64-dim student feature to the teacher's 256-dim latent
#    space. Trained against the frozen teacher encoder's mean-pooled 256-dim `opencgm_mean`
#    token at the same timestamps. Loss = 0.5 . MSE + 0.5 . (1 - cosine), masked.
#
# 2. **DirectGlucoseHead** projects the 64-dim student feature to a per-token glucose value
#    (mmol/L). Trained with causal Gaussian NLL (the same parameterisation the strict CGM
#    architecture uses; paper section 12). Masked.
#
# Both heads include their own validity-mask handling - the CGM observation mask is the
# authoritative input to the loss. No interpolation.

from __future__ import annotations

import torch
from torch import nn


class TeacherLatentHead(nn.Module):
    """Maps (batch, 64) -> (batch, teacher_dim). Aligns to the teacher's opencgm token.

    teacher_dim defaults to 128 because the strict ep40 teacher emits 24 tokens at 128
    dims. If a different teacher is used, pass the matching dim explicitly.
    """

    def __init__(self, feature_dim: int = 64, teacher_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, teacher_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2 or features.shape[-1] != self.proj[0].in_features:
            raise ValueError(
                f"expected (batch, {self.proj[0].in_features}) features, got "
                f"{tuple(features.shape)}"
            )
        return self.proj(features)


class DirectGlucoseHead(nn.Module):
    """Maps (batch, 64) -> (batch, 2) = (mean_mmol_per_l, log_sigma)."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2 or features.shape[-1] != self.proj[0].in_features:
            raise ValueError(
                f"expected (batch, {self.proj[0].in_features}) features, got "
                f"{tuple(features.shape)}"
            )
        return self.proj(features)


def alignment_loss(
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """0.5 * MSE + 0.5 * (1 - cosine) on per-token aligned latents, masked.

    Args:
        student_latent: (batch, teacher_dim) from the TeacherLatentHead.
        teacher_latent: (batch, teacher_dim) from the frozen teacher, at the same timestamps.
        mask: (batch,) 1.0 where the CGM timestamp is observed, 0.0 where missing.
    """
    if student_latent.shape != teacher_latent.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_latent.shape)} vs teacher "
            f"{tuple(teacher_latent.shape)}"
        )
    if mask.shape != (student_latent.shape[0],):
        raise ValueError(f"mask must be (batch,), got {tuple(mask.shape)}")
    valid = mask > 0.5
    if not valid.any():
        # Zero loss without grads, but valid for backward.
        return student_latent.sum() * 0.0
    s = student_latent[valid]
    t = teacher_latent[valid]
    mse = (s - t).pow(2).mean()
    cos = torch.nn.functional.cosine_similarity(s, t, dim=-1).mean()
    return 0.5 * mse + 0.5 * (1.0 - cos)


def gaussian_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Causal Gaussian NLL on (mean, log_sigma) predictions vs target glucose, masked.

    Args:
        pred: (batch, 2) = (mean_mmol_per_l, log_sigma) from DirectGlucoseHead.
        target: (batch,) mmol/L glucose values, NaN where missing.
        mask: (batch,) 1.0 where the CGM timestamp is observed, 0.0 where missing.
    """
    if pred.shape[-1] != 2:
        raise ValueError(f"pred last dim must be 2, got {pred.shape}")
    mean = pred[:, 0]
    # log_sigma clamped to [-3, 3] = sigma in [0.05, 20].
    log_sigma = pred[:, 1].clamp(min=-3.0, max=3.0)
    valid = mask > 0.5
    if not valid.any():
        return mean.sum() * 0.0
    m = mean[valid]
    t = target[valid]
    s = log_sigma[valid]
    var = (2 * s).exp()  # sigma squared
    nll = 0.5 * ((t - m) ** 2 / var) + s + 0.5 * torch.log(torch.tensor(2 * torch.pi))
    return nll.mean()
