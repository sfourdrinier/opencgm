"""Context encoder and the full front-end pipeline. Blueprint §10.1, §13.5, §14.

The operation order in `encode` is the part that matters and the part that fails silently.
Blueprint §10.1: patches selected for JEPA masking are removed from the visible signal
**before** normalization, filtering and statistics. An implementation that normalises the whole
day, filters it, tokenises, and only then swaps in mask tokens leaks the hidden patches into
every visible token through the shared statistics. It trains, the loss falls, and nothing
reports a problem — the model has simply been given the answer.

`tests/golden/test_no_leakage.py` fails if that order is disturbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .causal_gaussian import CausalGaussian, masked_instance_norm
from .reference import PATCHES, STEPS_PER_PATCH
from .statistics import (
    intra_patch_difference,
    patch_density,
    patch_mean_std,
    rate_of_change,
    roc_patch_mean_std,
)
from .stream_embedder import (
    FUSED_DIM,
    EventEmbedder,
    Fusion,
    StateEmbedder,
    TimePositionEmbedding,
)

ENCODER_LAYERS = 3  # PAPER_EXACT §13.1
ENCODER_HEADS = 4  # PAPER_EXACT
ENCODER_FFN = 256  # PAPER_EXACT
ENCODER_DROPOUT = 0.1  # INFERRED_RECONSTRUCTION §13.5


def transformer(layers: int, *, dim: int = FUSED_DIM, dropout: float = ENCODER_DROPOUT):
    """Pre-norm GELU Transformer. Norm order and dropout are inferred defaults (§13.5)."""
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=ENCODER_HEADS,
        dim_feedforward=ENCODER_FFN,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


@dataclass
class EncoderOutput:
    """Blueprint §19.1."""

    contextual_tokens: Tensor  # [B,24,128]
    daily_embedding: Tensor  # [B,128]
    state_tokens: Tensor  # [B,24,64], pre-Transformer
    event_tokens: Tensor  # [B,24,64], pre-Transformer
    state_signal: Tensor  # [B,288]
    event_signal: Tensor  # [B,288]
    patch_density: Tensor  # [B,24]
    temporal: Tensor  # [B,24,128]
    sigma: Tensor


class ContextEncoder(nn.Module):
    """Front end plus the three-layer contextual Transformer."""

    #: Tier-1 ablations 1-3. Appendix D specifies the variants: "The Raw Input variant directly
    #: tokenizes the aligned glucose sequence[...] The State-stream Only variant uses the filtered
    #: low-frequency trend with trend differences and state statistics. The Event-stream Only
    #: variant uses the residual component with rate-of-change features and event statistics."
    #:
    #: Implemented by zeroing the other stream's token before fusion rather than by narrowing the
    #: fusion projection. That keeps parameter count, initialisation and optimiser state
    #: identical across variants, so the measured difference is the information the stream
    #: carries and not a capacity difference confounded with it. INFERRED_RECONSTRUCTION: the
    #: paper does not say how it parameterised the single-stream variants.
    STREAMS = ("both", "state", "event", "raw")

    def __init__(
        self, *, layers: int = ENCODER_LAYERS, dropout: float = ENCODER_DROPOUT,
        raw_statistics: bool = True, streams: str = "both",
        learnable_sigma: bool = True, use_circadian: bool = True,
        zero_empty_patches: bool = True,
    ) -> None:
        super().__init__()
        if streams not in self.STREAMS:
            raise ValueError(f"streams must be one of {self.STREAMS}, got {streams!r}")
        self.raw_statistics = raw_statistics
        self.streams = streams
        self.use_circadian = use_circadian
        self.gaussian = CausalGaussian(learnable=learnable_sigma)
        self.state_embedder = StateEmbedder(zero_empty_patches=zero_empty_patches)
        self.event_embedder = EventEmbedder(zero_empty_patches=zero_empty_patches)
        self.fusion = Fusion()
        self.temporal = TimePositionEmbedding(use_circadian=use_circadian)
        self.mask_token = nn.Parameter(torch.zeros(FUSED_DIM))
        nn.init.normal_(self.mask_token, std=0.02)
        self.transformer = transformer(layers, dropout=dropout)

    def tokenize(
        self, values: Tensor, mask: Tensor, circadian_start: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Signal to pre-Transformer tokens, using only what ``mask`` admits.

        Every statistic below is computed from ``mask`` alone. Passing a mask with the JEPA
        patches already removed is therefore sufficient to guarantee the §10.1 ordering; there
        is no separate code path that could see the hidden data.
        """
        normalized = masked_instance_norm(values, mask)
        state_signal, event_signal = self.gaussian(normalized, mask)
        if self.streams == "raw":
            # Ablation 1: no decomposition. Both branches see the undecomposed sequence, which is
            # what "directly tokenizes the aligned glucose sequence" means -- the filter is the
            # thing under test, so it is bypassed rather than merely ignored downstream.
            state_signal = normalized
            event_signal = normalized

        # Paper appendix C.2 computes patch statistics and rate-of-change from the *aligned*
        # sequence X-hat, in mg/dL, while C.3 applies the Gaussian filter to the *normalized*
        # sequence X-tilde. The two symbols are used consistently and exclusively: C.2 uses
        # X-hat four times and X-tilde never; C.3 uses X-tilde six times and X-hat never.
        #
        # This matters more than it looks. Per-window normalisation removes absolute glucose
        # level, so deriving the statistics from the normalised signal — as we did until now —
        # leaves the entire encoder blind to whether a day sat at 105 or 205 mg/dL. Feeding raw
        # statistics restores that, and it is measurably the information several downstream
        # phenotypes depend on: obesity classification loses 0.21 ROC-AUC when absolute level is
        # withheld from the hand-engineered baseline.
        #
        # `raw_statistics=False` keeps the previous behaviour for the ablation. See D019.
        statistics_source = values if self.raw_statistics else state_signal
        roc_source = values if self.raw_statistics else normalized

        b = values.shape[0]
        shape = (b, PATCHES, STEPS_PER_PATCH)
        s_mean, s_std = patch_mean_std(statistics_source, mask)
        diff, diff_valid = intra_patch_difference(state_signal, mask)
        roc, roc_valid = rate_of_change(roc_source, mask)
        r_mean, r_std = roc_patch_mean_std(roc, roc_valid)

        state_tokens = self.state_embedder(
            state_signal, mask, diff, diff_valid, s_mean, s_std
        )
        event_tokens = self.event_embedder(
            event_signal, mask, roc, roc_valid, r_mean, r_std
        )
        if self.streams == "state":
            event_tokens = torch.zeros_like(event_tokens)
        elif self.streams == "event":
            state_tokens = torch.zeros_like(state_tokens)
        physiological = self.fusion(state_tokens, event_tokens)
        tau = self.temporal(circadian_start)
        del shape
        return physiological, tau, state_tokens, event_tokens, state_signal, event_signal

    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        circadian_start: Tensor,
        context_patch_mask: Tensor | None = None,
        physical_mask: Tensor | None = None,
    ) -> EncoderOutput:
        """Encode a batch.

        ``mask`` is the branch-visible mask: for the online branch it already has the hidden
        patches removed. ``context_patch_mask`` is ``[B,24]`` and True where a patch is hidden;
        those token positions are replaced by the learned mask token, which keeps its temporal
        embedding so the predictor knows *where* the missing patch sits (§13.6).

        ``physical_mask`` supplies the density used for loss weighting. Density must come from
        the physical observation mask, not the online-visible one — otherwise a masked patch
        would have density zero and silently drop out of the loss it is supposed to drive.
        """
        physiological, tau, state_tokens, event_tokens, state_signal, event_signal = (
            self.tokenize(values, mask, circadian_start)
        )

        if context_patch_mask is not None:
            hidden = context_patch_mask.unsqueeze(-1)
            physiological = torch.where(hidden, self.mask_token.expand_as(physiological),
                                        physiological)

        contextual = self.transformer(physiological + tau)
        density_source = physical_mask if physical_mask is not None else mask

        return EncoderOutput(
            contextual_tokens=contextual,
            daily_embedding=contextual.mean(dim=1),  # unweighted, PAPER_EXACT §19.1
            state_tokens=state_tokens,
            event_tokens=event_tokens,
            state_signal=state_signal,
            event_signal=event_signal,
            patch_density=patch_density(density_source),
            temporal=tau,
            sigma=self.gaussian.sigma,
        )


def parameter_report(module: nn.Module) -> dict[str, int]:
    """Component-level trainable parameter counts. Blueprint §13.7 requires this in CI."""
    report: dict[str, int] = {}
    for name, child in module.named_children():
        report[name] = sum(p.numel() for p in child.parameters() if p.requires_grad)
    direct = sum(
        p.numel() for n, p in module.named_parameters(recurse=False) if p.requires_grad
    )
    if direct:
        report["_direct"] = direct
    report["TOTAL"] = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return report
