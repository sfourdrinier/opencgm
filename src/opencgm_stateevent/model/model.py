"""The complete pretraining model. Blueprint §10.1, §14, §15.

`forward` implements the ordering §10.1 sets out. The comments marking each step are not
decoration: the ordering is the one part of this method that fails silently when wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder import ContextEncoder, EncoderOutput
from .heads import EMATarget, Predictor, TransitionHead
from .losses import (
    apply_context_mask,
    masked_contextual_loss,
    sample_context_mask,
    temporal_dynamics_loss,
    total_loss,
    transition_weight,
)
from .reference import PATCHES
from .stream_embedder import EVENT_TOKEN, STATE_TOKEN


@dataclass
class PretrainOutput:
    loss: Tensor
    mcr: Tensor
    td: Tensor
    online: EncoderOutput
    context_patch_mask: Tensor
    realized_mask_ratio: Tensor


#: Constructor arguments that change what the model computes but add and remove no parameter
#: tensor, mapped to the value to assume when a checkpoint's config predates the flag. Because
#: they leave the state dict shape untouched, `load_state_dict(strict=True)` accepts a mismatch
#: silently and the model then computes different features from identical weights. Every consumer
#: of a checkpoint must therefore read these back from it rather than from today's defaults.
#: See D019.
ARCHITECTURE_FLAGS: dict[str, object] = {
    "raw_statistics": False, "normalize_targets": False,
    "streams": "both", "learnable_sigma": True, "use_circadian": True,
    "zero_empty_patches": False,
}


def architecture_of(state: dict) -> dict:
    """The architecture flags a checkpoint was *trained* with, defaulting to pre-flag behaviour."""
    config = state.get("config") or {}
    return {name: config.get(name, fallback) for name, fallback in ARCHITECTURE_FLAGS.items()}


class OpenCGMStateEvent(nn.Module):
    """Online encoder, EMA target, predictor and transition heads."""

    def __init__(self, *, dropout: float = 0.1, normalize_targets: bool = False,
                 raw_statistics: bool = True, streams: str = "both",
                 learnable_sigma: bool = True, use_circadian: bool = True,
                 zero_empty_patches: bool = True) -> None:
        """`normalize_targets` is a `PROPOSED_EXTENSION`, off by default.

        The blueprint regresses onto raw EMA target tokens. I-JEPA, BYOL and data2vec all
        normalise the target first, and BYOL's ablation reports runaway representation norm
        without it — which is what we observe here (target token norm drifting 18 -> 34 -> 16 while
        the loss rises). Regressing onto an unnormalised moving target mixes representation
        mismatch with target scale drift, so the loss stops meaning what it appears to mean.

        Left off by default so the strict reproduction stays exactly what the blueprint
        specifies. Enabled only in separately-labelled runs. See D018.
        """
        super().__init__()
        self.normalize_targets = normalize_targets
        self.online = ContextEncoder(
            dropout=dropout, raw_statistics=raw_statistics, streams=streams,
            learnable_sigma=learnable_sigma, use_circadian=use_circadian,
            zero_empty_patches=zero_empty_patches,
        )
        self.target = EMATarget(self.online)
        self.predictor = Predictor(dropout=dropout)
        self.transition_state = TransitionHead(STATE_TOKEN)
        self.transition_event = TransitionHead(EVENT_TOKEN)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        values: Tensor,
        physical_mask: Tensor,
        circadian_start: Tensor,
        generator: torch.Generator,
        *,
        lambda_td: float = 1.0,
    ) -> PretrainOutput:
        # 1. sample which patches the online branch may not see
        ctx = sample_context_mask(
            values.shape[0], PATCHES, generator, device=values.device
        )

        # 2. remove them from the online-visible mask BEFORE anything reads the signal.
        #    Every downstream statistic derives from this mask, so nothing can leak.
        online_mask = apply_context_mask(physical_mask, ctx)

        # 3. online branch: normalize, filter, tokenize on the reduced mask, then swap in the
        #    learned mask token at hidden positions before contextual attention.
        online = self.online(
            values, online_mask, circadian_start,
            context_patch_mask=ctx, physical_mask=physical_mask,
        )

        # 4. target branch: same augmented physical view, full mask, no gradients.
        with torch.no_grad():
            target = self.target(values, physical_mask, circadian_start)

        # Optional feature-wise normalisation of the regression targets. Applied to the *target*
        # side only and under no_grad, so it changes what is regressed onto, not the gradient path.
        def as_target(x: Tensor) -> Tensor:
            return F.layer_norm(x, (x.shape[-1],)) if self.normalize_targets else x

        # 5. contextual loss at hidden patches only, weighted by PHYSICAL density.
        predicted = self.predictor(online.contextual_tokens)
        mcr = masked_contextual_loss(
            predicted, as_target(target.contextual_tokens), ctx, online.patch_density
        )

        # 6. temporal dynamics from PRE-Transformer tokens on both sides, so contextual
        #    attention over all 24 patches cannot leak into a one-step-ahead target.
        s_next = self.transition_state(
            online.state_tokens[:, :-1], online.event_tokens[:, :-1], online.temporal[:, :-1]
        )
        e_next = self.transition_event(
            online.event_tokens[:, :-1], online.state_tokens[:, :-1], online.temporal[:, :-1]
        )
        td = temporal_dynamics_loss(
            s_next, as_target(target.state_tokens[:, 1:]),
            e_next, as_target(target.event_tokens[:, 1:]),
            transition_weight(ctx, online.patch_density),
        )

        return PretrainOutput(
            loss=total_loss(mcr, td, lambda_td=lambda_td),
            mcr=mcr,
            td=td,
            online=online,
            context_patch_mask=ctx,
            realized_mask_ratio=ctx.float().mean(dim=1),
        )

    @torch.no_grad()
    def encode(self, values: Tensor, mask: Tensor, circadian_start: Tensor) -> EncoderOutput:
        """Frozen inference path. No JEPA masking; the whole physical view is visible."""
        return self.online(values, mask, circadian_start)
