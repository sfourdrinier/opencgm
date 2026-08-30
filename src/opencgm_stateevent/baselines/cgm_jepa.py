"""CGM-JEPA, ported from the authors' released code. The comparator GlucoFM's headline claim uses.

GlucoFM's central quantitative claim is relative, not absolute: +4.11 PR-AUC and +4.34 ROC-AUC over
CGM-JEPA, with both re-pretrained on the same corpus. An absolute score degrades when you have 30.9%
of the pretraining hours; a margin between two models trained on the same reduced corpus does not.
So this comparator, not the paper's 66.7, is the thing worth reproducing.

Provenance
----------
`SOURCE_VERIFIED` against the authors' own implementation, MIT licensed:

    https://github.com/cruiseresearchgroup/CGM-JEPA   (master @ 2026-05-11, read 2026-08-28)
    weights: https://huggingface.co/CRUISEResearchGroup/CGM-JEPA

Every structural choice below is read from `models/encoder.py`, `models/predictor.py`,
`utils/embed.py`, `utils/modules.py`, `utils/mask_utils.py`, `config/config_pretrain.py` and
`pretrain/pretrain_cgm_jepa.py` in that repository, not inferred from GlucoFM's appendix B. It is a
reimplementation in our own idiom against their source, not a copy of their files.

An earlier draft of this module inferred four of these choices from appendix B alone, because the
appendix was assumed to be all that was available. Three of those guesses -- omitting the target
layer-norm, feeding mask tokens through the context encoder, dropout 0.1 with a half-width FFN --
each independently weakened the baseline. A baseline we build and then beat is the most self-serving
result this project could produce, so guessing was never acceptable when the source existed.

Parameter count: 521,584, which is the authors' own 522,160 less one module their config leaves
dead, and rounds to the 0.52M the GlucoFM paper reports for CGM-JEPA in Table 3. That agreement is
the gate that says the port is structurally the same model; `tests/golden/test_cgm_jepa.py`
asserts it, along with the exact 576-parameter explanation for the difference.

Where the authors and GlucoFM's appendix disagree
-------------------------------------------------
Appendix B says inputs are "linearly interpolated ... and **normalized**". The authors' default is
`normalize_x: False` -- raw mg/dL, no per-window standardization (`config/config_pretrain.py:17`).
GlucoFM claims to have retrained "using the official configuration", so it contradicts itself.

We follow the authors, and we do so in the direction that *helps* the baseline. Raw mg/dL preserves
absolute glucose level, and this project's own D019 finding measured absolute level as worth up to
0.21 ROC-AUC on obesity. Standardizing per window would strip that from the comparator while our
model keeps it -- a large, quantified, one-sided handicap. `normalize` below exists only so the
appendix's reading can be run as a labelled sensitivity arm. See DECISIONS.md D022.

The interpolation stands, and it is the substantive contrast rather than an oversight: CGM-JEPA
fills every gap and treats the filled values as observed; GlucoFM keeps the physical mask.
Our Tier-1 ablation measured that choice as worth 0.017 ROC-AUC inside our own architecture. A
faithful comparator must interpolate even though this project's standing rule forbids it everywhere
else.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --- config/config_pretrain.py, PAPER_EXACT and SOURCE_VERIFIED agree -----------------------------
EMBED_DIM = 96
NUM_LAYERS = 3
NUM_HEADS = 6
MASK_RATIO = 0.25
EPOCHS = 101
BASE_LR = 1e-4
BATCH_SIZE = 128
OFFICIAL_SEED = 43

# --- SOURCE_VERIFIED; absent from appendix B, read from the authors' code -------------------------
PATCH_SIZE = 12          # config_pretrain.py:20
CONV_KERNEL = 3          # encoder_kernel_size, applied *within* each patch
PATCHES = 24             # 288 / 12
MLP_RATIO = 4.0          # encoder.py:34 default
DROPOUT = 0.0            # config_pretrain.py:46 -- not 0.1
PREDICTOR_DIM = 48       # config_pretrain.py:49
PREDICTOR_HEADS = 2      # config_pretrain.py:50
PREDICTOR_LAYERS = 1     # config_pretrain.py:51
PROJ_HIDDEN = 1024       # encoder.py:79
PROJ_OUT = 48            # encoder.py:80
EMA_MOMENTUM = 0.997     # config_pretrain.py:38
IPE_SCALE = 1.25         # config_pretrain.py:24
WARMUP_RATIO = 0.15      # config_pretrain.py:23
CLIP_GRAD = 1.0          # config_pretrain.py:22

#: Our port's encoder parameter count. The gate that says the port is structurally the same model.
#:
#: The authors' own encoder counts 522,160. The 576 difference is exactly `Linear(5, 96)`: their
#: `DataEmbedding` always constructs a time-feature embedding that `use_time_feature: False`
#: (config_pretrain.py:29) leaves permanently dead -- their own comment says the evaluation data has
#: no timestamps. We omit the dead module rather than carry dead weights to make a number match.
#: Both round to the 0.52M the GlucoFM paper reports in Table 3.
ENCODER_PARAMETERS = 521_584
AUTHORS_ENCODER_PARAMETERS = 522_160
DEAD_TIME_EMBEDDING = 576


class MLP(nn.Module):
    """utils/modules.py MLP."""

    def __init__(self, in_features: int, hidden: int, out_features: int | None = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features or in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """utils/modules.py Block: pre-norm, qkv_bias=True, GELU, residual."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=DROPOUT, batch_first=True, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * MLP_RATIO))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


def sinusoidal(length: int, dim: int) -> Tensor:
    """utils/embed.py PositionalEmbedding -- fixed, not learned."""
    position = torch.arange(length).float().unsqueeze(1)
    div = (torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)).exp()
    out = torch.zeros(length, dim)
    out[:, 0::2] = torch.sin(position * div)
    out[:, 1::2] = torch.cos(position * div)
    return out.unsqueeze(0)


class ValueEmbedding(nn.Module):
    """utils/embed.py ValueEmbedding.

    Each 12-step patch is convolved *independently* -- Conv1d(1, 96, kernel 3, stride 3) gives four
    positions inside the patch, which are then flattened and projected. It is a small local filter
    bank inside the hour, not a single linear read of the whole patch. Our first draft used
    `nn.Linear(12, 96)`, which is strictly less expressive.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, EMBED_DIM, kernel_size=CONV_KERNEL, stride=CONV_KERNEL, bias=True)
        inner = (PATCH_SIZE - CONV_KERNEL) // CONV_KERNEL + 1
        self.fc = nn.Linear(EMBED_DIM * inner, EMBED_DIM)

    def forward(self, patches: Tensor) -> Tensor:
        b, n, length = patches.shape
        x = self.proj(patches.reshape(b * n, 1, length))
        return self.fc(x.reshape(b * n, -1)).reshape(b, n, EMBED_DIM)


def gather(x: Tensor, index: Tensor) -> Tensor:
    """utils/mask_utils.py apply_mask -- keep the patches named by `index`."""
    return torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.size(-1)))


class Encoder(nn.Module):
    """models/encoder.py Encoder. This is what is frozen and probed downstream."""

    def __init__(self) -> None:
        super().__init__()
        self.value_embedding = ValueEmbedding()
        self.register_buffer("position", sinusoidal(PATCHES, EMBED_DIM), persistent=False)
        self.blocks = nn.ModuleList(Block(EMBED_DIM, NUM_HEADS) for _ in range(NUM_LAYERS))
        self.encoder_norm = nn.LayerNorm(EMBED_DIM)
        self.proj = MLP(EMBED_DIM, PROJ_HIDDEN, PROJ_OUT)

    def forward(self, patches: Tensor, keep: Tensor | None = None) -> Tensor:
        """`patches` is [B,24,12]; `keep` names the context patches to retain.

        Masked patches are **dropped before attention**, not replaced by a learned placeholder.
        That is canonical I-JEPA and it is what the authors do; substituting mask tokens at the
        encoder input makes the pretext task easier and spends encoder capacity on placeholders.
        """
        x = self.value_embedding(patches) + self.position
        if keep is not None:
            x = gather(x, keep)
        for block in self.blocks:
            x = block(x)
        return self.encoder_norm(x)


class Predictor(nn.Module):
    """models/predictor.py Predictor: narrow, mask-query conditioned on target position.

    The 48-dim width is deliberate. A weak predictor cannot solve the task on its own, which forces
    the information into the encoder -- the representation we actually want. A full-width predictor
    would quietly relocate the model's competence away from the thing being evaluated.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predictor_embed = nn.Linear(EMBED_DIM, PREDICTOR_DIM)
        self.register_buffer("position", sinusoidal(PATCHES, PREDICTOR_DIM), persistent=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, PREDICTOR_DIM))
        self.blocks = nn.ModuleList(
            Block(PREDICTOR_DIM, PREDICTOR_HEADS) for _ in range(PREDICTOR_LAYERS)
        )
        self.predictor_norm = nn.LayerNorm(PREDICTOR_DIM)
        self.predictor_proj = nn.Linear(PREDICTOR_DIM, EMBED_DIM)

    def forward(self, context: Tensor, keep: Tensor, predict: Tensor) -> Tensor:
        b, n_context, _ = context.shape
        position = self.position.expand(b, -1, -1)
        x = self.predictor_embed(context) + gather(position, keep)
        queries = self.mask_token.expand(b, predict.size(1), -1) + gather(position, predict)
        x = torch.cat([x, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.predictor_proj(self.predictor_norm(x)[:, n_context:])


@dataclass
class JEPAOutput:
    loss: Tensor
    predict: Tensor


class CGMJEPA(nn.Module):
    """Online encoder, frozen EMA target encoder, narrow predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        """The teacher never runs in training mode.

        Moot at the authors' dropout of 0.0, but a stochastic regression target would reward the
        student for predicting the mean of noise, and the sensitivity arm may re-enable dropout.
        """
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(
            self.encoder.parameters(), self.target.parameters(), strict=True
        ):
            target.data.mul_(momentum).add_((1.0 - momentum) * online.detach().data)

    def forward(self, patches: Tensor, generator: torch.Generator) -> JEPAOutput:
        b = patches.shape[0]
        n_predict = max(1, round(MASK_RATIO * PATCHES))
        order = torch.rand(b, PATCHES, device=patches.device, generator=generator).argsort(dim=1)
        predict, keep = order[:, :n_predict], order[:, n_predict:]

        with torch.no_grad():
            # Layer-norming the target is the authors' own guard against representation drift --
            # the same failure this project diagnosed independently as D018. Omitting it would
            # hand us a win caused by the baseline collapsing, not by our method.
            target = gather(F.layer_norm(self.target(patches), (EMBED_DIM,)), predict)

        predicted = self.predictor(self.encoder(patches, keep=keep), keep, predict)
        return JEPAOutput(loss=(predicted - target).abs().mean(), predict=predict)

    @torch.no_grad()
    def embed(self, patches: Tensor) -> Tensor:
        """Frozen encoder, pre-projection, mean-pooled over the 24 tokens. Appendix B."""
        was_training = self.training
        self.eval()
        try:
            return self.encoder(patches).mean(dim=1)
        finally:
            self.train(was_training)


def ema_momentum(epoch: int, total_epochs: int = EPOCHS) -> float:
    """pretrain_cgm_jepa.py:105 -- linear ramp, stepped per epoch, never reaching 1.0.

    With ipe_scale 1.25 the schedule is sized for 1.25x the epochs actually run, so training ends
    around 0.9994 and the teacher keeps moving to the last step.
    """
    span = total_epochs * IPE_SCALE
    return EMA_MOMENTUM + epoch * (1.0 - EMA_MOMENTUM) / span


def learning_rate_scale(step: int, total_steps: int) -> float:
    """pretrain_cgm_jepa.py:32 -- 15% linear warmup, then linear decay to zero."""
    warmup = int(WARMUP_RATIO * total_steps)
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))


def to_patches(values: Tensor, normalize: bool = False) -> Tensor:
    """[B,288] dense sequence -> [B,24,12] patches.

    `normalize` is off by default because the authors' default is off. Turning it on reproduces
    GlucoFM appendix B's wording instead, and is only ever run as a labelled sensitivity arm --
    it strips absolute glucose level from the baseline while our model retains it.
    """
    if normalize:
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True).clamp_min(1e-4)
        values = (values - mean) / std
    return values.reshape(values.shape[0], PATCHES, PATCH_SIZE)
