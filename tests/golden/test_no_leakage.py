"""The mask-before-filter ordering. Blueprint §10.1, GOAL KR2.3.

This is the failure that does not announce itself. If normalization, filtering and statistics
run on the full day and masking happens only after tokenization, every visible token carries
information about the hidden patches through the shared window statistics. Training proceeds,
the loss falls faster than it should, and no diagnostic reports anything wrong — the model has
been given part of the answer it is being asked to predict.

The tests below make the ordering observable: change the data under a hidden patch and nothing
the online branch computes may move.
"""

from __future__ import annotations

import pytest
import torch

from opencgm_stateevent.model.encoder import ContextEncoder
from opencgm_stateevent.model.losses import apply_context_mask, sample_context_mask
from opencgm_stateevent.model.model import OpenCGMStateEvent
from opencgm_stateevent.model.reference import PATCHES, STEPS_PER_PATCH

pytestmark = pytest.mark.leakage

L = 288
B = 2


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def batch(seed: int = 0):
    torch.manual_seed(seed)
    values = torch.randn(B, L) * 30 + 120
    mask = torch.ones(B, L, dtype=torch.bool)
    start = torch.zeros(B, dtype=torch.long)
    return values, mask, start


def eval_encoder(dropout: float = 0.0) -> ContextEncoder:
    enc = ContextEncoder(dropout=dropout)
    enc.eval()
    return enc


# --- the core property --------------------------------------------------------------------


def test_hidden_patch_data_cannot_influence_online_tokens():
    """Perturb the signal under hidden patches only; every online output must be identical."""
    values, mask, start = batch(1)
    ctx = torch.zeros(B, PATCHES, dtype=torch.bool)
    ctx[:, 5:15] = True  # hide patches 5..14
    online_mask = apply_context_mask(mask, ctx)

    perturbed = values.clone()
    hidden_positions = ctx.repeat_interleave(STEPS_PER_PATCH, dim=1)
    perturbed[hidden_positions] += 500.0  # a change no leak-free pipeline can see

    enc = eval_encoder()
    with torch.no_grad():
        a = enc(values, online_mask, start, context_patch_mask=ctx, physical_mask=mask)
        b = enc(perturbed, online_mask, start, context_patch_mask=ctx, physical_mask=mask)

    assert torch.allclose(a.contextual_tokens, b.contextual_tokens, atol=1e-6)
    assert torch.allclose(a.state_tokens, b.state_tokens, atol=1e-6)
    assert torch.allclose(a.event_tokens, b.event_tokens, atol=1e-6)
    assert torch.allclose(a.state_signal, b.state_signal, atol=1e-6)
    assert torch.allclose(a.daily_embedding, b.daily_embedding, atol=1e-6)


def test_masking_after_tokenization_would_leak():
    """Demonstrates the bug this ordering prevents, so the test above is not vacuous.

    Tokenizing on the FULL mask -- the wrong order -- lets the perturbation through. If this
    ever stops failing to match, the leak test above has lost its meaning.
    """
    values, mask, start = batch(1)
    ctx = torch.zeros(B, PATCHES, dtype=torch.bool)
    ctx[:, 5:15] = True
    perturbed = values.clone()
    perturbed[ctx.repeat_interleave(STEPS_PER_PATCH, dim=1)] += 500.0

    enc = eval_encoder()
    with torch.no_grad():
        a = enc(values, mask, start, context_patch_mask=ctx, physical_mask=mask)
        b = enc(perturbed, mask, start, context_patch_mask=ctx, physical_mask=mask)

    assert not torch.allclose(a.contextual_tokens, b.contextual_tokens, atol=1e-6)


def test_apply_context_mask_removes_exactly_the_hidden_positions():
    mask = torch.ones(B, L, dtype=torch.bool)
    ctx = torch.zeros(B, PATCHES, dtype=torch.bool)
    ctx[:, 0] = True
    out = apply_context_mask(mask, ctx)
    assert out[:, :STEPS_PER_PATCH].sum() == 0
    assert out[:, STEPS_PER_PATCH:].all()


def test_apply_context_mask_never_adds_observations():
    _, mask, _ = batch(2)
    mask[:, ::3] = False
    ctx = sample_context_mask(B, PATCHES, gen(3))
    out = apply_context_mask(mask, ctx)
    assert not (out & ~mask).any()


# --- JEPA mask sampling, blueprint §15.1 --------------------------------------------------


def test_mask_ratio_stays_in_the_paper_range():
    ctx = sample_context_mask(4096, PATCHES, gen(11))
    counts = ctx.sum(dim=1)
    assert counts.min() >= 1
    assert counts.max() <= PATCHES - 1
    # round(0.50*24)=12 .. round(0.60*24)=14
    assert set(counts.unique().tolist()) <= {12, 13, 14}


def test_every_sample_keeps_at_least_one_visible_and_one_masked_patch():
    ctx = sample_context_mask(2048, PATCHES, gen(5))
    assert (ctx.sum(dim=1) > 0).all()
    assert (~ctx).sum(dim=1).min() > 0


def test_mask_sampling_is_deterministic_under_a_seeded_generator():
    a = sample_context_mask(16, PATCHES, gen(7))
    b = sample_context_mask(16, PATCHES, gen(7))
    assert torch.equal(a, b)


def test_samples_are_masked_independently():
    ctx = sample_context_mask(64, PATCHES, gen(9))
    assert len({tuple(row.tolist()) for row in ctx}) > 1


# --- density comes from the physical mask, not the online one -----------------------------


def test_loss_density_uses_the_physical_mask():
    """A hidden patch has zero online-visible density. Weighting the contextual loss by that
    would drop every masked patch out of the loss it exists to drive."""
    values, mask, start = batch(4)
    ctx = torch.zeros(B, PATCHES, dtype=torch.bool)
    ctx[:, :12] = True
    enc = eval_encoder()
    with torch.no_grad():
        out = enc(values, apply_context_mask(mask, ctx), start,
                  context_patch_mask=ctx, physical_mask=mask)
    assert out.patch_density[:, :12].min() > 0.99


# --- end to end ---------------------------------------------------------------------------


def test_pretraining_step_produces_finite_losses():
    values, mask, start = batch(6)
    model = OpenCGMStateEvent()
    out = model(values, mask, start, gen(0))
    for t in (out.loss, out.mcr, out.td):
        assert torch.isfinite(t)
    assert out.loss.item() >= 0


def test_target_branch_receives_no_gradients():
    """Blueprint §21.3. A target that learns would collapse both branches together."""
    values, mask, start = batch(7)
    model = OpenCGMStateEvent()
    model(values, mask, start, gen(1)).loss.backward()
    for p in model.target.parameters():
        assert p.grad is None


def test_online_and_target_are_equal_at_initialisation():
    model = OpenCGMStateEvent()
    for op, tp in zip(model.online.parameters(), model.target.target.parameters(), strict=True):
        assert torch.equal(op, tp)


def test_gradients_reach_every_trainable_component():
    values, mask, start = batch(8)
    model = OpenCGMStateEvent()
    model(values, mask, start, gen(2)).loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


@pytest.mark.parametrize("p_observed", [1.0, 0.33, 0.05])
def test_sparse_windows_stay_finite(p_observed):
    """15-minute sources sit at 0.33; a NaN here would poison the batch through attention."""
    torch.manual_seed(0)
    values = torch.randn(B, L) * 30 + 120
    mask = torch.rand(B, L) < p_observed
    model = OpenCGMStateEvent()
    out = model(values, mask, torch.zeros(B, dtype=torch.long), gen(3))
    assert torch.isfinite(out.loss)
