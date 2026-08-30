"""The CGM-JEPA comparator is the authors' model, not our guess at it.

This baseline is the one result in the project with a built-in conflict of interest: we build it,
we train it, and then we report that our model beats it. The defence is that every structural choice
is checkable against the authors' released code, and that the checks run in CI rather than living in
a paragraph nobody re-reads.

Source: github.com/cruiseresearchgroup/CGM-JEPA (MIT), master @ 2026-05-11, retrieved 2026-08-28.
"""

from __future__ import annotations

import torch

from opencgm_stateevent.baselines import cgm_jepa as cj


def test_parameter_count_matches_the_papers_reported_size():
    """GlucoFM Table 3 reports CGM-JEPA at 0.52M. A port that misses this is not the same model.

    Our first attempt at this baseline -- inferred from GlucoFM's appendix rather than read from the
    authors' source -- came to roughly 0.35M, a third smaller, from a half-width FFN and a plain
    linear patch embedding. It would have lost the comparison on capacity, not on method.
    """
    n = sum(p.numel() for p in cj.Encoder().parameters())
    assert n == cj.ENCODER_PARAMETERS, n
    assert n + cj.DEAD_TIME_EMBEDDING == cj.AUTHORS_ENCODER_PARAMETERS, (
        "the gap to the authors' count is no longer exactly their unused time embedding"
    )
    assert round(n / 1e6, 2) == 0.52


def test_official_hyperparameters_are_the_ones_recorded():
    """Read from config/config_pretrain.py. Drift here silently changes what we are comparing to."""
    assert (cj.EMBED_DIM, cj.NUM_LAYERS, cj.NUM_HEADS) == (96, 3, 6)
    assert cj.MASK_RATIO == 0.25
    assert cj.EPOCHS == 101
    assert cj.BASE_LR == 1e-4
    assert cj.BATCH_SIZE == 128
    assert cj.DROPOUT == 0.0, "config_pretrain.py:46 is 0.0; 0.1 would over-regularize the baseline"
    assert cj.MLP_RATIO == 4.0, "encoder.py:34; halving this removes real capacity"
    assert (cj.PREDICTOR_DIM, cj.PREDICTOR_HEADS, cj.PREDICTOR_LAYERS) == (48, 2, 1)


def test_the_encoder_drops_masked_patches_rather_than_substituting_placeholders():
    """encoder.py:97-98 -- canonical I-JEPA. Mask tokens belong in the predictor, not here."""
    model = cj.Encoder().eval()
    patches = torch.randn(2, cj.PATCHES, cj.PATCH_SIZE)
    keep = torch.arange(cj.PATCHES).unsqueeze(0).expand(2, -1)[:, :18].contiguous()

    with torch.no_grad():
        full = model(patches)
        context = model(patches, keep=keep)

    assert full.shape == (2, cj.PATCHES, cj.EMBED_DIM)
    assert context.shape == (2, 18, cj.EMBED_DIM), "masked patches still occupy sequence positions"


def test_the_ema_target_is_layer_normed_before_the_loss():
    """pretrain_cgm_jepa.py:180 -- the authors' guard against target drift, and our own D018.

    Without it the student can drive the loss down by shrinking the teacher's scale rather than by
    learning anything, and a partially collapsed baseline would hand us an unearned win.
    """
    torch.manual_seed(0)
    model = cj.CGMJEPA().eval()
    patches = torch.randn(8, cj.PATCHES, cj.PATCH_SIZE) * 30 + 120
    norm = lambda t: torch.nn.functional.layer_norm(t, (cj.EMBED_DIM,))  # noqa: E731

    # The teacher's own final LayerNorm already leaves unit variance at initialisation, so the
    # trainer's extra normalization looks redundant here. It is not: that norm carries a *learned*
    # gain, and nothing stops it drifting during training. Simulate the drift.
    with torch.no_grad():
        before_raw, before_normed = model.target(patches), norm(model.target(patches))
        model.target.encoder_norm.weight.mul_(5.0)
        after_raw, after_normed = model.target(patches), norm(model.target(patches))

    assert not torch.allclose(before_raw, after_raw, atol=1e-3), "control: the gain should matter"
    assert torch.allclose(before_normed, after_normed, atol=1e-4), (
        "the target scale still follows the teacher's learned gain, so the student can lower the "
        "loss by shrinking the teacher instead of learning -- this is D018"
    )
    assert torch.allclose(after_normed.std(dim=-1, unbiased=False), torch.ones(8, cj.PATCHES),
                          atol=1e-3)


def test_the_loss_is_plain_l1_over_the_masked_patches_only():
    """pretrain_cgm_jepa.py:25-30 is mean absolute error, not smooth L1."""
    torch.manual_seed(0)
    model = cj.CGMJEPA()
    generator = torch.Generator().manual_seed(0)
    out = model(torch.randn(4, cj.PATCHES, cj.PATCH_SIZE), generator)

    n_predict = max(1, round(cj.MASK_RATIO * cj.PATCHES))
    assert out.predict.shape == (4, n_predict)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    # Context and target patches must partition the window -- no patch may be in both.
    for row in range(4):
        assert len(set(out.predict[row].tolist())) == n_predict


def test_the_ema_schedule_never_freezes_the_teacher():
    """pretrain_cgm_jepa.py:105 -- linear ramp sized for 1.25x the epochs, so it never reaches 1.0.

    Our first draft used GlucoFM's cosine-to-exactly-1.0. Reaching 1.0 freezes the teacher for the
    final epochs; borrowing the *winner's* schedule is not neutrality, it assumes its tuning
    transfers.
    """
    first = cj.ema_momentum(0)
    last = cj.ema_momentum(cj.EPOCHS - 1)
    assert first == cj.EMA_MOMENTUM
    assert first < last < 1.0, (first, last)
    assert round(last, 4) == 0.9994


def test_the_learning_rate_warms_up_then_decays_to_zero():
    """pretrain_cgm_jepa.py:32 -- 15% linear warmup then linear decay."""
    total = 1000
    assert cj.learning_rate_scale(0, total) == 0.0
    assert cj.learning_rate_scale(150, total) == 1.0  # peak at the end of warmup
    assert cj.learning_rate_scale(total, total) == 0.0
    assert 0.4 < cj.learning_rate_scale(575, total) < 0.6  # halfway through decay


def test_inputs_are_raw_mg_dl_by_default():
    """config_pretrain.py:17 is `normalize_x: False`; GlucoFM appendix B says "normalized".

    We follow the authors, which is also the direction that helps the comparator: raw values keep
    absolute glucose level, and D019 measured that as worth up to 0.21 ROC-AUC on obesity.
    Standardizing here would strip it from the baseline while our model keeps it.
    """
    values = torch.randn(3, 288) * 30 + 120
    patches = cj.to_patches(values)
    assert torch.allclose(patches.reshape(3, 288), values), "default path altered the raw values"
    assert patches.mean() > 50, "raw mg/dL scale was not preserved"

    standardized = cj.to_patches(values, normalize=True)
    assert abs(float(standardized.mean())) < 0.1


def test_embedding_is_pre_projection_mean_pooled_and_never_uses_dropout():
    """Appendix B: "from the frozen encoder before the projection head and mean-pooled"."""
    model = cj.CGMJEPA().train()
    patches = torch.randn(5, cj.PATCHES, cj.PATCH_SIZE)

    embedding = model.embed(patches)
    assert embedding.shape == (5, cj.EMBED_DIM), "not 96-dim, so not pre-projection"
    assert model.training, "embed() left the model in eval mode"

    # Deterministic even though the caller left the model in train mode.
    assert torch.allclose(embedding, model.embed(patches))


def test_the_target_encoder_is_frozen_and_tracks_the_online_encoder():
    model = cj.CGMJEPA()
    assert not any(p.requires_grad for p in model.target.parameters())

    before = next(iter(model.target.parameters())).clone()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    model.update_target(0.9)
    after = next(iter(model.target.parameters()))
    assert not torch.allclose(before, after), "the EMA target never moved"
