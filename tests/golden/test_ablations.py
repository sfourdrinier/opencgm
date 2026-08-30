"""Tier-1 ablation switches actually ablate. Blueprint §20 Tier 1, items 1-3, 7, 8.

An ablation flag that silently does nothing produces the most expensive kind of wrong result: a
table of variants that all score the same, read as "the component does not matter" when the truth
is "the switch was not wired up". Each test below therefore checks the mechanism, not the flag.

The stream variants are deliberately parameter-identical -- the unused stream's token is zeroed
rather than the fusion narrowed -- so that a measured difference is the information the stream
carries and not a capacity difference confounded with it.
"""

from __future__ import annotations

import torch

from opencgm_stateevent.model.model import OpenCGMStateEvent

VALUES = torch.randn(4, 288) * 30 + 120
MASK = torch.rand(4, 288) > 0.2
CIRCADIAN = torch.tensor([0, 72, 144, 216])


def encoder(**kwargs):
    torch.manual_seed(0)
    return OpenCGMStateEvent(**kwargs).eval()


def test_every_stream_variant_has_the_same_parameter_count():
    """Otherwise the ablation measures capacity, not the stream."""
    counts = {
        s: sum(p.numel() for p in encoder(streams=s).online.parameters() if p.requires_grad)
        for s in ("both", "state", "event", "raw")
    }
    assert len(set(counts.values())) == 1, counts


def test_state_only_and_event_only_produce_different_representations():
    torch.manual_seed(0)
    both = encoder(streams="both")
    state = encoder(streams="state")
    event = encoder(streams="event")
    for other in (state, event):
        other.load_state_dict(both.state_dict())  # identical weights, only the switch differs

    with torch.no_grad():
        a = both.encode(VALUES, MASK, CIRCADIAN).contextual_tokens
        b = state.encode(VALUES, MASK, CIRCADIAN).contextual_tokens
        c = event.encode(VALUES, MASK, CIRCADIAN).contextual_tokens
    assert not torch.allclose(a, b, atol=1e-4), "state-only did not change the representation"
    assert not torch.allclose(a, c, atol=1e-4), "event-only did not change the representation"
    assert not torch.allclose(b, c, atol=1e-4), "the two single-stream variants are identical"


def test_raw_variant_bypasses_the_gaussian_decomposition():
    """The filter is the thing under test, so it must be bypassed, not ignored downstream."""
    torch.manual_seed(0)
    both = encoder(streams="both")
    raw = encoder(streams="raw")
    raw.load_state_dict(both.state_dict())

    with torch.no_grad():
        a = both.encode(VALUES, MASK, CIRCADIAN)
        b = raw.encode(VALUES, MASK, CIRCADIAN)
    assert torch.allclose(b.state_signal, b.event_signal), "raw variant still decomposed"
    assert not torch.allclose(a.state_signal, a.event_signal), "control: 'both' is decomposed"


def test_fixed_sigma_does_not_train_and_stays_at_its_initialisation():
    fixed = encoder(learnable_sigma=False)
    assert not fixed.online.gaussian.rho.requires_grad
    assert fixed.online.gaussian.rho.grad is None

    learned = encoder(learnable_sigma=True)
    assert learned.online.gaussian.rho.requires_grad

    # The bandwidth itself must start in the same place, or the ablation confounds "fixed" with
    # "fixed at a different value".
    assert torch.allclose(fixed.online.gaussian.rho, learned.online.gaussian.rho)


def test_removing_circadian_phase_keeps_patch_position():
    """Dropping both would confound 'no time of day' with 'no ordering'."""
    without = encoder(use_circadian=False)
    with torch.no_grad():
        midnight = without.online.temporal(torch.tensor([0]))
        noon = without.online.temporal(torch.tensor([144]))
    assert torch.allclose(midnight, noon), "circadian phase still reaches the embedding"
    # ...but the 24 patches must still differ from one another.
    assert midnight.squeeze(0).std(dim=0).mean() > 0, "patch position was removed too"

    with_phase = encoder(use_circadian=True)
    with torch.no_grad():
        a = with_phase.online.temporal(torch.tensor([0]))
        b = with_phase.online.temporal(torch.tensor([144]))
    assert not torch.allclose(a, b), "control: phase should matter when enabled"


def test_ablation_flags_are_recorded_in_the_checkpoint_architecture():
    """A variant checkpoint must not be evaluable as if it were the strict model. See D019."""
    from opencgm_stateevent.model.model import ARCHITECTURE_FLAGS, architecture_of

    for flag in ("streams", "learnable_sigma", "use_circadian"):
        assert flag in ARCHITECTURE_FLAGS, f"{flag} would not survive a checkpoint round trip"

    recovered = architecture_of({"config": {"streams": "event", "learnable_sigma": False}})
    assert recovered["streams"] == "event"
    assert recovered["learnable_sigma"] is False
    assert recovered["use_circadian"] is True  # default for a config predating the flag


def test_an_ablation_checkpoint_refuses_to_resume_as_the_strict_model():
    import pytest

    from opencgm_stateevent.train.loop import TrainConfig
    from opencgm_stateevent.train.run import check_architecture

    trained = dict(raw_statistics=True, normalize_targets=False, streams="event",
                   learnable_sigma=True, use_circadian=True, zero_empty_patches=True)
    with pytest.raises(SystemExit, match="refusing to resume"):
        check_architecture({"config": trained}, TrainConfig(raw_statistics=True))
    check_architecture({"config": trained}, TrainConfig(**trained))


def test_training_step_survives_a_gaussian_with_no_gradient():
    """`--fixed-sigma` and `--streams raw` both leave `rho.grad` as None.

    Found by running the ablations, not by reasoning: the metrics code read `rho.grad.abs()`
    unconditionally and two of the nine Tier-1 variants crashed on their first step.
    """
    from opencgm_stateevent.train.loop import TrainConfig, Trainer

    for kwargs in ({"learnable_sigma": False}, {"streams": "raw"}):
        cfg = TrainConfig(device="cpu", deterministic=False, num_workers=0, batch_size=4)
        trainer = Trainer(model=OpenCGMStateEvent(**kwargs), cfg=cfg, steps_per_epoch=10)
        batch = (
            torch.randn(4, 288) * 30 + 120,
            torch.rand(4, 288) > 0.2,
            torch.tensor([0, 72, 144, 216]),
        )
        metrics = trainer.step(batch, 0, 0)
        assert metrics.loss == metrics.loss, f"loss is NaN for {kwargs}"


def test_a_fully_unobserved_patch_contributes_no_statistics_signal():
    """Appendix C.2: "Empty patches are zeroed by the validity mask". D020.

    Zeroing the inputs is not enough. The projection carries a bias and the activation is GELU,
    so an empty patch emitted `gelu(bias)` -- a learned, non-zero, input-independent vector that
    the Transformer could not tell apart from a real measurement. Measured L1 norm before the
    fix: 8.34.
    """
    zeroed = encoder(zero_empty_patches=True)
    legacy = encoder(zero_empty_patches=False)
    legacy.load_state_dict(zeroed.state_dict())

    mask = torch.ones(1, 288, dtype=torch.bool)
    mask[0, :12] = False  # patch 0 entirely unobserved
    valid = mask.reshape(1, 24, 12).any(dim=-1)
    empty_stats = torch.zeros(1, 24)

    with torch.no_grad():
        fixed = zeroed.online.state_embedder.statistics(empty_stats, empty_stats, valid)
        before = legacy.online.state_embedder.statistics(empty_stats, empty_stats, valid)

    assert float(fixed[0, 0].abs().sum()) == 0.0, "empty patch still emits a learned bias"
    assert float(before[0, 0].abs().sum()) > 1.0, "control: the old behaviour was non-zero"
    assert float(fixed[0, 5].abs().sum()) > 0.0, "an observed patch was zeroed too"


def test_zero_empty_patches_is_recorded_in_the_checkpoint():
    from opencgm_stateevent.model.model import ARCHITECTURE_FLAGS, architecture_of

    assert ARCHITECTURE_FLAGS["zero_empty_patches"] is False, (
        "the fallback must be the pre-D020 behaviour, or old checkpoints are re-read wrongly"
    )
    assert architecture_of({"config": {}})["zero_empty_patches"] is False
    assert architecture_of({"config": {"zero_empty_patches": True}})["zero_empty_patches"] is True


def test_dense_interpolation_fills_gaps_and_declares_them_observed():
    """Tier-1 ablation 6, and the only interpolation in this codebase.

    The standing rule is that CGM is never interpolated and the physical mask is authoritative.
    This switch violates it deliberately so the cost of the rule can be measured. What it must
    not do is fabricate beyond the data: edges are held, not extrapolated.
    """
    import numpy as np

    from opencgm_stateevent.train.dataset import interpolate_dense

    values = np.array([100.0, 0, 0, 130.0, 0, 160.0], dtype=np.float32)
    mask = np.array([True, False, False, True, False, True])
    filled, dense = interpolate_dense(values, mask)

    assert np.allclose(filled, [100, 110, 120, 130, 145, 160])
    assert dense.all()
    assert np.allclose(filled[mask], values[mask]), "observed values were altered"

    # Leading and trailing gaps are held at the nearest observation, not extrapolated.
    edge_values = np.array([0.0, 0.0, 120.0, 0.0], dtype=np.float32)
    edge_mask = np.array([False, False, True, False])
    edge_filled, _ = interpolate_dense(edge_values, edge_mask)
    assert np.allclose(edge_filled, 120.0)

    # A window with nothing in it cannot be invented into existence.
    empty = np.zeros(5, dtype=np.float32)
    none = np.zeros(5, dtype=bool)
    _, still_none = interpolate_dense(empty, none)
    assert not still_none.any(), "an entirely unobserved window was declared observed"


def test_dense_interpolation_is_off_by_default():
    """It violates a project non-negotiable, so it must never be reachable by accident."""
    from opencgm_stateevent.train.loop import TrainConfig

    cfg = TrainConfig()
    assert cfg.dense_interpolation is False
    assert cfg.augment is True
    assert cfg.exclude_dataset == ""
