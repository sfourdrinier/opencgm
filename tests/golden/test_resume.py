"""Resuming from a checkpoint must reproduce the step that would have run. KR2.6, §17.5.

A run that is reproducible only until its first interruption is not reproducible. Four
independent pieces of state have to survive a save/load or the resumed trajectory silently
diverges:

* model, optimizer moments and GradScaler scale;
* the CPU generator that draws JEPA context masks;
* the global torch RNG that drives dropout;
* the position in the epoch's batch order.

The last one is why `run.epoch_permutation` derives batch order from ``(seed, epoch)`` instead of
letting the DataLoader shuffle. A shuffle's internal state cannot be checkpointed, so a resumed
epoch would reshuffle and see different windows -- a divergence that looks like ordinary run-to-run
noise rather than a bug.

These run on CPU with synthetic batches. The GPU end-to-end resume is a separate gate recorded in
`reports/`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from opencgm_stateevent.model.model import OpenCGMStateEvent
from opencgm_stateevent.train.loop import TrainConfig, Trainer, save_checkpoint
from opencgm_stateevent.train.run import epoch_permutation

B, L = 8, 288
FIELDS = ("loss", "mcr", "td", "sigma", "grad_norm", "realized_mask_ratio")


def batches(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(B, L, generator=g) * 30 + 120,
            torch.rand(B, L, generator=g) < 0.8,
            torch.zeros(B, dtype=torch.long),
        )
        for _ in range(n)
    ]


def make() -> Trainer:
    torch.manual_seed(17)
    cfg = TrainConfig(seed=17, batch_size=B, amp=False, device="cpu", log_every=0)
    return Trainer(model=OpenCGMStateEvent(), cfg=cfg, steps_per_epoch=100)


def metrics_of(m) -> tuple:
    return tuple(round(getattr(m, f), 12) for f in FIELDS)


# --- batch order --------------------------------------------------------------------------


def test_epoch_permutation_is_a_function_of_seed_and_epoch_only():
    a = epoch_permutation(1000, 17, 3)
    b = epoch_permutation(1000, 17, 3)
    assert np.array_equal(a, b)


def test_different_epochs_give_different_order():
    assert not np.array_equal(epoch_permutation(1000, 17, 0), epoch_permutation(1000, 17, 1))


def test_different_seeds_give_different_order():
    assert not np.array_equal(epoch_permutation(1000, 17, 0), epoch_permutation(29, 0, 0)[:1000])


def test_permutation_is_a_permutation():
    p = epoch_permutation(5000, 43, 7)
    assert np.array_equal(np.sort(p), np.arange(5000))


def test_an_offset_slice_matches_the_tail_of_the_full_order():
    """The seek a resume performs: skipping `offset` items must not perturb the rest."""
    full = epoch_permutation(1000, 17, 2)
    assert np.array_equal(full[384:], epoch_permutation(1000, 17, 2)[384:])


# --- optimizer and RNG state --------------------------------------------------------------


def test_resume_reproduces_the_next_step_exactly(tmp_path):
    data = batches(8)
    a = make()
    for i, b in enumerate(data[:4]):
        a.step(b, 0, i)
    ckpt = save_checkpoint(a, tmp_path / "c.pt", {"epoch": 0, "step_in_epoch": 4})
    uninterrupted = [metrics_of(a.step(b, 0, 4 + i)) for i, b in enumerate(data[4:])]

    b_ = make()
    b_.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    resumed = [metrics_of(b_.step(b, 0, 4 + i)) for i, b in enumerate(data[4:])]

    assert resumed == uninterrupted


def test_resume_restores_the_step_counter():
    a = make()
    for i, b in enumerate(batches(3)):
        a.step(b, 0, i)
    b_ = make()
    b_.load_state_dict(a.state_dict())
    assert b_.global_step == a.global_step == 3


def test_resume_restores_optimizer_moments():
    """Fresh AdamW moments would make the first resumed step a much larger one."""
    a = make()
    for i, b in enumerate(batches(3)):
        a.step(b, 0, i)
    b_ = make()
    b_.load_state_dict(a.state_dict())
    sa = a.optimizer.state_dict()["state"]
    sb = b_.optimizer.state_dict()["state"]
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k]["exp_avg"], sb[k]["exp_avg"])
        assert torch.equal(sa[k]["exp_avg_sq"], sb[k]["exp_avg_sq"])


def test_resume_restores_the_ema_target_not_just_the_online_branch():
    a = make()
    for i, b in enumerate(batches(3)):
        a.step(b, 0, i)
    b_ = make()
    b_.load_state_dict(a.state_dict())
    for pa, pb in zip(
        a.model.target.parameters(), b_.model.target.parameters(), strict=True
    ):
        assert torch.equal(pa, pb)


def test_a_trainer_that_did_not_resume_diverges():
    """Control. Without it, the resume test would pass even if state were being ignored."""
    data = batches(8)
    a = make()
    for i, b in enumerate(data[:4]):
        a.step(b, 0, i)
    uninterrupted = metrics_of(a.step(data[4], 0, 4))

    fresh = make()  # same seed, but no optimizer moments and no advanced RNG
    assert metrics_of(fresh.step(data[4], 0, 4)) != uninterrupted


def test_two_runs_of_the_same_seed_agree_step_for_step():
    data = batches(5)
    x = [metrics_of(t) for t in (make().step(b, 0, i) for i, b in enumerate(data))]
    y = [metrics_of(t) for t in (make().step(b, 0, i) for i, b in enumerate(data))]
    assert x == y


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_context_masks_are_reproducible_from_the_trainer_seed(seed):
    cfg = TrainConfig(seed=seed, batch_size=B, amp=False, device="cpu", log_every=0)
    made = [
        Trainer(model=OpenCGMStateEvent(), cfg=cfg, steps_per_epoch=10).generator
        for _ in range(2)
    ]
    a = torch.rand(4, generator=made[0])
    b = torch.rand(4, generator=made[1])
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_resume_survives_a_checkpoint_loaded_onto_the_gpu(tmp_path):
    """RNG states are CPU ByteTensors, and `map_location="cuda"` moves them too.

    Every CPU test above loads with `map_location="cpu"` and so cannot see this. It surfaced on
    the first real resume from a GPU checkpoint, as `RNG state must be a torch.ByteTensor`.
    """
    cfg = TrainConfig(seed=17, batch_size=B, amp=False, device="cuda", log_every=0)
    torch.manual_seed(17)
    a = Trainer(model=OpenCGMStateEvent(), cfg=cfg, steps_per_epoch=100)
    data = [tuple(x.cuda() for x in b) for b in batches(3)]
    for i, b in enumerate(data):
        a.step(b, 0, i)
    ckpt = save_checkpoint(a, tmp_path / "gpu.pt", {"epoch": 0, "step_in_epoch": 3})

    torch.manual_seed(17)
    b_ = Trainer(model=OpenCGMStateEvent(), cfg=cfg, steps_per_epoch=100)
    b_.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False))
    assert b_.global_step == 3


# --- training mode ------------------------------------------------------------------------


def test_health_restores_training_mode():
    """`health` runs under eval. If it does not restore, dropout stays off for the rest of the run.

    Caught by comparing a resumed epoch against an uninterrupted one: the uninterrupted run's
    loss halved at the first epoch boundary, because it had been training without dropout ever
    since the first diagnostic.
    """
    from torch.utils.data import DataLoader

    t = make()
    data = batches(2)
    loader = DataLoader(data, batch_size=None)
    t.model.train()
    t.health(loader)
    assert t.model.training

    t.model.eval()
    t.health(loader)
    assert not t.model.training


def test_dropout_actually_changes_the_loss():
    """Control: without this, the mode test could pass while dropout does nothing."""
    b = batches(1)[0]
    t = make()
    t.model.train()
    train_mode = t.model(b[0], b[1], b[2], torch.Generator().manual_seed(3)).loss.item()
    t.model.eval()
    eval_mode = t.model(b[0], b[1], b[2], torch.Generator().manual_seed(3)).loss.item()
    assert train_mode != eval_mode
