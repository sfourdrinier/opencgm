"""Three defects that let a collapsing model look healthy.

Found by an independent review after downstream probes showed a trained encoder scoring *below*
its random initialisation. Each was invisible on its own, and together they made the failure
unreadable: the collapse metric said "improving", the training target was noise, and the baseline
being compared against was quietly handicapped.

* **effective rank was computed from singular values, not covariance eigenvalues.** Covariance
  eigenvalues are the squares. On data where one direction carries 99.9% of the variance the wrong
  form reports 4.37 and the right form reports 1.02. This is the one that mattered most: it turned
  a severe dimensional collapse into a number that appeared to be recovering.
* **the EMA teacher ran with dropout active.** `torch.no_grad()` blocks gradients but not dropout,
  and the target is an ordinary child module, so `model.train()` reached it. Two identical target
  forwards differed by RMS 0.44 — the regression target was stochastic.
* **two clinical baseline features were identically zero everywhere,** because `np.percentile`
  propagates NaN and every window has unobserved positions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from opencgm_stateevent.eval.baselines import clinical_metrics
from opencgm_stateevent.eval.windows import WindowSet
from opencgm_stateevent.model.model import OpenCGMStateEvent
from opencgm_stateevent.train.loop import spectrum

B, L = 32, 288


def one_dominant_direction(n: int = 512, d: int = 64) -> torch.Tensor:
    """One direction carries essentially all the variance. True effective rank is ~1."""
    scale = torch.tensor([10.0] + [0.05] * (d - 1))
    return torch.randn(n, d) @ torch.diag(scale)


# --- effective rank -------------------------------------------------------------------------


def test_rank_is_one_when_one_direction_carries_all_variance():
    rank, top1, _ = spectrum(one_dominant_direction())
    assert rank == pytest.approx(1.0, abs=0.2)
    assert top1 > 0.99


def test_singular_value_form_would_overstate_rank():
    """Control pinning the original defect, so the fix cannot be reverted silently."""
    x = one_dominant_direction()
    xc = x - x.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(xc)
    p = sv / sv.sum()
    wrong = float(torch.exp(-(p * p.log()).sum()))
    assert wrong > 3.0, "the buggy form should look healthy on collapsed data"
    assert spectrum(x)[0] < 1.5


def test_rank_is_full_for_isotropic_data():
    rank, top1, _ = spectrum(torch.randn(4096, 16))
    assert rank > 13.0
    assert top1 < 0.25


def test_rank_is_monotone_in_concentration():
    """Rank must fall as variance concentrates into fewer directions."""
    ranks = []
    for decay in (1.0, 0.5, 0.2, 0.05):
        scale = torch.tensor([decay**i for i in range(32)])
        ranks.append(spectrum(torch.randn(2048, 32) @ torch.diag(scale))[0])
    assert ranks == sorted(ranks, reverse=True), ranks


def test_top_fractions_are_ordered_and_bounded():
    _, top1, top4 = spectrum(torch.randn(512, 32))
    assert 0.0 < top1 <= top4 <= 1.0


# --- the EMA teacher ------------------------------------------------------------------------


def gpu_or_cpu(model: OpenCGMStateEvent):
    return model.cuda() if torch.cuda.is_available() else model


def test_the_target_stays_in_eval_mode_when_the_model_is_trained():
    model = OpenCGMStateEvent()
    model.train()
    assert model.training
    assert not model.target.target.training, "teacher inherited train mode; dropout is active"


def test_the_target_is_deterministic():
    """The property that matters: a regression target must not be noise."""
    torch.manual_seed(0)
    model = gpu_or_cpu(OpenCGMStateEvent())
    model.train()
    device = next(model.parameters()).device
    values = (torch.randn(B, L) * 30 + 120).to(device)
    mask = (torch.rand(B, L) < 0.8).to(device)
    circadian = torch.zeros(B, dtype=torch.long, device=device)
    with torch.no_grad():
        a = model.target.target(values, mask, circadian).contextual_tokens
        b = model.target.target(values, mask, circadian).contextual_tokens
    assert torch.equal(a, b)


def test_the_online_branch_keeps_its_dropout():
    """The fix must not disable dropout everywhere — only in the teacher."""
    model = OpenCGMStateEvent()
    model.train()
    assert model.online.training
    dropouts = [m for m in model.online.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropouts, "expected dropout in the online encoder"
    assert all(m.training for m in dropouts)


def test_switching_to_eval_and_back_leaves_the_target_in_eval():
    model = OpenCGMStateEvent()
    model.eval()
    model.train()
    assert not model.target.target.training


def test_the_target_still_tracks_the_online_branch_after_the_fix():
    """Forcing eval mode must not break the EMA update itself."""
    model = OpenCGMStateEvent()
    before = model.target.target.gaussian.rho.detach().clone()
    with torch.no_grad():
        model.online.gaussian.rho.add_(1.0)
    model.target.update(model.online, 0.5)
    assert not torch.equal(model.target.target.gaussian.rho, before)


# --- the clinical baseline ------------------------------------------------------------------


def synthetic_windows(n: int = 16, density: float = 0.33) -> WindowSet:
    """Sparse windows spanning a realistic glycemic range.

    Per-window means are spread from hypo to hyper so that every clinical range fraction —
    including the >250 mg/dL band — is exercised. A narrow fixture would leave the extreme bands
    constant at zero for a legitimate reason and mask a genuinely dead feature.
    """
    rng = np.random.default_rng(0)
    centres = rng.uniform(60, 280, size=(n, 1))
    values = (centres + rng.normal(0, 45, size=(n, L))).clip(40, 400).astype(np.float32)
    mask = rng.random((n, L)) < density
    mask[:, 0] = True  # every window has at least one observation
    return WindowSet(
        source="synthetic", values=values, mask=mask,
        circadian=np.zeros(n, dtype=np.int64),
        subjects=np.array([f"s{i}" for i in range(n)], dtype=object),
        sessions=np.array(["x"] * n, dtype=object),
    )


def test_quartile_features_are_not_all_zero_on_sparse_windows():
    features = clinical_metrics(synthetic_windows())
    q25, q75 = features[:, 13], features[:, 14]
    assert not (q25 == 0).all(), "25th percentile feature is dead"
    assert not (q75 == 0).all(), "75th percentile feature is dead"


def test_quartiles_are_ordered_and_physiological():
    features = clinical_metrics(synthetic_windows())
    q25, q75 = features[:, 13], features[:, 14]
    assert (q25 <= q75).all()
    assert (q25 > 20).all() and (q75 < 400).all()


def test_no_clinical_feature_is_constant_across_windows():
    """A dead feature is a column with no variance. Catches the whole class."""
    features = clinical_metrics(synthetic_windows(n=64))
    dead = [i for i in range(features.shape[1]) if np.ptp(features[:, i]) == 0]
    assert not dead, f"constant clinical features at columns {dead}"
