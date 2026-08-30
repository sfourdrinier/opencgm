"""A checkpoint must be evaluated with the front end it was trained with. §19.2.

The failure this pins is silent by construction. `raw_statistics` changes what
`ContextEncoder.tokenize` computes but adds and removes no parameter tensor, so a model
built from the current defaults loads an older checkpoint under `strict=True` without a
word of complaint and then produces different embeddings from identical weights. Measured
on a real checkpoint that moved a headline probe by 0.025, which is larger than every
effect this project is trying to detect.

Two things therefore have to hold: the flags are read back from the checkpoint, and they
are part of the embedding cache key so that pre- and post-change runs cannot collide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from opencgm_stateevent.eval.embed import EncoderRef
from opencgm_stateevent.model.model import ARCHITECTURE_FLAGS, OpenCGMStateEvent, architecture_of
from opencgm_stateevent.model.reference import PATCHES, STEPS_PER_PATCH


def _ref(**overrides) -> EncoderRef:
    base = dict(
        checkpoint=Path("x.pt"), weights_sha256="w", epoch=1, seed=17, git_sha="g",
        windows_sha256="s",
    )
    return EncoderRef(**{**base, **overrides})


def test_missing_flag_falls_back_to_the_pre_change_behaviour():
    """A checkpoint written before the flag existed was trained without it."""
    assert architecture_of({"config": {"seed": 17}}) == ARCHITECTURE_FLAGS
    assert ARCHITECTURE_FLAGS["raw_statistics"] is False


def test_flag_is_read_from_the_checkpoint_not_the_current_default():
    assert architecture_of({"config": {"raw_statistics": True}})["raw_statistics"] is True
    assert architecture_of({"config": {"raw_statistics": False}})["raw_statistics"] is False


def test_architecture_participates_in_the_cache_key():
    a = _ref(architecture=json.dumps({"raw_statistics": False}))
    b = _ref(architecture=json.dumps({"raw_statistics": True}))
    assert a.tag != b.tag, "two front ends would share one cache entry"


def test_the_flag_actually_changes_the_embedding():
    """Guards the premise. If this ever stops being true the test above is theatre."""
    torch.manual_seed(0)
    raw = OpenCGMStateEvent(raw_statistics=True).eval()
    normalized = OpenCGMStateEvent(raw_statistics=False).eval()
    normalized.load_state_dict(raw.state_dict())  # identical weights, by construction

    rng = np.random.default_rng(0)
    shape = (4, PATCHES * STEPS_PER_PATCH)
    values = torch.from_numpy(rng.normal(120, 40, shape).astype(np.float32))
    mask = torch.from_numpy(rng.random(shape) < 0.8)
    circadian = torch.zeros(4, dtype=torch.long)

    with torch.no_grad():
        a = raw.encode(values, mask, circadian).contextual_tokens
        b = normalized.encode(values, mask, circadian).contextual_tokens
    assert not torch.allclose(a, b, atol=1e-4)


def test_resume_into_a_different_front_end_is_refused():
    """Half a run of one architecture and half of another is not a result. See D019.

    The loss curve stays continuous across such a resume, so nothing downstream reveals it.
    """
    import pytest

    from opencgm_stateevent.train.loop import TrainConfig
    from opencgm_stateevent.train.run import check_architecture

    legacy = dict(raw_statistics=False, normalize_targets=False, streams="both",
                  learnable_sigma=True, use_circadian=True, zero_empty_patches=False)
    trained_normalized = {"config": legacy}
    with pytest.raises(SystemExit, match="refusing to resume"):
        check_architecture(trained_normalized, TrainConfig(raw_statistics=True))

    check_architecture(trained_normalized, TrainConfig(**legacy))


def test_a_config_predating_the_flags_resumes_under_the_old_behaviour():
    from opencgm_stateevent.train.loop import TrainConfig
    from opencgm_stateevent.train.run import check_architecture

    # Every flag must be given its pre-flag value, which is what a config that predates them
    # implies. Defaulting any of them to today's value would silently change the run.
    check_architecture(
        {"config": {"seed": 29}},
        TrainConfig(raw_statistics=False, zero_empty_patches=False),
    )
