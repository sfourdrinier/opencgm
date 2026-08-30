"""Parity test for the streams ONNX artifact (`artifacts/glucofm_encoder_streams.onnx`).

The dual-stream decomposition is the one component the Tier-1 ablation shows to be
load-bearing (event-only costs -0.0504 ROC-AUC, ~6 sigma), and this artifact is what lets a
browser render it. The test pins the ONNX `state_signal` / `event_signal` outputs to the
PyTorch model, the same way `test_onnx_parity.py` pins the embedding.

What the two streams mean numerically, for whoever plots them:

  - Units: per-window z-scores, NOT mg/dL. The input is masked-instance-normalized
    (observed-only mean/std per window) before the causal Gaussian filter runs, and both
    streams live on that normalized scale.
  - Sum: at observed positions `state_signal + event_signal` reconstructs the normalized
    input exactly; the streams do NOT sum to the raw mg/dL trace.
  - Unobserved positions: `event_signal` is exactly 0 (the residual is multiplied by the
    mask). `state_signal` holds the causal filter's estimate from past observed samples
    inside its 36-step radius, and 0 where that window contains no observation — it is an
    extrapolation, not data, so plot it accordingly.

The heavily-masked cases matter most: every statistic, the filter and the normalization must
see the physical observation mask first, and empty patches must follow the checkpoint's
`zero_empty_patches` flag (False for the released checkpoint — read back, not assumed). A
fully-observed-only test would pass with that ordering broken.

Marked `@pytest.mark.golden`. CPU only. Skips if either artefact is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = REPO_ROOT / "artifacts" / "glucofm_encoder_streams.onnx"
CKPT_PATH = REPO_ROOT / "runs_5090" / "rawstats120" / "ckpt_ep040.pt"

EMBED_TOL = 1e-4  # same gate as test_onnx_parity.py: 3 transformer layers of float32 noise
STREAM_TOL = 1e-5  # streams bypass the transformer; conv + norm only, so tighter


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"{path.name} not generated; run scripts/export_encoder_onnx.py --streams first"
        )


def _load_backends():
    import onnxruntime as ort

    from opencgm_stateevent.eval.embed import load_encoder
    from scripts.export_encoder_onnx import EncoderStreamsEmbed, replace_transformer_for_onnx

    # Same transformer swap as the export script, so embedding differences measure the export
    # and not the fused-vs-decomposed MHA implementations. The streams never reach the
    # transformer, so for them the swap is irrelevant either way.
    model, _ref = load_encoder(CKPT_PATH, device="cpu")
    replace_transformer_for_onnx(model)
    wrapper = EncoderStreamsEmbed(model).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return wrapper, sess


def _run_both(wrapper, sess, values: np.ndarray, mask: np.ndarray, circ: np.ndarray):
    import torch

    with torch.no_grad():
        pt = wrapper(
            torch.from_numpy(values), torch.from_numpy(mask), torch.from_numpy(circ)
        )
    pt = tuple(t.numpy() for t in pt)
    ox = sess.run(
        ["embedding", "state_signal", "event_signal"],
        {"values": values, "mask": mask, "circadian_start": circ},
    )
    return pt, ox


def _assert_parity(pt, ox, label: str) -> None:
    for name, tol, a, b in [
        ("embedding", EMBED_TOL, pt[0], ox[0]),
        ("state_signal", STREAM_TOL, pt[1], ox[1]),
        ("event_signal", STREAM_TOL, pt[2], ox[2]),
    ]:
        assert a.shape == b.shape, f"{label}/{name}: shape {a.shape} vs {b.shape}"
        assert np.isfinite(b).all(), f"{label}/{name}: ONNX output contains non-finite values"
        abs_err = float(np.abs(a - b).max())
        assert abs_err < tol, f"{label}/{name}: max |pt - onnx| = {abs_err:.2e} exceeds {tol}"


@pytest.mark.golden
def test_streams_onnx_matches_pytorch() -> None:
    """Dense-ish windows: streams and embedding must match PyTorch."""
    _skip_if_missing(ONNX_PATH)
    _skip_if_missing(CKPT_PATH)
    wrapper, sess = _load_backends()

    rng = np.random.default_rng(17)
    values = rng.uniform(60.0, 250.0, size=(16, 288)).astype(np.float32)
    mask = (rng.uniform(0.0, 1.0, size=(16, 288)) < 0.85).astype(np.float32)
    circ = rng.integers(0, 288, size=(16,)).astype(np.int64)

    pt, ox = _run_both(wrapper, sess, values, mask, circ)
    assert ox[0].shape == (16, 128) and ox[1].shape == ox[2].shape == (16, 288)
    _assert_parity(pt, ox, "dense")


@pytest.mark.golden
def test_streams_onnx_with_substantial_missing_data() -> None:
    """Heavily-masked windows: gaps, empty patches, sparse sampling.

    This is where the mask ordering breaks silently. The batch covers a contiguous 6-hour gap
    (patches 8-13 fully empty — exercises the checkpoint's `zero_empty_patches` behaviour
    through the embedding), an unobserved leading 5 hours (the causal filter has no past to
    draw on, `state_signal` must go through its den->0 guard), and 20% Bernoulli sampling
    (roughly a 15-minute source with dropout).
    """
    _skip_if_missing(ONNX_PATH)
    _skip_if_missing(CKPT_PATH)
    wrapper, sess = _load_backends()

    rng = np.random.default_rng(29)
    values = rng.uniform(70.0, 220.0, size=(4, 288)).astype(np.float32)
    mask = np.ones((4, 288), dtype=np.float32)
    mask[0] = (rng.uniform(size=288) < 0.85).astype(np.float32)
    mask[0, 96:168] = 0.0  # 6h sensor gap: whole patches empty
    mask[1, :60] = 0.0  # nothing observed before 05:00
    mask[1, 60:] = (rng.uniform(size=228) < 0.7).astype(np.float32)
    mask[2] = (rng.uniform(size=288) < 0.20).astype(np.float32)  # very sparse
    mask[3] = (rng.uniform(size=288) < 0.50).astype(np.float32)
    circ = rng.integers(0, 288, size=(4,)).astype(np.int64)

    pt, ox = _run_both(wrapper, sess, values, mask, circ)
    _assert_parity(pt, ox, "masked")

    # Pin the documented semantics on the ONNX outputs themselves.
    _, state, event = ox
    assert (event[mask == 0.0] == 0.0).all(), "event_signal must be exactly 0 where unobserved"

    import torch

    from opencgm_stateevent.model.causal_gaussian import masked_instance_norm

    normalized = masked_instance_norm(torch.from_numpy(values), torch.from_numpy(mask)).numpy()
    recon_err = np.abs((state + event - normalized)[mask == 1.0]).max()
    assert recon_err < STREAM_TOL, (
        f"state + event must reconstruct the normalized input at observed positions; "
        f"max err {recon_err:.2e}"
    )


@pytest.mark.golden
def test_streams_sidecar_matches_artifact() -> None:
    """The sidecar must describe this exact file and the checkpoint's own flags."""
    _skip_if_missing(ONNX_PATH)
    _skip_if_missing(CKPT_PATH)

    import torch

    from opencgm_stateevent.model.model import architecture_of
    from scripts.export_encoder_onnx import sha256_of

    meta = json.loads((ONNX_PATH.parent / f"{ONNX_PATH.name}.meta.json").read_text())
    assert meta["sha256"] == sha256_of(ONNX_PATH)
    assert meta["output_names"] == ["embedding", "state_signal", "event_signal"]

    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    assert meta["architecture"] == architecture_of(state)
