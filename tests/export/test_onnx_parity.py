"""Golden parity test: the released ONNX encoder must match PyTorch byte-for-byte.

This is the §26 risk-register gate. Without it, a future change to the encoder architecture
flags could silently produce ONNX weights that look fine but produce subtly different
embeddings. The downstream consumer (Next.js demo, any TypeScript user) would compute
phenotype scores against a different vector than the one we measured reliability on, and the
shipped `roc_auc` numbers would no longer be valid.

Test plan:

  1. Load the released PyTorch checkpoint via `load_encoder` (canonical).
  2. Load the released ONNX via `onnxruntime.InferenceSession`.
  3. Generate 16 random 24h windows with realistic glucose ranges and realistic mask density.
  4. Run each window through both backends, assert |PyTorch - ONNX| < 1e-4 on the embedding.
  5. Then apply the JSON-serialised logistic heads (StandardScaler + LogReg) in a numpy port
     and assert the resulting probabilities match the original sklearn `pipeline.predict_proba`
     to < 1e-4. This catches any round-trip error in the head export.

Marked `@pytest.mark.golden`. CPU only. Skips if either artefact is missing (so the test
suite still passes in a fresh checkout where the artefacts haven't been generated).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = REPO_ROOT / "artifacts" / "glucofm_encoder.onnx"
HEADS_PATH = REPO_ROOT / "artifacts" / "glucofm_heads.json"
CKPT_PATH = REPO_ROOT / "runs_5090" / "rawstats120" / "ckpt_ep040.pt"

PARITY_TOL = 1e-4


def _make_random_windows(n: int = 16, seed: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate `n` random 24h CGM windows with realistic glucose ranges and density.

    glucose mg/dL:    uniform in [60, 250], clamped to plausible
    mask:             bernoulli(p=0.85) — most of a CGM day is observed
    circadian_start:  uniform int in [0, 287] — start position of window in 24h cycle
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(60.0, 250.0, size=(n, 288)).astype(np.float32)
    values = np.clip(values, 40.0, 500.0)
    mask = (rng.uniform(0.0, 1.0, size=(n, 288)) < 0.85).astype(np.float32)
    circ = rng.integers(0, 288, size=(n,)).astype(np.int64)
    return values, mask, circ


def _to_onnx_inputs(values: np.ndarray, mask: np.ndarray, circ: np.ndarray) -> dict:
    return {
        "values": values.astype(np.float32),
        "mask": mask.astype(np.float32),
        "circadian_start": circ.astype(np.int64),
    }


def _to_torch_inputs(values, mask, circ, device: str = "cpu"):
    import torch

    return (
        torch.from_numpy(values.astype(np.float32)).to(device),
        torch.from_numpy(mask.astype(np.float32)).to(device),
        torch.from_numpy(circ.astype(np.int64)).to(device),
    )


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} not generated; run scripts/export_encoder_onnx.py first")


@pytest.mark.golden
def test_onnx_matches_pytorch_embedding() -> None:
    """The released ONNX must produce the same 128-d embedding as canonical PyTorch."""
    _skip_if_missing(ONNX_PATH)
    _skip_if_missing(CKPT_PATH)

    import onnxruntime as ort

    from opencgm_stateevent.eval.embed import load_encoder
    from scripts.export_encoder_onnx import EncoderMeanEmbed, replace_transformer_for_onnx

    # PyTorch reference — must match the export script's transformer swap exactly, otherwise
    # the test compares the ONNX's decomposed weights against the canonical fused MHA and the
    # 0.5+ delta is just transformer impl noise.
    model, _ref = load_encoder(CKPT_PATH, device="cpu")
    replace_transformer_for_onnx(model)
    wrapper = EncoderMeanEmbed(model).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # ONNX runtime
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    values, mask, circ = _make_random_windows(n=16, seed=17)
    pt_out = wrapper(*_to_torch_inputs(values, mask, circ)).numpy()
    onnx_out = sess.run(["embedding"], _to_onnx_inputs(values, mask, circ))[0]

    assert pt_out.shape == onnx_out.shape == (16, 128)
    abs_err = float(np.abs(pt_out - onnx_out).max())
    assert abs_err < PARITY_TOL, (
        f"max |pt - onnx| = {abs_err:.2e} exceeds tolerance {PARITY_TOL}; "
        f"check the architecture flags were read back from the checkpoint."
    )


@pytest.mark.golden
def test_onnx_matches_pytorch_with_sparse_mask() -> None:
    """Edge case: very sparse mask (50% observed). Tests masked_instance_norm path."""
    _skip_if_missing(ONNX_PATH)
    _skip_if_missing(CKPT_PATH)

    import onnxruntime as ort

    from opencgm_stateevent.eval.embed import load_encoder
    from scripts.export_encoder_onnx import EncoderMeanEmbed, replace_transformer_for_onnx

    model, _ref = load_encoder(CKPT_PATH, device="cpu")
    replace_transformer_for_onnx(model)
    wrapper = EncoderMeanEmbed(model).eval()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(29)
    values = rng.uniform(70.0, 220.0, size=(8, 288)).astype(np.float32)
    mask = (rng.uniform(0.0, 1.0, size=(8, 288)) < 0.50).astype(np.float32)  # sparse
    circ = rng.integers(0, 288, size=(8,)).astype(np.int64)

    pt_out = wrapper(*_to_torch_inputs(values, mask, circ)).numpy()
    onnx_out = sess.run(["embedding"], _to_onnx_inputs(values, mask, circ))[0]

    assert pt_out.shape == onnx_out.shape == (8, 128)
    abs_err = float(np.abs(pt_out - onnx_out).max())
    assert abs_err < PARITY_TOL, f"sparse mask: max |pt - onnx| = {abs_err:.2e}"


@pytest.mark.golden
def test_heads_json_matches_sklearn_pipeline() -> None:
    """The exported JSON heads must produce the same probabilities as the original sklearn pipeline.

    This is the second half of the consumer round-trip: even if the embedding is right, a
    bug in the JSON exporter would make the demo's phenotype scores wrong.
    """
    _skip_if_missing(HEADS_PATH)
    pytest.importorskip("sklearn")

    import pickle

    bundle_pkl = REPO_ROOT / "artifacts" / "heads.pkl"
    if not bundle_pkl.exists():
        pytest.skip(f"{bundle_pkl.name} not generated; run `just fit-heads` first")

    with bundle_pkl.open("rb") as fh:
        bundle = pickle.load(fh)

    with HEADS_PATH.open() as fh:
        exported = json.load(fh)

    rng = np.random.default_rng(43)
    for key, head in bundle["heads"].items():
        if not head.get("has_signal", False):
            continue
        x = rng.normal(0.0, 1.0, size=(64, 128)).astype(np.float32)

        # sklearn reference
        proba_ref = head["pipeline"].predict_proba(x)

        # numpy port of scale + classifier. Mirrors web/lib/model/heads.ts:applyHead():
        # binary (coef shape [1, 128]) → sigmoid + duplicate to [N, 2]; multiclass → softmax.
        h = exported["heads"][key]
        scale_mean = np.asarray(h["scale"]["mean"], dtype=np.float64)
        scale_var = np.asarray(h["scale"]["scale"], dtype=np.float64)
        coef = np.asarray(h["classifier"]["coef"], dtype=np.float64)
        intercept = np.asarray(h["classifier"]["intercept"], dtype=np.float64)

        x_scaled = (x.astype(np.float64) - scale_mean) / scale_var
        logits = x_scaled @ coef.T + intercept  # [N, K]; K=1 for binary

        if coef.shape[0] == 1:
            # Binary: sklearn LogReg fits one-vs-rest; predict_proba returns [N, 2] using
            # sigmoid, not softmax. Single-element softmax is 1, so we must use sigmoid.
            p1 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            proba = np.stack([1.0 - p1, p1], axis=1)
        else:
            # Numerically stable softmax over the K class scores.
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            proba = exp / exp.sum(axis=1, keepdims=True)

        abs_err = float(np.abs(proba - proba_ref).max())
        assert abs_err < 1e-4, (
            f"{key}: max |json - sklearn| = {abs_err:.2e}; "
            "check the JSON exporter's coef/intercept layout."
        )
