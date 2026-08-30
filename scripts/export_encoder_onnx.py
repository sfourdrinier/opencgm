"""Export the frozen encoder to ONNX for browser/TypeScript inference.

This is the load-bearing artefact for the Hugging Face model card, the Next.js demo, and any
non-Python downstream consumer. The released encoder is the §19.1 *mean-pooled* 128-d daily
embedding (unweighted mean over the 24 patch tokens).

Inputs:
  - values:           [B, 288] float32, glucose mg/dL
  - mask:             [B, 288] float32, 0 = unobserved, 1 = observed (never interpolated)
  - circadian_start:  [B]    int64, position of the window start in the 24-hour cycle (0..287)

Output:
  - embedding:        [B, 128] float32, the §19.1 daily embedding

The encoder is wrapped in a tiny Module that calls `model.online(...)` and reduces the 24 patch
tokens to a single 128-d vector via unweighted mean. No JEPA masking, no predictor, no EMA
target — those are training-only and are intentionally excluded from the released graph.

The output ONNX file is at `artifacts/glucofm_encoder.onnx`. The architecture flags are
read back from the checkpoint's own config block (`architecture_of`), not from current
defaults, so a silent regression in the model definition cannot smuggle into a shipped ONNX.

Why a custom transformer block? `nn.TransformerEncoder` calls a fused C++ op
(`aten::_transformer_encoder_layer_fwd`) that ONNX cannot decompose for opset 17. We swap
it for a decomposed `DecomposedTransformerEncoder` (`MultiheadAttention` + `Linear` +
`LayerNorm` + `GELU` + residual) with the same pre-LN structure, then copy weights from the
original checkpoint. The exported graph is fully ONNX-Runtime-Web compatible.

    uv run python scripts/export_encoder_onnx.py \\
        --checkpoint runs_5090/rawstats120/ckpt_ep040.pt \\
        --out artifacts/glucofm_encoder.onnx

`--streams` exports a second artifact (`artifacts/glucofm_encoder_streams.onnx`) that adds
`state_signal` and `event_signal` [B, 288] outputs, so the demo can plot the dual-stream
decomposition next to the raw trace. It is a separate file on purpose: the base encoder's
SHA-256 is pinned by the website, the API and `tests/export/test_onnx_parity.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from opencgm_stateevent.model.model import OpenCGMStateEvent, architecture_of


class DecomposedMultiheadAttention(nn.Module):
    """Manual MHA using only `Linear` + `matmul` + `softmax`. ONNX-decomposable.

    Replaces `nn.MultiheadAttention`, which uses a fused C++ op
    (`aten::_native_multi_head_attention`) that the ONNX exporter cannot decompose.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        # Three separate projections (the original MHA stacks them into one in_proj_weight)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        batch, length, _ = query.shape
        # [B, L, d_model] → [B, L, nhead, d_k] → [B, nhead, L, d_k]
        q = self.q_proj(query).view(batch, length, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch, length, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch, length, self.nhead, self.d_k).transpose(1, 2)
        # Scaled dot-product attention: scores = Q K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        # dropout is a no-op in eval mode (model is frozen); skip the F.dropout call to
        # keep the ONNX graph tight.
        out = torch.matmul(attn, v)  # [B, nhead, L, d_k]
        # [B, nhead, L, d_k] → [B, L, nhead, d_k] → [B, L, d_model]
        out = out.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.out_proj(out), None


class DecomposedTransformerLayer(nn.Module):
    """Pre-LN transformer layer decomposed for ONNX export.

    Mirrors `nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, activation='gelu',
    batch_first=True, norm_first=True)`. Decomposed into `ManualMultiheadAttention` +
    `LayerNorm` + `Linear` + `GELU` so each op maps cleanly to an ONNX primitive.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = DecomposedMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: norm first, then attention with residual.
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        # Pre-LN: norm, then FFN with residual.
        x_norm = self.norm2(x)
        ff = self.linear2(self.dropout2(self.activation(self.linear1(x_norm))))
        x = x + self.dropout3(ff)
        return x


class DecomposedTransformerEncoder(nn.Module):
    """Stand-in for `nn.TransformerEncoder` that ONNX can decompose."""

    def __init__(
        self, d_model: int, nhead: int, dim_ff: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecomposedTransformerLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _copy_attention_weights(
    src_attn: nn.MultiheadAttention, dst_attn: DecomposedMultiheadAttention
) -> None:
    """Split the original MHA's `in_proj_weight: [3*d_model, d_model]` into Q, K, V projections.

    The original `nn.MultiheadAttention.in_proj_weight` stacks [W_q; W_k; W_v] along dim 0.
    PyTorch stores biases the same way: [b_q; b_k; b_v].
    """
    d = dst_attn.d_model
    w = src_attn.in_proj_weight.detach()
    b = src_attn.in_proj_bias.detach() if src_attn.in_proj_bias is not None else None
    dst_attn.q_proj.weight.data.copy_(w[:d])
    dst_attn.k_proj.weight.data.copy_(w[d : 2 * d])
    dst_attn.v_proj.weight.data.copy_(w[2 * d :])
    if b is not None:
        dst_attn.q_proj.bias.data.copy_(b[:d])
        dst_attn.k_proj.bias.data.copy_(b[d : 2 * d])
        dst_attn.v_proj.bias.data.copy_(b[2 * d :])
    dst_attn.out_proj.weight.data.copy_(src_attn.out_proj.weight.detach())
    dst_attn.out_proj.bias.data.copy_(src_attn.out_proj.bias.detach())


def replace_transformer_for_onnx(model: OpenCGMStateEvent) -> None:
    """Swap `model.online.transformer` for an ONNX-decomposed equivalent, copying weights.

    Walks the original `nn.TransformerEncoder` to discover d_model, nhead, dim_ff, num_layers,
    and dropout, then constructs a `DecomposedTransformerEncoder` and copies the equivalent
    weights, splitting the fused `in_proj_weight` of the original MHA into separate Q/K/V
    projections.
    """
    orig = model.online.transformer
    if not isinstance(orig, nn.TransformerEncoder):
        raise TypeError(f"expected nn.TransformerEncoder, got {type(orig).__name__}")
    layers = orig.layers
    if not layers:
        raise ValueError("transformer has no layers")
    sample = layers[0]
    d_model = sample.linear1.in_features
    dim_ff = sample.linear1.out_features
    nhead = sample.self_attn.num_heads
    dropout = sample.dropout.p if hasattr(sample.dropout, "p") else 0.0
    num_layers = len(layers)

    decomposed = DecomposedTransformerEncoder(d_model, nhead, dim_ff, num_layers, dropout)
    decomposed.eval()

    # Copy weights layer-by-layer. The ManualMultiheadAttention takes three separate
    # projections where the original took a fused in_proj, so we split explicitly.
    with torch.no_grad():
        for i, src_layer in enumerate(layers):
            dst_layer = decomposed.layers[i]
            _copy_attention_weights(src_layer.self_attn, dst_layer.self_attn)
            dst_layer.linear1.weight.data.copy_(src_layer.linear1.weight.detach())
            dst_layer.linear1.bias.data.copy_(src_layer.linear1.bias.detach())
            dst_layer.linear2.weight.data.copy_(src_layer.linear2.weight.detach())
            dst_layer.linear2.bias.data.copy_(src_layer.linear2.bias.detach())
            dst_layer.norm1.weight.data.copy_(src_layer.norm1.weight.detach())
            dst_layer.norm1.bias.data.copy_(src_layer.norm1.bias.detach())
            dst_layer.norm2.weight.data.copy_(src_layer.norm2.weight.detach())
            dst_layer.norm2.bias.data.copy_(src_layer.norm2.bias.detach())

    model.online.transformer = decomposed
    for p in model.online.transformer.parameters():
        p.requires_grad_(False)


class EncoderMeanEmbed(nn.Module):
    """Inference-only wrapper: online encoder → mean of 24 patch tokens (§19.1 headline).

    Pulled out of `scripts/export_encoder_onnx.py` so it can be reused by the parity test.
    """

    def __init__(self, model: OpenCGMStateEvent) -> None:
        super().__init__()
        self.online = model.online

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, circadian_start: torch.Tensor
    ) -> torch.Tensor:
        out = self.online(values, mask, circadian_start)
        return out.contextual_tokens.mean(dim=1)  # [B, 128]


class EncoderStreamsEmbed(nn.Module):
    """Inference-only wrapper: embedding plus the state/event streams at 288-sample resolution.

    Same forward as `EncoderMeanEmbed` with two extra outputs, so a browser can plot the
    decomposition the architecture is built around. The streams are taken *before* the
    transformer: `state_signal` is the causal Gaussian smoothing of the masked-instance-
    normalized input, `event_signal` the masked residual. Both are therefore in per-window
    z-score units, not mg/dL. At observed positions state + event equals the normalized
    input; at unobserved positions event is exactly 0 and state is the filter's estimate
    from past observed samples (0 when none fall inside the causal radius).
    """

    def __init__(self, model: OpenCGMStateEvent) -> None:
        super().__init__()
        self.online = model.online

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, circadian_start: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.online(values, mask, circadian_start)
        embedding = out.contextual_tokens.mean(dim=1)  # [B, 128]
        return embedding, out.state_signal, out.event_signal  # + 2 x [B, 288]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs_5090/rawstats120/ckpt_ep040.pt"),
        help="Path to the released encoder checkpoint (default: 5090 ep40 demo ckpt).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to write the ONNX file (default: artifacts/glucofm_encoder.onnx, or"
        " artifacts/glucofm_encoder_streams.onnx with --streams).",
    )
    ap.add_argument(
        "--streams",
        action="store_true",
        help="Also expose state_signal and event_signal [B, 288] alongside the embedding."
        " Written to a separate artifact so the existing encoder ONNX keeps its pinned SHA-256.",
    )
    ap.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version (17 = modern, broad support)."
    )
    args = ap.parse_args()
    if args.out is None:
        args.out = Path(
            "artifacts/glucofm_encoder_streams.onnx"
            if args.streams
            else "artifacts/glucofm_encoder.onnx"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}  -- did you mean a different path?"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load the full checkpoint (config + state); architecture flags come from the ckpt's own
    # config block, NOT from current defaults — this is the §19.2 invariant.
    print(f"loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = architecture_of(state)
    print(f"  architecture flags: {architecture}")

    model = OpenCGMStateEvent(**architecture)
    model.load_state_dict(state["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Sanity: run the canonical forward before swap, record a reference embedding.
    print("  sanity: reference PyTorch forward (canonical transformer)...")
    sample_values = torch.linspace(80.0, 200.0, 288).unsqueeze(0).float()
    sample_mask = torch.ones(1, 288)
    sample_circ = torch.zeros(1, dtype=torch.int64)
    with torch.no_grad():
        ref_out = model.online(sample_values, sample_mask, sample_circ)
        ref_emb = ref_out.contextual_tokens.mean(dim=1)

    # Swap the fused transformer for a decomposed one (identical numerics, ONNX-friendly).
    print("  swapping nn.TransformerEncoder → DecomposedTransformerEncoder")
    replace_transformer_for_onnx(model)

    # Sanity: post-swap embedding should match the pre-swap reference within float32 epsilon.
    with torch.no_grad():
        post_out = model.online(sample_values, sample_mask, sample_circ)
        post_emb = post_out.contextual_tokens.mean(dim=1)
    diff = float((ref_emb - post_emb).abs().max())
    print(f"  max |pre-swap - post-swap| = {diff:.2e}")
    if diff > 1e-4:
        raise RuntimeError(
            f"transformer swap changed numerics by {diff:.2e} on a sanity window"
            " - refusing to export. "
            f"Check the DecomposedTransformerLayer pre-LN structure vs nn.TransformerEncoderLayer."
        )

    wrapper_cls = EncoderStreamsEmbed if args.streams else EncoderMeanEmbed
    wrapper = wrapper_cls(model).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Sanity dummies. A *sparse* mask is critical: `rate_of_change` has
    # `if not take.any(): continue`, and the ONNX tracer records that as a Python bool
    # constant per iteration. With an all-ones dummy, every iteration after the first
    # records `take.any()=False` and the exporter elides the corresponding `where` op.
    # A real sparse input at runtime has `take.any()=True` at those iterations, so the
    # graph is missing updates and produces different embeddings. Using a partially-observed
    # dummy here forces every iteration to be traced with the `where` op in place, which
    # matches the live input distribution.
    batch = 2
    rng = torch.Generator().manual_seed(13)
    values = torch.zeros(batch, 288, dtype=torch.float32)
    mask = (torch.rand(batch, 288, generator=rng) < 0.5).to(torch.float32)
    circadian_start = torch.zeros(batch, dtype=torch.int64)

    # Export. dynamic_axes lets the consumer call with any batch size.
    # We use `dynamo=False` (the legacy TorchScript tracer) because the encoder contains
    # data-dependent control flow (`if not take.any(): ...`) in `rate_of_change` that the
    # newer Dynamo-based exporter cannot statically guard. The legacy tracer handles this
    # by tracing the control flow with concrete inputs; the dynamic_axes still lets the
    # consumer use any batch size.
    output_names = ["embedding"]
    if args.streams:
        output_names += ["state_signal", "event_signal"]
    dynamic_axes = {name: {0: "B"} for name in ["values", "mask", "circadian_start", *output_names]}

    print(f"exporting to {args.out} (opset={args.opset}, legacy tracer, decomposed transformer)")
    torch.onnx.export(
        wrapper,
        (values, mask, circadian_start),
        str(args.out),
        input_names=["values", "mask", "circadian_start"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify the export round-trips before declaring success.
    import onnx

    print("checking exported graph with onnx.checker...")
    onnx.checker.check_model(str(args.out), full_check=True)

    # SHA-256 of the file — pinned into the HF model card and the web/ README.
    sha = sha256_of(args.out)
    print(f"\nexported: {args.out}")
    print(f"  size: {args.out.stat().st_size:,} bytes")
    print(f"  sha256: {sha}")

    # Sidecar provenance JSON. Same fields as the existing heads.json encoder block so any
    # downstream verifier can pin all artefacts to one checkpoint.
    sidecar = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(state.get("epoch", -1)),
        "seed": int(state.get("config", {}).get("seed", -1)),
        "architecture": architecture,
        "opset": args.opset,
        "input_names": ["values", "mask", "circadian_start"],
        "input_dtypes": ["float32", "float32", "int64"],
        "input_shapes": ["[B, 288]", "[B, 288]", "[B]"],
        "output_names": output_names,
        "output_dtype": "float32",
        "output_shape": "[B, 128]" if not args.streams else ["[B, 128]", "[B, 288]", "[B, 288]"],
        "size_bytes": args.out.stat().st_size,
        "sha256": sha,
        "mask_convention": "0 = unobserved, 1 = observed; never interpolate before this point",
        "units": "mg/dL",
    }
    if args.streams:
        sidecar["stream_convention"] = (
            "state_signal and event_signal are in per-window z-score units (masked instance"
            " norm over observed positions), not mg/dL. At observed positions state + event"
            " reconstructs the normalized input; at unobserved positions event is exactly 0"
            " and state is the causal filter's estimate from past observed samples (0 when"
            " none fall inside the 36-step causal radius)."
        )
    sidecar_path = args.out.with_suffix(".onnx.meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"  sidecar: {sidecar_path}")


if __name__ == "__main__":
    main()
