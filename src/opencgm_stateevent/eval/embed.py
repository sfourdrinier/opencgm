"""Frozen-encoder embedding cache. Blueprint §19.1, §19.2.

The headline daily representation is the unweighted mean of the 24 contextual patch tokens
(§19.1). Density weighting is available as a separate, separately-reported variant — §19.1 is
explicit that it must not replace the headline.

§19.2 requires every cached embedding to record what produced it, and the cache to be invalidated
on any mismatch. That is enforced here rather than documented: the provenance is hashed into the
cache filename, so a changed checkpoint, config or preprocessing simply cannot hit a stale entry.
Silent reuse of embeddings from a different checkpoint is the kind of error that produces a clean,
plausible, wrong results table.

At inference the EMA target, predictor and transition heads are discarded (§19.1) and no JEPA
masking is applied — the whole physical view is visible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ..model.model import OpenCGMStateEvent, architecture_of
from ..provenance import git_state
from .windows import WindowSet

CACHE = Path("data/canonical/embeddings")
BATCH = 256

#: Pooling variants. `mean` is the §19.1 headline; the others are reported separately.
POOLINGS = ("mean", "density_weighted", "mean_max")


@dataclass(frozen=True)
class EncoderRef:
    """Everything that determines an embedding. §19.2."""

    checkpoint: Path
    weights_sha256: str
    epoch: int
    seed: int
    git_sha: str
    dtype: str = "float32"
    backend: str = "pytorch"
    #: hash of the windows being embedded, set per source by `window_fingerprint`
    windows_sha256: str = ""
    #: architecture flags read back from the checkpoint's own config, not from current defaults
    architecture: str = ""

    @property
    def tag(self) -> str:
        """Cache key. §19.2 requires invalidation on *any* mismatch.

        The window fingerprint is part of the key. Without it, regenerating the evaluation
        windows — a different binning rule, a re-read source, a fixed reader — silently hits
        embeddings computed from the old ones, and the resulting table looks clean and is wrong.
        Weight hash alone cannot catch that, because the weights did not change.
        """
        payload = json.dumps({
            "weights": self.weights_sha256, "epoch": self.epoch, "seed": self.seed,
            "git": self.git_sha, "dtype": self.dtype, "backend": self.backend,
            "windows": self.windows_sha256, "architecture": self.architecture,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def for_windows(self, ws: WindowSet) -> EncoderRef:
        return replace(self, windows_sha256=window_fingerprint(ws))

    def to_dict(self) -> dict:
        return {**self.__dict__, "checkpoint": str(self.checkpoint), "tag": self.tag}


def load_encoder(checkpoint: Path, device: str = "cuda") -> tuple[OpenCGMStateEvent, EncoderRef]:
    """Load a checkpoint's *online* encoder, frozen, in eval mode.

    The target branch is deliberately not loaded. §19.1 discards it, and evaluating it instead of
    the online encoder would be an easy and invisible substitution.
    """
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = architecture_of(state)
    model = OpenCGMStateEvent(**architecture)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    online = {k: v for k, v in state["model"].items() if k.startswith("online.")}
    digest = hashlib.sha256()
    for key in sorted(online):
        digest.update(key.encode())
        digest.update(online[key].detach().cpu().numpy().tobytes())

    return model, EncoderRef(
        checkpoint=checkpoint,
        weights_sha256=digest.hexdigest(),
        epoch=int(state.get("epoch", -1)),
        seed=int(state.get("config", {}).get("seed", -1)),
        git_sha=git_state().get("sha", "unknown"),
        architecture=json.dumps(architecture, sort_keys=True),
    )


def _pool(tokens: torch.Tensor, density: torch.Tensor, how: str) -> torch.Tensor:
    """`tokens` is [B,24,D]; returns [B,D] or [B,2D] for `mean_max`."""
    if how == "mean":
        return tokens.mean(dim=1)  # §19.1 headline, unweighted
    if how == "density_weighted":
        w = density.unsqueeze(-1)
        return (tokens * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)
    if how == "mean_max":
        return torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
    raise ValueError(f"unknown pooling {how!r}")


@torch.no_grad()
def embed(
    model: OpenCGMStateEvent,
    ws: WindowSet,
    *,
    device: str = "cuda",
    poolings: tuple[str, ...] = POOLINGS,
) -> dict[str, np.ndarray]:
    """Embed every window in `ws`. Returns one array per pooling, plus the patch tokens."""
    out: dict[str, list[torch.Tensor]] = {p: [] for p in poolings}
    out["patch_density"] = []
    for start in range(0, len(ws), BATCH):
        stop = start + BATCH
        values = torch.from_numpy(ws.values[start:stop]).to(device)
        mask = torch.from_numpy(ws.mask[start:stop]).to(device)
        circadian = torch.from_numpy(ws.circadian[start:stop]).to(device)
        result = model.encode(values, mask, circadian)
        for how in poolings:
            out[how].append(_pool(result.contextual_tokens, result.patch_density, how).cpu())
        out["patch_density"].append(result.patch_density.cpu())
    return {k: torch.cat(v).float().numpy() for k, v in out.items() if v}


def window_fingerprint(ws: WindowSet) -> str:
    """Content hash of the windows themselves, not their filename or count.

    Covers values, mask, circadian phase and the subject/entry keys, so any change to
    preprocessing or to which windows exist produces a different key.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(ws.values).tobytes())
    digest.update(np.ascontiguousarray(ws.mask).tobytes())
    digest.update(np.ascontiguousarray(ws.circadian).tobytes())
    digest.update("|".join(map(str, ws.subjects)).encode())
    if ws.entries is not None:
        digest.update("|".join(map(str, ws.entries)).encode())
    return digest.hexdigest()


def cache_file(source: str, ref: EncoderRef) -> Path:
    return CACHE / f"{source}.{ref.tag}.npz"


def load_or_embed(
    model: OpenCGMStateEvent, ws: WindowSet, ref: EncoderRef, *, device: str = "cuda"
) -> dict[str, np.ndarray]:
    """Cache keyed on the provenance hash, so a stale entry cannot be reused. §19.2."""
    ref = ref.for_windows(ws)
    path = cache_file(ws.source, ref)
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    features = embed(model, ws, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)
    path.with_suffix(".json").write_text(
        json.dumps({"source": ws.source, "windows": len(ws), **ref.to_dict()}, indent=2)
    )
    return features
