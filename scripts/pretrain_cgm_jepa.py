"""Pretrain CGM-JEPA on our corpus, for the head-to-head against our own model.

The paper's central claim is a *margin*: +4.11 PR-AUC over CGM-JEPA, both trained on the same data.
An absolute score degrades at reduced corpus size; a margin does not. This script retrains the
authors' model end-to-end on our windows using exactly the hyperparameters they published, so the
later comparison is "GlucoFM vs CGM-JEPA, both on our 30.9% of the corpus".

GlucoFM was pretrained for 120 epochs; CGM-JEPA's official recipe is 101. Different numbers, not
different methods -- we use theirs.

The corpus is the same `WindowSet` cache that `scripts/evaluate.py` and the existing training
pipeline already use, so the data lineage matches without another copy of the cache. The only
adaptation is the per-window linear interpolation that `interpolate_dense` already provides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from opencgm_stateevent.baselines.cgm_jepa import (
    BASE_LR,
    BATCH_SIZE,
    CGMJEPA,
    EMA_MOMENTUM,
    EPOCHS,
    IPE_SCALE,
    MASK_RATIO,
    PATCH_SIZE,
    PATCHES,
)
from opencgm_stateevent.eval.windows import build_all
from opencgm_stateevent.train.dataset import interpolate_dense

torch.backends.cuda.matmul.allow_tf32 = True


class FilledWindows(Dataset):
    """One 288-step dense sequence per index.

    CGM-JEPA interpolates and treats gaps as observed (the substantive contrast with our model),
    so what arrives at the encoder is the same `[B, 288]` sequence on every call. Per-window
    interpolation is the cheap thing this script actually does.
    """

    def __init__(self, sets: dict) -> None:
        self.values, self.mask = [], []
        for ws in sets.values():
            for v, m in zip(ws.values, ws.mask, strict=True):
                filled, dense = interpolate_dense(v.astype(np.float32, copy=False), m)
                self.values.append(filled.astype(np.float32))
                self.mask.append(dense)
        self.values = np.stack(self.values)
        self.mask = np.stack(self.mask)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, i: int) -> Tensor:
        return torch.from_numpy(self.values[i]).unsqueeze(0)  # [1, 288] one channel


def lr_lambda_factory(num_steps: int):
    """The authors' schedule: 15% linear warmup, then linear decay to zero."""
    warmup = int(0.15 * num_steps)

    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, num_steps - warmup))

    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=43, help="CGM-JEPA's official seed is 43")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rebuild-windows", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("loading windows ...", flush=True)
    sets = build_all(rebuild=args.rebuild_windows)
    ds = FilledWindows(sets)
    print(f"  {len(ds):,} interpolated windows from {len(sets)} sources", flush=True)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    span = args.epochs * IPE_SCALE

    model = CGMJEPA().to(args.device)
    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(total_steps))

    n_params = sum(p.numel() for p in params)
    print(f"  {n_params:,} trainable parameters", flush=True)
    msg = f"  {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps:,} steps"
    print(msg, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    for epoch in range(args.epochs):
        # Linear ramp 0.997 -> ~0.9994, stepped once per epoch, never reaches 1.0.
        m = EMA_MOMENTUM + epoch * (1.0 - EMA_MOMENTUM) / span
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(loader):
            patches = (
                batch.squeeze(1)
                .reshape(-1, PATCHES, PATCH_SIZE)
                .to(args.device, non_blocking=True)
            )
            generator = torch.Generator(device=args.device).manual_seed(
                args.seed * 10_000 + epoch * 100 + step
            )

            optimizer.zero_grad(set_to_none=True)
            out = model(patches, generator)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.update_target(m)

            epoch_loss += float(out.loss.detach())

        avg = epoch_loss / max(1, steps_per_epoch)
        print(
            f"  epoch {epoch:>3}/{args.epochs}  loss {avg:.4f}  ema {m:.5f}  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}  {time.time() - t0:5.1f}s",
            flush=True,
        )
        log.append(
            {"epoch": epoch, "loss": avg, "ema": m, "lr": optimizer.param_groups[0]["lr"]}
        )

    ckpt = args.out / "ckpt_final.pt"
    torch.save(
        {
            "state_dict": model.encoder.state_dict(),
            "config": {
                "model": "cgm_jepa",
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": BASE_LR,
                "mask_ratio": MASK_RATIO,
                "embed_dim": 96,
                "n_layers": 3,
                "n_heads": 6,
                "predictor_dim": 48,
                "predictor_heads": 2,
                "predictor_layers": 1,
            },
            "values_sha256": hashlib.sha256(ds.values.tobytes()).hexdigest(),
        },
        ckpt,
    )

    (args.out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
