"""Measure real training throughput on this machine. Blueprint §18.1 step 3.

Extrapolating 120 epochs from a guess is how a five-seed run becomes a two-week run nobody
planned for. This measures steps per second under the actual configuration -- real cache, real
augmentation, real dataloader -- and reports what 120 epochs and five seeds actually cost.

The first steps are discarded: cuDNN autotunes its convolution algorithms on the first call and
the page cache is cold, so including them understates steady-state throughput.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencgm_stateevent.model.encoder import parameter_report
from opencgm_stateevent.model.model import OpenCGMStateEvent
from opencgm_stateevent.train.dataset import CachePaths, WindowDataset
from opencgm_stateevent.train.loop import TrainConfig, Trainer

WARMUP = 30


def collate(batch):
    v = torch.from_numpy(np.stack([b[0] for b in batch]))
    m = torch.from_numpy(np.stack([b[1] for b in batch]))
    c = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return v, m, c


def human(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        return f"{h // 24}d {h % 24}h {m:02d}m"
    return f"{h}h {m:02d}m {s:02d}s"


def run(cfg: TrainConfig, steps: int, tag: str) -> dict:
    paths = CachePaths.for_tag("strict_seed17")
    ds = WindowDataset(paths, seed=cfg.seed)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    steps_per_epoch = len(ds) // cfg.batch_size
    model = OpenCGMStateEvent()
    trainer = Trainer(model=model, cfg=cfg, steps_per_epoch=steps_per_epoch)

    times: list[float] = []
    it = iter(loader)
    for i in range(steps):
        batch = next(it)
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        trainer.step(batch, 0, i)
        if i >= WARMUP:
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()

    a = np.asarray(times)
    sps = 1.0 / a.mean()
    epoch_s = steps_per_epoch / sps
    return {
        "tag": tag,
        "amp": cfg.amp,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "measured_steps": len(a),
        "steps_per_sec": sps,
        "windows_per_sec": sps * cfg.batch_size,
        "ms_per_step_mean": float(a.mean() * 1000),
        "ms_per_step_p50": float(np.percentile(a, 50) * 1000),
        "ms_per_step_p95": float(np.percentile(a, 95) * 1000),
        "steps_per_epoch": steps_per_epoch,
        "total_steps_120ep": steps_per_epoch * 120,
        "seconds_per_epoch": epoch_s,
        "seconds_120_epochs": epoch_s * 120,
        "seconds_5_seeds": epoch_s * 120 * 5,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=330)
    ap.add_argument("--out", type=Path, default=Path("reports/throughput.json"))
    args = ap.parse_args()

    print(parameter_report(OpenCGMStateEvent().online))
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")

    results = []
    for tag, kw in [
        ("fp16_bs128_w6", {"amp": True, "batch_size": 128, "num_workers": 6}),
        ("fp32_bs128_w6", {"amp": False, "batch_size": 128, "num_workers": 6}),
        ("fp16_bs128_w12", {"amp": True, "batch_size": 128, "num_workers": 12}),
        ("fp16_bs512_w6", {"amp": True, "batch_size": 512, "num_workers": 6}),
    ]:
        torch.cuda.reset_peak_memory_stats()
        r = run(TrainConfig(log_every=0, **kw), args.steps, tag)
        results.append(r)
        print(
            f"{tag:<16} {r['ms_per_step_mean']:6.1f} ms/step  "
            f"{r['steps_per_sec']:6.2f} step/s  {r['windows_per_sec']:8,.0f} win/s  "
            f"p95 {r['ms_per_step_p95']:6.1f}  {r['peak_vram_gb']:.2f} GB"
        )

    print()
    for r in results:
        print(
            f"{r['tag']:<16} epoch {human(r['seconds_per_epoch']):>12}   "
            f"120ep {human(r['seconds_120_epochs']):>12}   "
            f"5 seeds {human(r['seconds_5_seeds']):>12}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
