"""Can several seeds share one 3090? Blueprint §18.1 step 3.

A step is launch-latency bound, not arithmetic bound: 22 ms of GPU time for a 0.44 M-parameter
encoder over 24 tokens means the device spends most of each step idle between kernels. Peak
memory is 0.18 GB against 24 GB available. Both facts point the same way -- concurrent seeds
should interleave into each other's gaps almost for free.

This matters because it is the only speedup available that changes nothing about the arithmetic.
Each process runs the identical eager float32 code it would run alone, so a concurrently-trained
seed is bit-identical to a serially-trained one. torch.compile was 1.44x but shifted the loss by
1.3e-3 relative, which is TF32 matmul precision rather than fusion, and that is not a trade worth
making on a headline number.

Reports aggregate throughput -- total steps per second across all workers -- which is the figure
that determines wall-clock for a five-seed sweep.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path


def worker(rank: int, steps: int, warmup: int, q) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from opencgm_stateevent.model.model import OpenCGMStateEvent
    from opencgm_stateevent.train.dataset import CachePaths, WindowDataset
    from opencgm_stateevent.train.loop import TrainConfig, Trainer

    def collate(batch):
        v = torch.from_numpy(np.stack([b[0] for b in batch]))
        m = torch.from_numpy(np.stack([b[1] for b in batch]))
        c = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return v, m, c

    cfg = TrainConfig(seed=17 + rank, amp=False, batch_size=128, num_workers=4, log_every=0)
    ds = WindowDataset(CachePaths.for_tag("strict_seed17"), seed=cfg.seed)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=True,
        prefetch_factor=4,
    )
    trainer = Trainer(
        model=OpenCGMStateEvent(), cfg=cfg, steps_per_epoch=len(ds) // cfg.batch_size
    )
    it = iter(loader)
    for i in range(warmup):
        trainer.step(next(it), 0, i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        trainer.step(next(it), 0, i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    q.put({
        "rank": rank,
        "steps_per_sec": steps / elapsed,
        "ms_per_step": elapsed / steps * 1000,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    })


def human(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--out", type=Path, default=Path("reports/concurrency.json"))
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    results = []
    for n in args.levels:
        q = ctx.Queue()
        procs = [
            ctx.Process(target=worker, args=(r, args.steps, args.warmup, q)) for r in range(n)
        ]
        for p in procs:
            p.start()
        got = [q.get() for _ in range(n)]
        for p in procs:
            p.join()

        agg = sum(g["steps_per_sec"] for g in got)
        slowest = max(g["ms_per_step"] for g in got)
        # 331,057 steps per 120-epoch seed at 2,759 steps/epoch
        steps_per_seed = 2759 * 120
        wall_5 = steps_per_seed * 5 / agg if n >= 5 else None
        r = {
            "workers": n,
            "aggregate_steps_per_sec": agg,
            "per_worker_ms": [round(g["ms_per_step"], 2) for g in got],
            "slowest_ms_per_step": slowest,
            "total_peak_vram_gb": sum(g["peak_vram_gb"] for g in got),
            "seconds_per_seed": steps_per_seed / (1000 / slowest),
            "seconds_all_5_seeds": wall_5,
        }
        results.append(r)
        print(
            f"{n} worker(s): aggregate {agg:6.2f} step/s   slowest {slowest:6.2f} ms/step   "
            f"vram {r['total_peak_vram_gb']:.2f} GB   "
            f"per seed {human(r['seconds_per_seed'])}"
            + (f"   all 5 in {human(wall_5)}" if wall_5 else "")
        )

    base = results[0]["aggregate_steps_per_sec"]
    print("\nscaling vs 1 worker: " + "  ".join(
        f"{r['workers']}x={r['aggregate_steps_per_sec'] / base:.2f}" for r in results
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
