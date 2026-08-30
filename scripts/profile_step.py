"""Where does a step go? Blueprint §18.1 step 3, supporting the throughput measurement.

The benchmark showed ~20 ms of fixed cost per step independent of batch size, which means the
step is dominated by something other than arithmetic. Before spending ten hours of GPU time it
is worth knowing whether that cost is the input pipeline or the GPU itself, because only one of
those is worth fixing.

Three configurations, each isolating one suspect:

* ``gpu_only``   -- one batch held resident, no loader. Pure model time.
* ``loader_only``-- iterate the loader, never touch the GPU. Pure input time.
* ``end_to_end`` -- both, as training actually runs.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from benchmark_throughput import collate
from torch.utils.data import DataLoader

from opencgm_stateevent.model.model import OpenCGMStateEvent
from opencgm_stateevent.train.dataset import CachePaths, WindowDataset
from opencgm_stateevent.train.loop import TrainConfig, Trainer

STEPS = 200
WARMUP = 30


def timed(fn, steps: int = STEPS) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts) * 1000
    return float(a.mean()), float(np.percentile(a, 95))


def main() -> None:
    cfg = TrainConfig(amp=False, batch_size=128, num_workers=6, log_every=0)
    paths = CachePaths.for_tag("strict_seed17")

    for augment_enabled in (True, False):
        ds = WindowDataset(paths, seed=cfg.seed, augment_enabled=augment_enabled)
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            collate_fn=collate, pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=4,
        )
        it = iter(loader)
        batch = next(it)

        trainer = Trainer(
            model=OpenCGMStateEvent(), cfg=cfg, steps_per_epoch=len(ds) // cfg.batch_size
        )
        gpu = timed(lambda t=trainer, b=batch: t.step(b, 0, 0))
        ld = timed(lambda i=it: next(i))
        e2e = timed(lambda t=trainer, i=it: t.step(next(i), 0, 0))

        tag = "augment on " if augment_enabled else "augment off"
        print(
            f"{tag}  gpu_only {gpu[0]:6.2f} ms (p95 {gpu[1]:6.2f})   "
            f"loader_only {ld[0]:6.2f} ms (p95 {ld[1]:6.2f})   "
            f"end_to_end {e2e[0]:6.2f} ms (p95 {e2e[1]:6.2f})"
        )


if __name__ == "__main__":
    main()
