"""One training-throughput worker, to be launched as an independent process.

`scripts/concurrency.sh` runs N of these at once. Separate OS processes rather than a
multiprocessing pool, because that is exactly how the seed sweep will actually run: N copies of
`opencgm_stateevent.train.run` sharing one GPU. Measuring the deployment shape avoids measuring
an artefact of the harness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from opencgm_stateevent.model.model import OpenCGMStateEvent
from opencgm_stateevent.train.dataset import CachePaths, WindowDataset
from opencgm_stateevent.train.loop import TrainConfig, Trainer
from opencgm_stateevent.train.run import epoch_permutation, make_loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = TrainConfig(
        seed=args.seed, amp=False, batch_size=args.batch_size,
        num_workers=args.workers, log_every=0,
    )
    ds = WindowDataset(CachePaths.for_tag("strict_seed17"), seed=cfg.seed)
    loader = make_loader(ds, epoch_permutation(len(ds), cfg.seed, 0), cfg)
    trainer = Trainer(
        model=OpenCGMStateEvent(), cfg=cfg, steps_per_epoch=len(ds) // cfg.batch_size
    )

    it = iter(loader)
    for i in range(args.warmup):
        trainer.step(next(it), 0, i)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(args.steps):
        trainer.step(next(it), 0, i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    args.out.write_text(json.dumps({
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "steps_per_sec": args.steps / elapsed,
        "ms_per_step": elapsed / args.steps * 1000,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }))
    # Skip interpreter shutdown: persistent DataLoader workers make a clean exit slow and
    # occasionally hang, and every measured result is already on disk.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
