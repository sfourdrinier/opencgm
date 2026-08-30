"""End-to-end resume equivalence on the GPU. KR2.6.

Trains N epochs uninterrupted, then trains the same seed again but interrupted at an epoch
boundary and resumed from the checkpoint. Every logged metric of the epochs after the resume must
match the uninterrupted run.

The unit tests in `tests/golden/test_resume.py` cover the same property with synthetic batches on
CPU. This covers what they structurally cannot: real data, the real dataloader, a checkpoint that
has actually been through `torch.save`/`torch.load` on a CUDA device, and the epoch-boundary
bookkeeping in `train.run`. Both failures found so far -- RNG byte tensors moved to the GPU by
`map_location`, and `health` leaving the model in eval mode -- were invisible to the CPU tests and
showed up here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from opencgm_stateevent.train.loop import TrainConfig
from opencgm_stateevent.train.run import train

COMPARE = ("loss", "mcr", "td", "sigma", "grad_norm", "realized_mask_ratio",
           "health_effective_rank", "health_latent_std_mean")
TOL = 1e-9


def epochs_of(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text().splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--interrupt-after", type=int, default=1, help="epoch index to resume from")
    ap.add_argument("--root", type=Path, default=Path("runs/_resume_gate"))
    args = ap.parse_args()

    if args.root.exists():
        shutil.rmtree(args.root)
    cfg = TrainConfig(
        seed=args.seed, amp=False, num_workers=4, log_every=0,
        checkpoint_epochs=tuple(range(args.epochs + 1)),
    )

    reference = args.root / "uninterrupted"
    train(cfg, reference, tag="resume_gate_reference", max_epochs=args.epochs)

    # Second run: same seed, restarted from the checkpoint written after `interrupt_after` epochs.
    resumed = args.root / "resumed"
    resumed.mkdir(parents=True, exist_ok=True)
    ckpt = reference / f"ckpt_ep{args.interrupt_after:03d}.pt"
    shutil.copy(ckpt, resumed / ckpt.name)
    train(cfg, resumed, tag="resume_gate_resumed", resume=resumed / ckpt.name,
          max_epochs=args.epochs)

    ref = {e["epoch"]: e for e in epochs_of(reference)}
    res = {e["epoch"]: e for e in epochs_of(resumed)}

    failures = []
    checked = 0
    for epoch in sorted(res):
        for field in COMPARE:
            a, b = ref[epoch][field], res[epoch][field]
            checked += 1
            if abs(a - b) > TOL:
                failures.append(f"  epoch {epoch} {field}: {a!r} vs {b!r}  (delta {a - b:.3e})")

    print(f"\ncompared {checked} values across epochs {sorted(res)}")
    if failures:
        print("RESUME GATE FAILED")
        print("\n".join(failures))
        return 1
    print("RESUME GATE PASSED — resumed epochs are identical to uninterrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
