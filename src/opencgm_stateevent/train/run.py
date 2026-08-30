"""Pretraining entrypoint. Blueprint §17.5, §18.

Batch order is a deterministic function of ``(seed, epoch)`` rather than a DataLoader shuffle.
This is what makes KR2.6 achievable: resuming means recomputing the permutation for the epoch and
seeking to an offset, so the step after a resume is the step that would have run had nothing been
interrupted. A DataLoader's internal shuffle state cannot be checkpointed that way, and a resume
that silently reshuffles produces a run that is reproducible only until the first interruption.

Everything a run needs to be re-derived later is written next to it: the resolved config, the
window manifest hash, the git SHA, the environment, and a JSONL metrics stream with one record
per logged step.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..model.model import OpenCGMStateEvent, architecture_of
from ..provenance import environment, git_state
from .dataset import CachePaths, WindowDataset, manifest_sha256, resolve_manifest
from .loop import BASE_LR, TrainConfig, Trainer, enable_determinism, save_checkpoint

MANIFEST = Path("manifests/windows/strict_public_seed17_legal_start_fraction.json")


def collate(batch):
    values = torch.from_numpy(np.stack([b[0] for b in batch]))
    mask = torch.from_numpy(np.stack([b[1] for b in batch]))
    circadian = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return values, mask, circadian


def epoch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    """Batch order for one epoch, derived only from ``(seed, epoch)``.

    Deriving it rather than storing it means a checkpoint needs to record an integer offset, not
    a 353,127-element array, and means two runs of the same seed see windows in the same order
    regardless of where either was interrupted.
    """
    return np.random.default_rng([seed, epoch]).permutation(n)


def make_loader(ds: WindowDataset, order: np.ndarray, cfg: TrainConfig) -> DataLoader:
    """A sequential loader over a pre-permuted view: shuffling already happened in `order`.

    The order is handed to the dataset as a numpy array rather than wrapped in a `Subset`, which
    would pickle a 353,127-element Python list to each worker on every epoch.
    """
    return DataLoader(
        ds.reindexed(order),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )


def run_record(cfg: TrainConfig, out: Path, cache: CachePaths, tag: str) -> dict:
    meta = json.loads(cache.meta.read_text())
    return {
        "tag": tag,
        "config": asdict(cfg),
        "git": git_state(),
        "environment": environment(),
        "window_manifest": str(resolve_manifest(MANIFEST)),
        # Always the uncompressed hash, whichever form is on disk, so records
        # written before and after compression remain comparable.
        "window_manifest_sha256": manifest_sha256(MANIFEST),
        "cache_windows": meta["n"],
        "cache_manifest_hash": meta.get("manifest_hash"),
        "output_dir": str(out),
    }


def check_architecture(state: dict, cfg: TrainConfig) -> None:
    """Refuse to resume a checkpoint into a different front end. See D019.

    `raw_statistics` and `normalize_targets` change what the model computes but add and remove
    no parameter tensor, so a resume under today's defaults loads yesterday's weights without a
    word of complaint and silently continues the run as a different model. Half a run of one
    architecture and half of another is not a result, and nothing downstream would reveal it --
    the loss curve stays continuous.

    This fails rather than adapting. The flags are cheap to pass explicitly, and a resume that
    quietly rewrites what a run *is* is exactly the failure this project cannot afford.
    """
    trained = architecture_of(state)
    now = {name: getattr(cfg, name) for name in trained}
    if trained != now:
        raise SystemExit(
            f"refusing to resume: checkpoint was trained with {trained}, this run would use "
            f"{now}. Pass the matching flags, or start a new run under a new tag."
        )


def train(cfg: TrainConfig, out: Path, *, tag: str, resume: Path | None = None,
          max_epochs: int | None = None) -> Path:
    if cfg.deterministic:
        enable_determinism()
    out.mkdir(parents=True, exist_ok=True)
    cache = CachePaths.for_tag("strict_seed17")
    ds = WindowDataset(
        cache, seed=cfg.seed, augment_enabled=cfg.augment,
        dense_interpolation=cfg.dense_interpolation,
    )
    if cfg.exclude_dataset:
        # Tier-1 ablation 9: leave-one-dataset-out. Held-out windows are removed from pretraining
        # entirely, so the downstream evaluation on that cohort measures transfer to a population
        # the encoder has never seen -- which is the question a new sensor or clinic poses.
        codes = ds.dataset_code[ds.indices]
        vocab = list(ds.dataset_vocab)
        if cfg.exclude_dataset not in vocab:
            raise SystemExit(
                f"unknown dataset {cfg.exclude_dataset!r}; cache holds {vocab}"
            )
        keep = codes != vocab.index(cfg.exclude_dataset)
        print(f"excluding {cfg.exclude_dataset}: {len(ds.indices)} -> {int(keep.sum())} windows",
              flush=True)
        ds = ds.reindexed(ds.indices[keep])
    steps_per_epoch = len(ds) // cfg.batch_size

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trainer = Trainer(
        model=OpenCGMStateEvent(
            normalize_targets=cfg.normalize_targets, raw_statistics=cfg.raw_statistics,
            streams=cfg.streams, learnable_sigma=cfg.learnable_sigma,
            use_circadian=cfg.use_circadian, zero_empty_patches=cfg.zero_empty_patches,
        ),
        cfg=cfg, steps_per_epoch=steps_per_epoch,
    )

    start_epoch, start_step = 0, 0
    if resume is not None and resume.exists():
        state = torch.load(resume, map_location=cfg.device, weights_only=False)
        check_architecture(state, cfg)
        trainer.load_state_dict(state)
        start_epoch = state["epoch"]
        start_step = state["step_in_epoch"]
        print(f"resumed {resume} at epoch {start_epoch} step {start_step}", flush=True)
    else:
        (out / "run_record.json").write_text(
            json.dumps(run_record(cfg, out, cache, tag), indent=2)
        )
        save_checkpoint(trainer, out / "ckpt_ep000.pt", {"epoch": 0, "step_in_epoch": 0})

    metrics = (out / "metrics.jsonl").open("a")
    last_epoch = min(cfg.epochs, max_epochs) if max_epochs else cfg.epochs

    for epoch in range(start_epoch, last_epoch):
        trainer.model.train()  # belt and braces: `health` also restores it
        ds.set_epoch(epoch)
        order = epoch_permutation(len(ds), cfg.seed, epoch)
        offset = start_step * cfg.batch_size if epoch == start_epoch else 0
        loader = make_loader(ds, order[offset:], cfg)

        t0 = time.perf_counter()
        seen = 0
        agg: list[dict] = []
        for i, batch in enumerate(loader):
            step_in_epoch = i + (start_step if epoch == start_epoch else 0)
            m = trainer.step(batch, epoch, step_in_epoch)
            seen += batch[0].shape[0]
            agg.append(asdict(m))
            if cfg.log_every and trainer.global_step % cfg.log_every == 0:
                rec = asdict(m) | {"windows_per_sec": seen / (time.perf_counter() - t0)}
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                print(
                    f"ep{epoch:>3} {step_in_epoch:>5}/{steps_per_epoch}  loss {m.loss:.4f}  "
                    f"mcr {m.mcr:.4f}  td {m.td:.4f}  sigma {m.sigma:.4f}  "
                    f"|g| {m.grad_norm:.3f}  {rec['windows_per_sec']:,.0f} win/s",
                    flush=True,
                )
        start_step = 0

        summary = {
            "epoch": epoch,
            "seconds": time.perf_counter() - t0,
            "steps": len(agg),
            **{k: float(np.mean([a[k] for a in agg]))
               for k in ("loss", "mcr", "td", "sigma", "grad_norm", "realized_mask_ratio")},
            **{f"health_{k}": v for k, v in trainer.health(loader).items()},
        }
        trainer.history.append(summary)
        (out / "epochs.jsonl").open("a").write(json.dumps(summary) + "\n")
        print(
            f"== epoch {epoch} loss {summary['loss']:.5f} sigma {summary['sigma']:.4f} "
            f"rank {summary['health_effective_rank']:.2f} "
            f"({summary['seconds'] / 60:.1f} min)",
            flush=True,
        )

        save_checkpoint(trainer, out / "ckpt_last.pt", {"epoch": epoch + 1, "step_in_epoch": 0})
        if (epoch + 1) in cfg.checkpoint_epochs:
            save_checkpoint(
                trainer, out / f"ckpt_ep{epoch + 1:03d}.pt",
                {"epoch": epoch + 1, "step_in_epoch": 0},
            )

    metrics.close()
    return save_checkpoint(trainer, out / "ckpt_final.pt",
                           {"epoch": last_epoch, "step_in_epoch": 0})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--profile", default="paper_minimal")
    ap.add_argument("--amp", action="store_true", help="off by default: fp32 is faster here")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="~8% faster, but a seed no longer reproduces bit-for-bit")
    ap.add_argument("--normalized-statistics", action="store_true",
                    help="ablation: derive patch statistics from the normalised signal,\n"
                         "as we did before reading paper appendix C.2")
    ap.add_argument("--base-lr", type=float, default=None,
                    help="override the paper's 1e-4; any value other than the default makes\n"
                         "the run a labelled experiment, not the strict reproduction")
    ap.add_argument("--lambda-td", type=float, default=1.0,
                    help="weight on the temporal-dynamics loss; 0 disables it")
    ap.add_argument("--normalize-targets", action="store_true",
                    help="PROPOSED_EXTENSION: layer-norm EMA targets before regression")
    ap.add_argument("--streams", default="both", choices=["both", "state", "event", "raw"],
                    help="Tier-1 ablations 1-3: which stream reaches the fusion")
    ap.add_argument("--fixed-sigma", action="store_true",
                    help="Tier-1 ablation 7: freeze the Gaussian bandwidth at its initialisation")
    ap.add_argument("--no-circadian", action="store_true",
                    help="Tier-1 ablation 8: drop absolute time-of-day, keep patch position")
    ap.add_argument("--no-augment", action="store_true",
                    help="Tier-1 ablation 5: train on unaugmented windows")
    ap.add_argument("--dense-interpolation", action="store_true",
                    help="Tier-1 ablation 6: fill every gap, discard the physical mask")
    ap.add_argument("--exclude-dataset", default="",
                    help="Tier-1 ablation 9: leave-one-dataset-out pretraining")
    ap.add_argument("--nonzero-empty-patches", action="store_true",
                    help="ablation: let empty patches emit the learned bias (pre-D020 behaviour)")
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="strict")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = TrainConfig(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, profile=args.profile,
        amp=args.amp, num_workers=args.num_workers, log_every=args.log_every,
        deterministic=not args.nondeterministic,
        normalize_targets=args.normalize_targets,
        raw_statistics=not args.normalized_statistics,
        streams=args.streams,
        learnable_sigma=not args.fixed_sigma,
        use_circadian=not args.no_circadian,
        zero_empty_patches=not args.nonzero_empty_patches,
        augment=not args.no_augment,
        dense_interpolation=args.dense_interpolation,
        exclude_dataset=args.exclude_dataset,
        lambda_td=args.lambda_td,
        base_lr=args.base_lr if args.base_lr is not None else BASE_LR,
        # §17.2's modern-stable recipe is warmup + cosine decay + clipping. Defaults are
        # supplied here rather than in TrainConfig so the paper_minimal profile stays exactly
        # what the paper describes: constant LR, no warmup, no clipping.
        warmup_epochs=(
            args.warmup_epochs if args.warmup_epochs is not None
            else (5 if args.profile == 'modern_stable' else 0)
        ),
        grad_clip=(
            args.grad_clip if args.grad_clip is not None
            else (1.0 if args.profile == 'modern_stable' else None)
        ),
    )
    out = args.out or Path(f"runs/{args.tag}_seed{args.seed}")
    path = train(cfg, out, tag=args.tag, resume=args.resume, max_epochs=args.max_epochs)
    print(f"final checkpoint: {path}")


if __name__ == "__main__":
    main()
