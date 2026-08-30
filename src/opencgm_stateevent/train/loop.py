"""Pretraining loop. Blueprint §17.

The paper publishes epochs, batch size, both learning rates and weight decay. Optimizer,
betas, schedule, warmup, clipping and precision are not named, so the reference profile marks
them INFERRED_RECONSTRUCTION and the modern-stable profile (§17.2) exists as a separate,
separately-labelled recipe rather than a quiet substitution.

Two ordering requirements that are easy to get wrong and silent when wrong:

* the EMA target updates **after** the optimizer step (§14.1), not before;
* rho carries its own learning rate and zero weight decay (§17.1) — decaying a bandwidth
  parameter would pull sigma toward the middle of its range for reasons unrelated to the data.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..model.model import OpenCGMStateEvent

BASE_LR = 1e-4  # PAPER_EXACT §17.1
RHO_LR = 1e-3  # PAPER_EXACT §17.1
WEIGHT_DECAY = 1e-2  # PAPER_EXACT §17.1
GLOBAL_BATCH = 128  # PAPER_EXACT §17.1
EPOCHS = 120  # PAPER_EXACT §17.1


@dataclass
class TrainConfig:
    seed: int = 17
    epochs: int = EPOCHS
    batch_size: int = GLOBAL_BATCH
    base_lr: float = BASE_LR
    rho_lr: float = RHO_LR
    weight_decay: float = WEIGHT_DECAY
    lambda_td: float = 1.0
    # profile: "paper_minimal" (constant LR, no warmup, no clipping) or "modern_stable"
    profile: str = "paper_minimal"
    warmup_epochs: int = 0
    grad_clip: float | None = None
    amp: bool = False
    #: deterministic CUDA kernels, so two runs of a seed are bit-identical. Costs ~8%.
    deterministic: bool = True
    #: PROPOSED_EXTENSION, off for the strict reproduction. See D018.
    normalize_targets: bool = False
    #: paper appendix C.2 computes patch statistics from raw mg/dL, not the normalised signal
    raw_statistics: bool = True
    #: Tier-1 ablations 1-3, 7, 8. "both" is the reproduction; the rest are labelled variants.
    streams: str = "both"
    learnable_sigma: bool = True
    use_circadian: bool = True
    #: appendix C.2: "Empty patches are zeroed by the validity mask". See D020.
    zero_empty_patches: bool = True
    #: Tier-1 ablations 5, 6, 9. Data-side variants; none is part of the strict reproduction.
    augment: bool = True
    dense_interpolation: bool = False
    exclude_dataset: str = ""
    num_workers: int = 6
    device: str = "cuda"
    log_every: int = 50
    checkpoint_epochs: tuple[int, ...] = (0, 1, 5, 10, 20, 40, 60, 80, 100, 120)


def enable_determinism() -> None:
    """Make CUDA kernels deterministic, so a seed reproduces bit-for-bit.

    Without this, the atomics in several backward kernels make float32 results depend on thread
    scheduling. Two runs of the same seed then agree to about seven significant figures and drift
    apart slowly -- close enough to look reproducible in a plot, and not close enough to be.

    Measured cost on the RTX 3090: 23.0 -> 24.9 ms/step, about 8%. Cheap for the property.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before cuBLAS initialises, which is why this is called
    at the top of the entrypoint rather than lazily.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


@dataclass
class StepMetrics:
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    mcr: float = 0.0
    td: float = 0.0
    sigma: float = 0.0
    rho_grad: float = 0.0
    grad_norm: float = 0.0
    ema_momentum: float = 0.0
    lr: float = 0.0
    realized_mask_ratio: float = 0.0
    windows_per_sec: float = 0.0
    #: representation health, §15.6
    target_latent_std: float = 0.0
    effective_rank: float = 0.0


def build_optimizer(model: OpenCGMStateEvent, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Two groups. rho is excluded from weight decay and gets the paper's higher LR."""
    rho = [model.online.gaussian.rho]
    rho_ids = {id(p) for p in rho}
    ordinary = [p for p in model.parameters() if p.requires_grad and id(p) not in rho_ids]
    return torch.optim.AdamW(
        [
            {"params": ordinary, "lr": cfg.base_lr, "weight_decay": cfg.weight_decay},
            {"params": rho, "lr": cfg.rho_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999) if cfg.profile == "paper_minimal" else (0.9, 0.95),
        eps=1e-8,
    )


def lr_scale(cfg: TrainConfig, epoch: int, step_in_epoch: int, steps_per_epoch: int) -> float:
    """Constant for paper_minimal; warmup + cosine for modern_stable (§17.2)."""
    if cfg.profile == "paper_minimal":
        return 1.0
    progress = (epoch + step_in_epoch / max(1, steps_per_epoch)) / max(1, cfg.epochs)
    if epoch < cfg.warmup_epochs:
        return (epoch + step_in_epoch / max(1, steps_per_epoch)) / max(1, cfg.warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def spectrum(x: torch.Tensor) -> tuple[float, float, float]:
    """Effective rank of the latent **covariance**, plus how concentrated it is. §15.6.

    Returns ``(effective_rank, top_eigenvalue_fraction, top4_fraction)``.

    Covariance eigenvalues are the *squares* of the centred singular values. Normalising the
    singular values directly measures the entropy of the wrong distribution and inflates the
    result badly: on data where a single direction carries 99.9% of the variance, the singular
    value form reports 4.37 and the correct form reports 1.02. That error hid a real dimensional
    collapse behind a number that appeared to be improving, so the top-eigenvalue fraction is
    reported alongside — a rank that looks healthy while one component explains 90% of the
    variance is not healthy.
    """
    x = x.reshape(-1, x.shape[-1]).float()
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return 0.0, 1.0, 1.0
    eigenvalues = torch.linalg.svdvals(x).square()
    p = eigenvalues / (eigenvalues.sum() + 1e-12)
    nonzero = p[p > 0]
    rank = float(torch.exp(-(nonzero * nonzero.log()).sum()))
    return rank, float(p[0]), float(p[:4].sum())


def effective_rank(x: torch.Tensor) -> float:
    return spectrum(x)[0]


@dataclass
class Trainer:
    model: OpenCGMStateEvent
    cfg: TrainConfig
    steps_per_epoch: int
    optimizer: torch.optim.Optimizer = field(init=False)
    scaler: torch.amp.GradScaler = field(init=False)
    global_step: int = 0
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.amp)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.cfg.seed)
        self.max_steps = self.steps_per_epoch * self.cfg.epochs

    def step(self, batch, epoch: int, step_in_epoch: int) -> StepMetrics:
        values, mask, circadian = batch
        values = values.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        circadian = circadian.to(self.device, non_blocking=True)

        scale = lr_scale(self.cfg, epoch, step_in_epoch, self.steps_per_epoch)
        for group, base in zip(
            self.optimizer.param_groups, (self.cfg.base_lr, self.cfg.rho_lr), strict=True
        ):
            group["lr"] = base * scale

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.cfg.amp):
            out = self.model(values, mask, circadian, self.generator, lambda_td=self.cfg.lambda_td)

        self.scaler.scale(out.loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.trainable_parameters(),
            self.cfg.grad_clip if self.cfg.grad_clip else float("inf"),
        )
        # `rho` has no gradient in two Tier-1 ablations: `--fixed-sigma` freezes it, and
        # `--streams raw` bypasses the filter entirely so nothing upstream of it is reached.
        # Both are legitimate configurations, and reporting 0.0 for "how hard is the bandwidth
        # being pushed" is correct in each -- it is not being pushed at all.
        rho = self.model.online.gaussian.rho
        rho_grad = float(rho.grad.abs().item()) if rho.grad is not None else 0.0
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # EMA strictly after the optimizer step, blueprint §14.1
        m = self.model.target.momentum(self.global_step, self.max_steps)
        self.model.target.update(self.model.online, m)
        self.global_step += 1

        return StepMetrics(
            step=self.global_step,
            epoch=epoch,
            loss=float(out.loss.detach()),
            mcr=float(out.mcr.detach()),
            td=float(out.td.detach()),
            sigma=float(self.model.online.gaussian.sigma.detach()),
            rho_grad=rho_grad,
            grad_norm=float(grad_norm),
            ema_momentum=m,
            lr=self.optimizer.param_groups[0]["lr"],
            realized_mask_ratio=float(out.realized_mask_ratio.mean()),
        )

    def run_epoch(self, loader: DataLoader, epoch: int, *, max_steps: int | None = None) -> dict:
        self.model.train()
        t0 = time.perf_counter()
        seen = 0
        agg: list[StepMetrics] = []
        for i, batch in enumerate(loader):
            if max_steps is not None and i >= max_steps:
                break
            m = self.step(batch, epoch, i)
            seen += batch[0].shape[0]
            agg.append(m)
            if self.cfg.log_every and self.global_step % self.cfg.log_every == 0:
                el = time.perf_counter() - t0
                print(
                    f"  ep{epoch:>3} step {self.global_step:>7}  loss {m.loss:.4f}  "
                    f"mcr {m.mcr:.4f}  td {m.td:.4f}  sigma {m.sigma:.3f}  "
                    f"{seen / el:,.0f} win/s",
                    flush=True,
                )
        elapsed = time.perf_counter() - t0
        summary = {
            "epoch": epoch,
            "steps": len(agg),
            "windows_per_sec": seen / max(elapsed, 1e-9),
            "seconds": elapsed,
            **{
                k: float(np.mean([getattr(a, k) for a in agg]))
                for k in ("loss", "mcr", "td", "sigma", "rho_grad", "grad_norm",
                          "realized_mask_ratio")
            },
        }
        self.history.append(summary)
        return summary

    @torch.no_grad()
    def health(self, loader: DataLoader) -> dict:
        """Representation-health diagnostics. Blueprint §15.6.

        Restores the previous training mode on the way out. Leaving the model in eval mode
        silently disables dropout for every subsequent epoch, which shows up as a loss that drops
        at the first epoch boundary and then looks perfectly healthy.
        """
        was_training = self.model.training
        self.model.eval()
        values, mask, circadian = next(iter(loader))
        out = self.model.encode(
            values.to(self.device), mask.to(self.device), circadian.to(self.device)
        )
        z = out.daily_embedding
        self.model.train(was_training)
        rank, top1, top4 = spectrum(z)
        return {
            "latent_std_mean": float(z.std(dim=0).mean()),
            "latent_std_min": float(z.std(dim=0).min()),
            "effective_rank": rank,
            "top_eigenvalue_fraction": top1,
            "top4_eigenvalue_fraction": top4,
            "dims": int(z.shape[-1]),
            "sigma": float(out.sigma),
        }

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "generator": self.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "config": asdict(self.cfg),
            "history": self.history,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        # RNG states are ByteTensors and must live on the CPU. Loading a checkpoint with
        # ``map_location="cuda"`` moves every tensor in it, these included, and ``set_state``
        # then rejects them. A CPU-only test cannot see this, because it loads to CPU anyway.
        self.generator.set_state(state["generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"])
        self.history = state.get("history", [])


def save_checkpoint(trainer: Trainer, path: Path, extra: dict | None = None) -> Path:
    """Atomic write, so an interrupted save cannot leave an unloadable checkpoint. §17.5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = trainer.state_dict()
    if extra:
        state.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path
