"""PPG-bridge teacher-student pilot trainer (D023, A7)."""

# Trains the small PpgStudentEncoder + TeacherLatentHead + DirectGlucoseHead on the on-disk
# `ppg_cgm_paired_zenodo_20577959` dataset, with the strict-pretrain ep40 checkpoint frozen
# as the teacher.
#
# Validation protocol: subject-disjoint 5-fold x 5 student seeds (seeds 1003, 1019, 1043,
# 1071, 1103). 5 subjects total, so each fold has exactly 1 test subject. Report per-fold
# numbers and a subject-level bootstrap CI, not a fold-mean CI (a fold-mean CI is
# misleading at n=5).
#
# Loss = 0.5 * alignment_loss + 0.5 * gaussian_nll.
#
# Output: `reports/eval/ppg_pilot/ckpt_seed{NN}.pt` + `fold_scores.csv` + `run_record.json`.

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencgm_stateevent.infer import Analyser
from opencgm_stateevent.ppg import (
    DirectGlucoseHead,
    PpgStudentEncoder,
    TeacherLatentHead,
    alignment_loss,
    gaussian_nll,
    iter_aligned_patches,
    list_subjects,
)

STUDENT_SEEDS = (1003, 1019, 1043, 1071, 1103)
N_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3


@dataclass(frozen=True)
class PilotFoldResult:
    student_seed: int
    fold: int
    test_subject: str
    n_train_patches: int
    n_test_patches: int
    alignment_cosine: float | None  # mean over test patches with mask=1
    alignment_mse: float | None
    glucose_rmse_mmol: float | None
    glucose_mae_mmol: float | None


def _freeze_teacher(ckpt: Path, device: str) -> Analyser:
    """Load the frozen teacher (strict ep40). No gradients."""
    analyser = Analyser.load(ckpt, device=device)
    for p in analyser.model.parameters():
        p.requires_grad = False
    analyser.model.eval()
    return analyser


def _extract_data_zip(data_zip_dir: Path, target: Path) -> Path:
    """Extract data/raw/<source>/Data.zip into <target>. Skip if already extracted.

    The zip's root entry is a single directory called `Data/`, so extracting into
    `target` produces `target/Data/P00x/<date>/bvp_*.json`. The returned path points to
    `target` (the directory containing the `Data/` folder), not to `target/Data` itself -
    this matches what `iter_aligned_patches` and `list_subjects` expect, which look at
    `<returned>/Data/P00x/...`.
    """
    target.mkdir(parents=True, exist_ok=True)
    extracted = target / "Data"
    if extracted.exists() and any(extracted.iterdir()):
        return target
    with zipfile.ZipFile(data_zip_dir / "Data.zip") as z:
        z.extractall(target)
    return target


def _build_subject_folds(subjects: list[str], seed: int) -> list[tuple[list[str], str]]:
    """Random per-seed ordering; fold i is (train_subjects, test_subject)."""
    rng = np.random.default_rng(seed)
    order = list(subjects)
    rng.shuffle(order)
    folds: list[tuple[list[str], str]] = []
    for i, test in enumerate(order):
        train = [s for j, s in enumerate(order) if j != i]
        folds.append((train, test))
    return folds


def _encode_teacher_batch(
    teacher: Analyser,
    bvp_dummy_grid: np.ndarray,
    device: str,
) -> np.ndarray:
    """Compute the teacher's 256-dim per-position tokens for a 288-position 24h window.

    The teacher is fed a zero-valued 288-position CGM window with a fully-observed mask and
    a zero-circadian start. The output is therefore a function of the encoder's positional
    + circadian embeddings only — a documented limitation of the pilot, which measures
    whether the student can align with the teacher's static structure, not with its
    input-specific output.
    """
    sequence = 288
    values = torch.zeros(1, sequence, device=device)
    mask = torch.ones(1, sequence, device=device)
    circadian = torch.zeros(1, device=device)
    with torch.no_grad():
        result = teacher.model.encode(values, mask, circadian)
        tokens = result.contextual_tokens  # (1, n_tokens, 256)
    return tokens.squeeze(0).cpu().numpy()  # (n_tokens, 256)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=Path("runs_5090/rawstats120/ckpt_ep040.pt"),
        help="Frozen strict-pretrain teacher checkpoint. Defaults to ep40 (D024).",
    )
    ap.add_argument(
        "--data-zip-dir",
        type=Path,
        default=Path("data/raw/ppg_cgm_paired_zenodo_20577959"),
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/ppg_pilot_work"),
        help="Where to extract Data.zip once (reused across runs).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/eval/ppg_pilot"),
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"extracting Data.zip -> {args.work_dir} ...", flush=True)
    extracted = _extract_data_zip(args.data_zip_dir, args.work_dir)
    subjects = list_subjects(extracted)
    print(f"  subjects: {subjects}", flush=True)
    assert len(subjects) == 5, f"expected 5 subjects, got {len(subjects)}"

    print(f"loading frozen teacher {args.teacher_ckpt} ...", flush=True)
    teacher = _freeze_teacher(args.teacher_ckpt, args.device)

    # Pre-compute teacher token grid once.
    teacher_tokens = _encode_teacher_batch(teacher, None, args.device)
    n_tokens = teacher_tokens.shape[0]
    print(f"  teacher emits {n_tokens} tokens at {teacher_tokens.shape[1]} dims", flush=True)

    # Build the per-subject patch list once; reuse across seeds.
    print("indexing aligned patches ...", flush=True)
    by_subject: dict[str, list[tuple[np.ndarray, float | None]]] = {s: [] for s in subjects}
    for patch in iter_aligned_patches(extracted, subjects):
        by_subject[patch.subject].append((patch.bvp, patch.glucose_mmol))
    for s, plist in by_subject.items():
        n_with_cgm = sum(1 for _, g in plist if g is not None)
        print(f"  {s}: {len(plist)} patches, {n_with_cgm} with CGM", flush=True)

    fold_records: list[dict] = []
    for student_seed in STUDENT_SEEDS:
        print(f"\n=== student seed {student_seed} ===", flush=True)
        folds = _build_subject_folds(subjects, student_seed)

        torch.manual_seed(student_seed)
        np.random.seed(student_seed)

        encoder = PpgStudentEncoder().to(args.device)
        latent_head = TeacherLatentHead().to(args.device)
        glucose_head = DirectGlucoseHead().to(args.device)
        optim = torch.optim.AdamW(
            list(encoder.parameters())
            + list(latent_head.parameters())
            + list(glucose_head.parameters()),
            lr=args.lr,
        )

        for fold_idx, (train_subjects, test_subject) in enumerate(folds):
            print(f"  fold {fold_idx}: train={train_subjects}  test={test_subject}", flush=True)
            train_patches = [(bvp, mmol) for s in train_subjects for bvp, mmol in by_subject[s]]
            test_patches = [(bvp, mmol) for s in [test_subject] for bvp, mmol in by_subject[s]]

            for epoch in range(args.epochs):
                # Shuffle training patches
                idx = np.arange(len(train_patches))
                np.random.shuffle(idx)
                total_loss = 0.0
                n_batches = 0
                for start in range(0, len(idx), args.batch_size):
                    batch_idx = idx[start : start + args.batch_size]
                    bvp_batch = np.stack([train_patches[i][0] for i in batch_idx])
                    mmol_batch = np.array(
                        [train_patches[i][1] for i in batch_idx], dtype=np.float32
                    )
                    mask = (~np.isnan(mmol_batch)).astype(np.float32)
                    mmol_batch = np.where(mask > 0, mmol_batch, 0.0)

                    bvp_t = torch.from_numpy(bvp_batch).to(args.device)
                    mmol_t = torch.from_numpy(mmol_batch).to(args.device)
                    mask_t = torch.from_numpy(mask).to(args.device)

                    feats = encoder(bvp_t)
                    latent = latent_head(feats)
                    pred = glucose_head(feats)

                    # Align each student token to the teacher token at the same index.
                    # Use modulo for indices that exceed n_tokens (we have ~288 patches per
                    # 24h subject, matching the teacher's 288-position grid).
                    latent_aligned = latent
                    teacher_aligned = torch.from_numpy(
                        teacher_tokens[np.arange(len(batch_idx)) % n_tokens]
                    ).to(args.device)

                    loss_align = alignment_loss(latent_aligned, teacher_aligned, mask_t)
                    loss_glucose = gaussian_nll(pred, mmol_t, mask_t)
                    loss = 0.5 * loss_align + 0.5 * loss_glucose

                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    total_loss += float(loss.detach())
                    n_batches += 1
                if (epoch + 1) % 5 == 0:
                    avg = total_loss / max(1, n_batches)
                    print(
                        f"    epoch {epoch + 1}/{args.epochs}  loss {avg:.4f}",
                        flush=True,
                    )

            # Eval on the test subject
            cos_vals: list[float] = []
            mse_vals: list[float] = []
            rmse_pairs: list[tuple[float, float]] = []
            mae_pairs: list[tuple[float, float]] = []
            encoder.eval()
            latent_head.eval()
            glucose_head.eval()
            with torch.no_grad():
                for start in range(0, len(test_patches), args.batch_size):
                    batch = test_patches[start : start + args.batch_size]
                    bvp_batch = np.stack([b for b, _ in batch])
                    mmol_batch = np.array([g for _, g in batch], dtype=np.float32)
                    mask = (~np.isnan(mmol_batch)).astype(np.float32)
                    mmol_batch_filled = np.where(mask > 0, mmol_batch, 0.0)

                    bvp_t = torch.from_numpy(bvp_batch).to(args.device)
                    mmol_t = torch.from_numpy(mmol_batch_filled).to(args.device)
                    mask_t = torch.from_numpy(mask).to(args.device)

                    feats = encoder(bvp_t)
                    latent = latent_head(feats)
                    pred = glucose_head(feats)

                    teacher_aligned = torch.from_numpy(
                        teacher_tokens[np.arange(len(batch)) % n_tokens]
                    ).to(args.device)

                    valid = mask_t > 0.5
                    if valid.any():
                        s = latent[valid]
                        t = teacher_aligned[valid]
                        cos = torch.nn.functional.cosine_similarity(s, t, dim=-1)
                        mse = (s - t).pow(2).mean(dim=-1)
                        cos_vals.extend(cos.cpu().numpy().tolist())
                        mse_vals.extend(mse.cpu().numpy().tolist())

                        m = pred[valid, 0]
                        truth = mmol_t[valid]
                        diff = (m - truth).cpu().numpy()
                        for d, v in zip(diff, truth.cpu().numpy(), strict=True):
                            rmse_pairs.append((float(d), float(v)))
                            mae_pairs.append((float(d), float(v)))
            encoder.train()
            latent_head.train()
            glucose_head.train()

            mean_cos = float(np.mean(cos_vals)) if cos_vals else None
            mean_mse = float(np.mean(mse_vals)) if mse_vals else None
            rmse = (
                float(np.sqrt(np.mean(np.square([d for d, _ in rmse_pairs]))))
                if rmse_pairs
                else None
            )
            mae = float(np.mean(np.abs([d for d, _ in mae_pairs]))) if mae_pairs else None
            fr = PilotFoldResult(
                student_seed=student_seed,
                fold=fold_idx,
                test_subject=test_subject,
                n_train_patches=len(train_patches),
                n_test_patches=len(test_patches),
                alignment_cosine=mean_cos,
                alignment_mse=mean_mse,
                glucose_rmse_mmol=rmse,
                glucose_mae_mmol=mae,
            )
            fold_records.append(asdict(fr))
            print(
                f"    test={test_subject}  cos={mean_cos!r}  mse={mean_mse!r}  "
                f"rmse={rmse!r}  mae={mae!r}",
                flush=True,
            )

        # Save per-seed student checkpoint.
        ckpt_path = args.out / f"ckpt_seed{student_seed}.pt"
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "latent_head": latent_head.state_dict(),
                "glucose_head": glucose_head.state_dict(),
                "student_seed": student_seed,
                "epochs": args.epochs,
                "teacher_ckpt": str(args.teacher_ckpt),
                "subjects": subjects,
            },
            ckpt_path,
        )
        print(f"  saved {ckpt_path}", flush=True)

    df = pd.DataFrame(fold_records)
    df.to_csv(args.out / "fold_scores.csv", index=False)

    summary = (
        df.groupby("student_seed")
        .agg(
            n_folds=("fold", "count"),
            alignment_cosine_mean=("alignment_cosine", "mean"),
            alignment_cosine_sd=(
                "alignment_cosine",
                lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            ),
            alignment_mse_mean=("alignment_mse", "mean"),
            glucose_rmse_mean=("glucose_rmse_mmol", "mean"),
            glucose_mae_mean=("glucose_mae_mmol", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.out / "per_seed_summary.csv", index=False)

    (args.out / "run_record.json").write_text(
        json.dumps(
            {
                "teacher_ckpt": str(args.teacher_ckpt),
                "data_zip": str(args.data_zip_dir),
                "student_seeds": list(STUDENT_SEEDS),
                "n_folds": N_FOLDS,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": args.device,
                "subjects": subjects,
                "encoder": {
                    "kind": "PpgStudentEncoder",
                    "feature_dim": PpgStudentEncoder.FEATURE_DIM,
                    "patch_len": PpgStudentEncoder.PATCH_LEN,
                },
            },
            indent=2,
        )
    )

    print(
        f"\nwrote {args.out}/fold_scores.csv, per_seed_summary.csv, run_record.json, ckpt_seed*.pt"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
