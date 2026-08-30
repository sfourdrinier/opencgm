"""PPG teacher-student pilot, INPUT-CONDITIONED variant (D023, A7 extension).

# Like `ppg_teacher_student.py` but the teacher's input is the *actual* 24h CGM context
# window centered on the patch's timestamp, with mask=observed. The teacher therefore
# produces a representation that depends on the CGM history and the target's glucose
# value, not just positional + circadian priors. The student is asked to match that
# conditional representation from BVP alone.

# The student architecture is identical to the marginal pilot (PpgStudentEncoder +
# TeacherLatentHead + DirectGlucoseHead). The only change is the teacher's input and
# which teacher token (the center patch) is the alignment target.

# Same 5-fold x 5-seed protocol as the marginal pilot for direct comparability.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
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
    precompute_teacher_targets,
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
    n_train_targets: int
    n_test_targets: int
    alignment_cosine: float | None
    alignment_mse: float | None
    glucose_rmse_mmol: float | None
    glucose_mae_mmol: float | None


def _freeze_teacher(ckpt: Path, device: str) -> Analyser:
    analyser = Analyser.load(ckpt, device=device)
    for p in analyser.model.parameters():
        p.requires_grad = False
    analyser.model.eval()
    return analyser


def _extract_data_zip(data_zip_dir: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    extracted = target / "Data"
    if extracted.exists() and any(extracted.iterdir()):
        return target
    with zipfile.ZipFile(data_zip_dir / "Data.zip") as z:
        z.extractall(target)
    return target


def _build_subject_folds(subjects: list[str], seed: int) -> list[tuple[list[str], str]]:
    rng = np.random.default_rng(seed)
    order = list(subjects)
    rng.shuffle(order)
    folds: list[tuple[list[str], str]] = []
    for i, test in enumerate(order):
        train = [s for j, s in enumerate(order) if j != i]
        folds.append((train, test))
    return folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=Path("runs_5090/rawstats120/ckpt_ep040.pt"),
        help="Frozen strict-pretrain teacher checkpoint.",
    )
    ap.add_argument(
        "--data-zip-dir",
        type=Path,
        default=Path("data/raw/ppg_cgm_paired_zenodo_20577959"),
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/ppg_pilot_conditional_work"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/eval/ppg_pilot_conditional"),
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--teacher-targets-cache",
        type=Path,
        default=Path("artifacts/ppg_teacher_targets.npz"),
        help=(
            "Cache file for the precomputed teacher targets. If the file exists, training "
            "loads from it; otherwise it is computed once and saved here."
        ),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"extracting Data.zip -> {args.work_dir} ...", flush=True)
    extracted = _extract_data_zip(args.data_zip_dir, args.work_dir)
    subjects = list_subjects(extracted)
    print(f"  subjects: {subjects}", flush=True)
    assert len(subjects) == 5, f"expected 5 subjects, got {len(subjects)}"

    print(f"loading frozen teacher {args.teacher_ckpt} ...", flush=True)
    teacher = _freeze_teacher(args.teacher_ckpt, args.device)

    # Index aligned patches once.
    print("indexing aligned patches ...", flush=True)
    all_patches = list(iter_aligned_patches(extracted, subjects))
    print(f"  {len(all_patches)} patches indexed", flush=True)

    # Pre-compute the per-patch conditional teacher targets.
    targets = precompute_teacher_targets(
        teacher, extracted, subjects, all_patches, args.device,
        cache_path=args.teacher_targets_cache,
    )
    print(f"  {len(targets)} conditional targets ready", flush=True)

    # Build per-subject patch list with the (subject, timestamp) key for lookup.
    # Each entry preserves the subject id (avoids the train/test subject lookup in the batch loop).
    by_subject: dict[str, list[tuple[np.ndarray, float | None, datetime]]] = {
        s: [] for s in subjects
    }
    for patch in all_patches:
        by_subject[patch.subject].append(
            (patch.bvp, patch.glucose_mmol, patch.timestamp_local)
        )
    for s, plist in by_subject.items():
        n_with_cgm = sum(1 for _, g, _ in plist if g is not None)
        n_with_target = sum(1 for _, _, t in plist if (s, t) in targets)
        n_with_target_str = f"{n_with_target} with target"
        print(
            f"  {s}: {len(plist)} patches, {n_with_cgm} with CGM, "
            f"{n_with_target_str}",
            flush=True,
        )

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
            train_items: list[tuple[str, np.ndarray, float | None, datetime]] = [
                (s, bvp, mmol, ts)
                for s in train_subjects
                for bvp, mmol, ts in by_subject[s]
            ]
            test_items: list[tuple[str, np.ndarray, float | None, datetime]] = [
                (test_subject, bvp, mmol, ts)
                for bvp, mmol, ts in by_subject[test_subject]
            ]
            train_targets_keys = [(s, ts) for s, _, _, ts in train_items]
            test_targets_keys = [(s, ts) for s, _, _, ts in test_items]
            n_train_targets = sum(1 for k in train_targets_keys if k in targets)
            n_test_targets = sum(1 for k in test_targets_keys if k in targets)

            for epoch in range(args.epochs):
                idx = np.arange(len(train_items))
                np.random.shuffle(idx)
                total_loss = 0.0
                n_batches = 0
                for start in range(0, len(idx), args.batch_size):
                    batch_idx = idx[start : start + args.batch_size]
                    bvp_batch = np.stack([train_items[i][1] for i in batch_idx])
                    mmol_batch = np.array(
                        [train_items[i][2] for i in batch_idx], dtype=np.float32
                    )
                    key_batch = [(train_items[i][0], train_items[i][3]) for i in batch_idx]
                    cgm_mask = (~np.isnan(mmol_batch)).astype(np.float32)
                    align_mask = np.array(
                        [1.0 if k in targets else 0.0 for k in key_batch], dtype=np.float32
                    )
                    mmol_batch_filled = np.where(cgm_mask > 0, mmol_batch, 0.0)
                    teacher_batch = np.stack(
                        [targets[k] if k in targets else np.zeros(128, dtype=np.float32)
                         for k in key_batch]
                    )

                    bvp_t = torch.from_numpy(bvp_batch).to(args.device)
                    mmol_t = torch.from_numpy(mmol_batch_filled).to(args.device)
                    cgm_mask_t = torch.from_numpy(cgm_mask).to(args.device)
                    align_mask_t = torch.from_numpy(align_mask).to(args.device)
                    teacher_t = torch.from_numpy(teacher_batch).to(args.device)

                    feats = encoder(bvp_t)
                    latent = latent_head(feats)
                    pred = glucose_head(feats)

                    loss_align = alignment_loss(latent, teacher_t, align_mask_t)
                    loss_glucose = gaussian_nll(pred, mmol_t, cgm_mask_t)
                    loss = 0.5 * loss_align + 0.5 * loss_glucose

                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    total_loss += float(loss.detach())
                    n_batches += 1
                if (epoch + 1) % 5 == 0:
                    avg = total_loss / max(1, n_batches)
                    print(f"    epoch {epoch + 1}/{args.epochs}  loss {avg:.4f}", flush=True)

            # Eval on the test subject
            cos_vals: list[float] = []
            mse_vals: list[float] = []
            rmse_pairs: list[tuple[float, float]] = []
            mae_pairs: list[tuple[float, float]] = []
            encoder.eval()
            latent_head.eval()
            glucose_head.eval()
            with torch.no_grad():
                for start in range(0, len(test_items), args.batch_size):
                    batch = test_items[start : start + args.batch_size]
                    bvp_batch = np.stack([b for _, b, _, _ in batch])
                    mmol_batch = np.array([g for _, _, g, _ in batch], dtype=np.float32)
                    key_batch = [(s, ts) for s, _, _, ts in batch]
                    cgm_mask = (~np.isnan(mmol_batch)).astype(np.float32)
                    align_mask = np.array(
                        [1.0 if k in targets else 0.0 for k in key_batch], dtype=np.float32
                    )
                    mmol_batch_filled = np.where(cgm_mask > 0, mmol_batch, 0.0)
                    teacher_batch = np.stack(
                        [targets[k] if k in targets else np.zeros(128, dtype=np.float32)
                         for k in key_batch]
                    )

                    bvp_t = torch.from_numpy(bvp_batch).to(args.device)
                    mmol_t = torch.from_numpy(mmol_batch_filled).to(args.device)
                    cgm_mask_t = torch.from_numpy(cgm_mask).to(args.device)
                    align_mask_t = torch.from_numpy(align_mask).to(args.device)
                    teacher_t = torch.from_numpy(teacher_batch).to(args.device)

                    feats = encoder(bvp_t)
                    latent = latent_head(feats)
                    pred = glucose_head(feats)

                    valid = (align_mask_t > 0.5) & (cgm_mask_t > 0.5)
                    if valid.any():
                        s_lat = latent[valid]
                        t_lat = teacher_t[valid]
                        cos = torch.nn.functional.cosine_similarity(s_lat, t_lat, dim=-1)
                        mse = (s_lat - t_lat).pow(2).mean(dim=-1)
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
                n_train_patches=len(train_items),
                n_test_patches=len(test_items),
                n_train_targets=n_train_targets,
                n_test_targets=n_test_targets,
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
                "conditional": True,
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
                "teacher_kind": "conditional",
                "teacher_target": (
                    "contextual_tokens[:, 12, :] of 24h CGM context window centered on patch"
                ),
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
