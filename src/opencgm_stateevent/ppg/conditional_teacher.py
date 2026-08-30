"""Conditional teacher for the section-23 Lane-D PPG pilot (input-conditioned extension).

Variant of the section-23 teacher-student pilot. The marginal pilot feeds the frozen strict
ep40 teacher a zero-CGM window with mask=ones; its output is purely a function of positional
+ circadian embeddings. The conditional variant feeds the teacher the *actual* CGM context
window centered on the patch's timestamp, with mask=observed. The teacher's representation
is then a function of the CGM history and the target's value, so the alignment target the
student learns to predict from BVP is *conditional* rather than marginal.

Per-patch pipeline:
    1. Take the per-subject glucose CSV (mmol/L at 15-min native cadence).
    2. Convert to mg/dL (x 18.0182) and resample to the 5-min grid by nearest-within-2.5-min,
       leaving unobserved positions as zero with mask=0.
    3. Build a 288-position 24h window centered on the patch's timestamp, so the patch is at
       position 144, i.e. token index 12 of the 24 token-level outputs.
    4. Run the frozen teacher; the contextual token at index 12 is the alignment target.

The student architecture is unchanged from `ppg.heads.TeacherLatentHead` (projects 64-dim
BVP feature to 128-dim). Only the teacher's input and the choice of which teacher token
to match change.

This module is a PROPOSED_EXTENSION (D023 + A7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from .align import _parse_cgm_csv

#: Strict CGM pipeline is mg/dL; the PPG CSV is mmol/L. Convert at the boundary.
MMOL_TO_MGDL = 18.0182

#: 24h window at the 5-min grid.
WINDOW_POSITIONS = 288
#: 24 hourly patches.
PATCHES = 24
#: 12 grid positions per patch (1h).
STEPS_PER_PATCH = 12
#: 5 minutes per grid step.
GRID_MINUTES = 5
#: Center position of a 24h window centered on a patch's timestamp.
CENTER_POSITION = WINDOW_POSITIONS // 2  # 144
#: Center patch index.
CENTER_PATCH = CENTER_POSITION // STEPS_PER_PATCH  # 12


@dataclass(frozen=True)
class CgmContext:
    """A 24h CGM context window centered on one patch's timestamp."""

    values: np.ndarray  # (288,) mg/dL; 0 where unobserved
    mask: np.ndarray  # (288,) 1 where observed, 0 elsewhere
    n_observed: int  # how many of 288 positions are observed


def _cgm_rows_for_subject(data_zip_dir: Path, subject: str) -> list[tuple[datetime, float]]:
    """Read the per-subject glucose CSV and return (naive-local datetime, mmol/L) sorted ascend."""
    p = data_zip_dir / "Data" / subject / f"{subject}_glucose.csv"
    if not p.exists():
        return []
    return _parse_cgm_csv(p)


def _cgm_at_grid_step(
    rows: list[tuple[datetime, float]], t: datetime
) -> float | None:
    """Return mmol/L at the grid step t (5-min aligned), or None if no reading within ±2.5 min.

    The PPG dataset's CGM is 15-min native; readings land on :05, :20, :35, :50. ±2.5 min
    is the natural nearest-neighbour radius on the 5-min grid. This never interpolates.
    """
    if not rows:
        return None
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    best: tuple[datetime, float] | None = None
    best_dt: timedelta | None = None
    for cand in (lo - 1, lo):
        if 0 <= cand < len(rows):
            dt = abs(rows[cand][0] - t)
            if dt <= timedelta(minutes=2, seconds=30) and (
                best_dt is None or dt < best_dt
            ):
                best_dt = dt
                best = rows[cand]
    return best[1] if best is not None else None


def build_cgm_context(
    cgm_rows_mmol: list[tuple[datetime, float]],
    patch_timestamp: datetime,
) -> CgmContext:
    """Build a 24h CGM context window centered on `patch_timestamp`.

    The window spans [patch_timestamp - 12h, patch_timestamp + 12h) on the 5-min grid;
    position 144 is the patch's timestamp. Unobserved positions stay zero with mask=0.
    Values are in mg/dL (the strict pipeline's unit).
    """
    values = np.zeros(WINDOW_POSITIONS, dtype=np.float32)
    mask = np.zeros(WINDOW_POSITIONS, dtype=np.float32)
    window_start = patch_timestamp - timedelta(hours=12)
    n_observed = 0
    for i in range(WINDOW_POSITIONS):
        t = window_start + timedelta(minutes=i * GRID_MINUTES)
        mmol = _cgm_at_grid_step(cgm_rows_mmol, t)
        if mmol is not None:
            values[i] = mmol * MMOL_TO_MGDL
            mask[i] = 1.0
            n_observed += 1
    return CgmContext(values=values, mask=mask, n_observed=n_observed)


@torch.no_grad()
def encode_center_token(
    teacher,
    context: CgmContext,
    device: str,
) -> np.ndarray:
    """Run the frozen teacher on the context and return the center-patch token.

    Args:
        teacher: an `opencgm_stateevent.infer.Analyser` whose model has been frozen.
        context: a `CgmContext` from `build_cgm_context`.
        device: torch device string.

    Returns:
        128-dim numpy array: contextual_tokens[0, CENTER_PATCH, :].
    """
    values = torch.from_numpy(context.values).unsqueeze(0).to(device)
    mask = torch.from_numpy(context.mask).unsqueeze(0).to(device)
    # circadian_start=0 because the window is anchored to a clock time, not a session.
    # The teacher was trained with arbitrary circadian anchors and treats them as additive.
    circadian = torch.zeros(1, dtype=torch.long, device=device)
    out = teacher.model.encode(values, mask, circadian)
    return out.contextual_tokens[0, CENTER_PATCH, :].cpu().numpy()


def precompute_teacher_targets(
    teacher,
    data_zip_dir: Path,
    subjects: list[str],
    patches_iter,
    device: str,
    *,
    cache_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Pre-compute the per-patch teacher alignment targets.

    Args:
        teacher: frozen teacher Analyser.
        data_zip_dir: the dataset root (containing Data/P00x/...).
        subjects: list of subject IDs (P001..P005).
        patches_iter: iterable of `BvpCgmPatch` (or any object with `.subject` and
            `.timestamp_local`).
        device: torch device.
        cache_path: if given and the file exists, load targets from this .npz instead of
            re-computing. The file stores a (N, 128) array plus parallel arrays of
            `(subject, timestamp_iso)` keys.
        verbose: print progress.

    Returns:
        dict mapping `(subject, patch_timestamp)` -> 128-dim numpy array. The patch's
        timestamp is the patch's local-naive start datetime.
    """
    if cache_path is not None and Path(cache_path).exists():
        if verbose:
            print(f"loading cached targets from {cache_path} ...", flush=True)
        npz = np.load(cache_path, allow_pickle=True)
        targets: dict = {}
        for subj, ts_iso, vec in zip(
            npz["subjects"].tolist(),
            npz["timestamps"].tolist(),
            npz["vectors"],
            strict=True,
        ):
            ts = datetime.fromisoformat(ts_iso)
            targets[(subj, ts)] = vec
        if verbose:
            print(f"  loaded {len(targets)} targets from cache", flush=True)
        return targets

    cgm_rows_by_subject = {s: _cgm_rows_for_subject(data_zip_dir, s) for s in subjects}
    targets = {}
    if verbose:
        print("pre-computing conditional teacher targets ...", flush=True)
    n_total = 0
    n_skipped = 0
    for patch in patches_iter:
        key = (patch.subject, patch.timestamp_local)
        rows = cgm_rows_by_subject.get(patch.subject, [])
        ctx = build_cgm_context(rows, patch.timestamp_local)
        if ctx.n_observed < 1:
            # No CGM context at all for this patch. Skip the alignment target; the direct
            # glucose head still produces a usable regression target from the BVP.
            n_skipped += 1
            continue
        targets[key] = encode_center_token(teacher, ctx, device)
        n_total += 1
        if verbose and n_total % 500 == 0:
            print(f"  {n_total} targets computed ...", flush=True)
    if verbose:
        print(
            f"  done: {n_total} targets, {n_skipped} skipped (no CGM context)",
            flush=True,
        )

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(targets.keys())
        np.savez(
            cache_path,
            subjects=np.array([k[0] for k in keys], dtype=object),
            timestamps=np.array([k[1].isoformat() for k in keys], dtype=object),
            vectors=np.stack([targets[k] for k in keys]),
        )
        if verbose:
            print(f"  cached {len(targets)} targets to {cache_path}", flush=True)
    return targets
