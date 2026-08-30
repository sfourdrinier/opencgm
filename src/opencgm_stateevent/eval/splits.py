"""Subject-grouped cross-validation folds. Blueprint §9.5, §19.4 — `PAPER_EXACT`.

Five folds, ten repeated fold assignments, no subject crossing train and test, and — the part
that makes comparisons meaningful — *identical folds for every compared representation*. A fold
assignment is therefore a function of `(task, repeat)` only. It never sees the embeddings, so a
baseline and the model are scored on exactly the same partitions and paired tests are valid.

Splits are computed over subjects and then broadcast to windows. Doing it the other way round is
the standard way subject leakage enters an evaluation: a window-level split puts two days from
the same person on both sides, and the probe scores the person rather than the representation.

Folds are stratified by label where that is possible. With eight positives in a 57-subject task
(`hall:hyperlipidemia`) stratification is what keeps a fold from containing zero positives and
producing an undefined PR-AUC.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

N_FOLDS = 5  # PAPER_EXACT §19.4
N_REPEATS = 10  # PAPER_EXACT §19.4


def repeat_seed(task_key: str, repeat: int) -> int:
    """Deterministic seed from the task name and repeat index.

    Derived rather than drawn from a shared stream so that adding, removing or reordering tasks
    cannot change the folds of any other task. A shared `RandomState` walked in loop order would
    silently re-randomise every downstream result the first time a task list changed.
    """
    digest = hashlib.sha256(f"{task_key}|{repeat}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class Fold:
    task: str
    repeat: int
    fold: int
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]

    def check_disjoint(self) -> None:
        overlap = set(self.train_subjects) & set(self.test_subjects)
        if overlap:
            raise ValueError(
                f"{self.task} r{self.repeat} f{self.fold}: subjects on both sides: {overlap}"
            )


def stratified_subject_folds(
    subjects: np.ndarray, labels: np.ndarray, *, seed: int, n_folds: int = N_FOLDS
) -> list[np.ndarray]:
    """Assign each subject to a fold, balancing classes across folds.

    Subjects are shuffled within each class and dealt round-robin, which spreads a small minority
    class as evenly as it can be spread. With eight positives over five folds the result is two
    folds of two and three of one — not balanced, but never zero, which is the property that
    matters for PR-AUC.
    """
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(subjects), dtype=int)
    for value in np.unique(labels):
        members = np.flatnonzero(labels == value)
        rng.shuffle(members)
        # Rotate the starting fold per class so the same fold does not absorb the remainder of
        # every class at once.
        offset = int(rng.integers(n_folds))
        assignment[members] = (np.arange(len(members)) + offset) % n_folds
    return [np.flatnonzero(assignment == f) for f in range(n_folds)]


def build_folds(
    task_key: str,
    subjects: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int = N_FOLDS,
    n_repeats: int = N_REPEATS,
) -> list[Fold]:
    """All `n_repeats * n_folds` folds for one task.

    `subjects` and `labels` are per subject and already filtered to those carrying a label.
    """
    if len(subjects) != len(labels):
        raise ValueError("subjects and labels must be parallel")
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{task_key}: only one class present")

    folds: list[Fold] = []
    for repeat in range(n_repeats):
        groups = stratified_subject_folds(
            subjects, labels, seed=repeat_seed(task_key, repeat), n_folds=n_folds
        )
        for f, test_idx in enumerate(groups):
            train_idx = np.setdiff1d(np.arange(len(subjects)), test_idx)
            fold = Fold(
                task=task_key,
                repeat=repeat,
                fold=f,
                train_subjects=tuple(subjects[train_idx]),
                test_subjects=tuple(subjects[test_idx]),
            )
            fold.check_disjoint()
            folds.append(fold)
    return folds


def window_mask(window_subjects: np.ndarray, subjects: tuple[str, ...]) -> np.ndarray:
    """Broadcast a subject set to a boolean mask over windows."""
    return np.isin(window_subjects, np.asarray(subjects))


def fold_manifest(folds: list[Fold]) -> dict:
    """A hashable record of the partition, written beside every result.

    Two methods claiming to have been compared on identical folds should produce identical
    manifest hashes. If they do not, the comparison is not paired and the statistics in §19.5 do
    not apply.
    """
    payload = "\n".join(
        f"{f.task}|{f.repeat}|{f.fold}|{','.join(sorted(f.test_subjects))}"
        for f in sorted(folds, key=lambda x: (x.task, x.repeat, x.fold))
    )
    return {
        "n_folds": len({f.fold for f in folds}),
        "n_repeats": len({f.repeat for f in folds}),
        "tasks": sorted({f.task for f in folds}),
        "hash": hashlib.sha256(payload.encode()).hexdigest(),
    }
