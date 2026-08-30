"""The §19.5 macro-averaged comparison, and what it must not do.

The macro-average is the headline number, so the ways it can be quietly wrong matter more than
usual. Three of them are pinned here: averaging must happen inside a (repeat, fold) cell so the
pairing survives into the corrected t-test; a cell with an undefined task must be dropped rather
than averaged over whatever remains; and the per-dataset weighting must actually undo CGMacros'
two sensor streams rather than merely renaming the column.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from evaluate import macro_cells, paired_macro


@dataclass
class FakeResult:
    repeat: int
    fold: int
    roc_auc: float


class FakeRun:
    def __init__(self, results):
        self.results = results

    def scores(self, metric: str) -> np.ndarray:
        values = np.array([getattr(r, metric) for r in self.results], dtype=float)
        return values[~np.isnan(values)]


def build(scores: dict[str, list[float]], method: str = "m") -> dict:
    """`scores` maps a task key to one value per (repeat, fold), in order."""
    runs = {}
    for key, values in scores.items():
        runs[(key, method)] = FakeRun(
            [FakeResult(repeat=i // 2, fold=i % 2, roc_auc=v) for i, v in enumerate(values)]
        )
    return runs


def test_averages_within_a_cell_not_across_them():
    """Two tasks, four cells. Each output is the mean of the two tasks in that same cell."""
    runs = build({"a:t": [0.6, 0.8, 0.4, 1.0], "b:t": [0.8, 0.6, 0.6, 0.0]})
    out = np.array(list(macro_cells(runs, "m", "roc_auc", by_dataset=False).values()))
    assert np.allclose(out, [0.7, 0.7, 0.5, 0.5])


def test_pairing_is_preserved_so_a_constant_offset_survives():
    """The whole point of averaging per cell: a paired difference stays paired."""
    a = build({"a:t": [0.6, 0.8, 0.4, 1.0], "b:t": [0.8, 0.6, 0.6, 0.0]}, method="a")
    b = build({"a:t": [0.5, 0.7, 0.3, 0.9], "b:t": [0.7, 0.5, 0.5, -0.1]}, method="b")
    left, right, _ = paired_macro({**a, **b}, "a", "b", "roc_auc", by_dataset=False)
    diff = left - right
    assert np.allclose(diff, 0.1), "per-cell pairing lost"


def test_a_cell_with_an_undefined_task_is_dropped_whole():
    """Not averaged over the surviving tasks, which would change what the number means."""
    runs = build({"a:t": [0.6, np.nan, 0.4, 1.0], "b:t": [0.8, 0.6, 0.6, 0.0]})
    out = np.array(list(macro_cells(runs, "m", "roc_auc", by_dataset=False).values()))
    assert len(out) == 3
    assert np.allclose(out, [0.7, 0.5, 0.5])
    assert not np.isnan(out).any()


def test_per_dataset_weighting_collapses_the_two_cgmacros_sensors():
    """Per entry, CGMacros counts twice; per dataset, once.

    Both CGMacros streams score 1.0 and the single other dataset scores 0.0. Weighting by entry
    gives 2/3; weighting by dataset gives 1/2. If the two agree, the weighting is not doing
    anything.
    """
    runs = build({
        "cgmacros:t[dexcom]": [1.0, 1.0],
        "cgmacros:t[libre]": [1.0, 1.0],
        "hall:t[hall]": [0.0, 0.0],
    })
    per_entry = np.array(list(macro_cells(runs, "m", "roc_auc", by_dataset=False).values()))
    per_dataset = np.array(list(macro_cells(runs, "m", "roc_auc", by_dataset=True).values()))
    assert np.allclose(per_entry, 2 / 3)
    assert np.allclose(per_dataset, 0.5)


def test_subject_aggregation_undoes_recording_length_weighting():
    """One subject with many windows must not outvote several with few. Sensitivity, not headline.

    Subject A carries ten windows of one class, subjects B..D one window each of the other. A
    window-weighted score is dominated by A; a subject-weighted score is not.
    """
    from opencgm_stateevent.eval.probe import _by_subject

    subjects = np.array(["a"] * 10 + ["b", "c", "d"])
    labels = np.array([1] * 10 + [0, 0, 0])
    proba = np.zeros((13, 2))
    proba[:10, 1] = 0.9  # confident and correct on the over-represented subject
    proba[10:, 1] = [0.8, 0.7, 0.6]  # wrong on all three of the others

    y, mean_proba, pred = _by_subject(subjects, labels, proba)
    assert list(y) == [1, 0, 0, 0]
    assert len(mean_proba) == 4, "one row per subject, not per window"
    assert np.isclose(mean_proba[0, 1], 0.9)
    assert list(pred) == [1, 1, 1, 1]


def test_pairing_survives_one_method_dropping_a_different_cell():
    """Equal lengths do not prove alignment.

    If each side is filtered independently and one drops a different cell, the arrays line up by
    position while describing different partitions, and the paired test subtracts mismatched
    folds. Here method "a" loses cell 1 and method "b" loses cell 2: both keep three of four, so
    a length check passes and the result is silently wrong.
    """
    a = build({"t:t": [0.5, np.nan, 0.7, 0.8]}, method="a")
    b = build({"t:t": [0.1, 0.2, np.nan, 0.4]}, method="b")

    left, right, shared = paired_macro({**a, **b}, "a", "b", "roc_auc", by_dataset=False)
    assert shared == [(0, 0), (1, 1)], shared
    assert np.allclose(left, [0.5, 0.8])
    assert np.allclose(right, [0.1, 0.4])


def test_task_bootstrap_interval_is_wider_than_the_fold_paired_one():
    """The whole reason the bootstrap exists. Each task draws its own folds, so cell-to-cell
    variation understates uncertainty; resampling tasks puts task heterogeneity back."""
    from evaluate import task_bootstrap_ci

    # Three tasks that disagree sharply about the model: +0.20, +0.02, -0.05.
    model = build({"x:t": [0.90, 0.90], "y:t": [0.62, 0.62], "z:t": [0.45, 0.45]}, method="m")
    base = build({"x:t": [0.70, 0.70], "y:t": [0.60, 0.60], "z:t": [0.50, 0.50]}, method="b")
    runs = {**model, **base}

    low, high = task_bootstrap_ci(runs, "m", "b", "roc_auc", by_dataset=False, draws=500)
    assert low < high
    # With tasks that disagree this much, the interval must admit zero.
    assert low <= 0 <= high, (low, high)
