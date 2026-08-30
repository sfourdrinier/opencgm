"""Statistical comparison. Blueprint §19.5.

Repeated cross-validation reuses the same data across folds and repeats, so fold scores are not
independent. Treating them as independent — the ordinary paired t-test — understates the variance
and produces confidence intervals far narrower than the evidence supports. Nadeau and Bengio's
correction inflates the variance by `1/k + n_test/n_train`, and §19.5 fixes the ratio at 1/4.

Holm adjustment across the planned comparator x metric family follows, and both raw fold
distributions and a nonparametric bootstrap are reported alongside, because §19.5 explicitly asks
not to rely on p-values alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

TEST_TRAIN_RATIO = 0.25  # PAPER_EXACT §19.5, n_test/n_train = 1/4
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class Comparison:
    """A paired difference between two methods on one task and metric."""

    task: str
    metric: str
    method: str
    baseline: str
    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    t_statistic: float
    p_value: float
    p_holm: float = float("nan")
    bootstrap_ci_low: float = float("nan")
    bootstrap_ci_high: float = float("nan")

    @property
    def significant(self) -> bool:
        """Holm-adjusted, two-sided, at 0.05. Uses the adjusted value once it is assigned."""
        p = self.p_holm if np.isfinite(self.p_holm) else self.p_value
        return bool(p < 0.05)

    def describe(self) -> str:
        direction = "better" if self.mean_difference > 0 else "worse"
        verdict = "" if self.significant else " (not significant)"
        return (
            f"{self.method} vs {self.baseline} on {self.task} {self.metric}: "
            f"{self.mean_difference:+.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}] "
            f"{direction}, p={self.p_holm if np.isfinite(self.p_holm) else self.p_value:.4f}"
            f"{verdict}"
        )


def corrected_variance(differences: np.ndarray, ratio: float = TEST_TRAIN_RATIO) -> float:
    """Nadeau-Bengio corrected variance of the mean paired difference. §19.5.

        var_corrected = var(d) * (1/n + n_test/n_train)

    The uncorrected term is `var(d)/n`. The extra `n_test/n_train` accounts for the overlap
    between training sets across folds, which no amount of repetition removes.
    """
    n = len(differences)
    return float(differences.var(ddof=1) * (1.0 / n + ratio))


def bootstrap_ci(
    differences: np.ndarray, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 17
) -> tuple[float, float]:
    """Percentile bootstrap on the paired differences. §19.5 sensitivity.

    Makes no normality assumption, but shares the correlated-folds problem, so it is reported
    beside the corrected interval rather than in place of it.
    """
    if len(differences) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(
    method_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    task: str,
    metric: str,
    method: str,
    baseline: str,
    ratio: float = TEST_TRAIN_RATIO,
) -> Comparison:
    """Two-sided Nadeau-Bengio corrected paired t-test with a 95% interval. §19.5.

    Scores must be aligned fold-for-fold: element i of each array is the same repeat and fold of
    the same partition. That is guaranteed by `splits.build_folds` depending only on
    `(task, repeat)`, never on the representation being scored.
    """
    if len(method_scores) != len(baseline_scores):
        raise ValueError("paired comparison needs fold-aligned scores")
    differences = np.asarray(method_scores, dtype=float) - np.asarray(baseline_scores, dtype=float)
    n = len(differences)
    if n < 2:
        raise ValueError("need at least two folds")

    mean = float(differences.mean())
    se = float(np.sqrt(corrected_variance(differences, ratio)))
    df = n - 1
    if se == 0.0:
        t_stat, p = (0.0, 1.0) if mean == 0.0 else (np.inf * np.sign(mean), 0.0)
        half = 0.0
    else:
        t_stat = mean / se
        p = float(2.0 * stats.t.sf(abs(t_stat), df))
        half = float(stats.t.ppf(0.975, df) * se)

    low, high = bootstrap_ci(differences)
    return Comparison(
        task=task, metric=metric, method=method, baseline=baseline, n=n,
        mean_difference=mean, ci_low=mean - half, ci_high=mean + half,
        t_statistic=float(t_stat), p_value=float(p),
        bootstrap_ci_low=low, bootstrap_ci_high=high,
    )


def holm_adjust(comparisons: list[Comparison]) -> list[Comparison]:
    """Holm-Bonferroni across the whole planned family. §19.5.

    Applied once over every comparator x metric x task test in the planned set. Adjusting within
    each task separately would understate the family size and inflate the apparent number of
    significant results.
    """
    if not comparisons:
        return []
    order = np.argsort([c.p_value for c in comparisons])
    m = len(comparisons)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * comparisons[idx].p_value
        running = max(running, value)  # Holm's adjusted values are monotone non-decreasing
        adjusted[idx] = min(1.0, running)
    return [
        Comparison(**{**c.__dict__, "p_holm": adjusted[i]}) for i, c in enumerate(comparisons)
    ]


def macro_average_per_fold(runs_by_task: dict[str, np.ndarray]) -> np.ndarray:
    """Macro-average across tasks within each repeat/fold. §19.5.

    Averaging per fold and then comparing preserves the pairing. Averaging each task's mean first
    and then comparing those would throw away the fold structure the corrected test depends on.
    """
    lengths = {len(v) for v in runs_by_task.values()}
    if len(lengths) != 1:
        raise ValueError(f"tasks have differing fold counts: {lengths}")
    return np.mean(np.stack(list(runs_by_task.values())), axis=0)
