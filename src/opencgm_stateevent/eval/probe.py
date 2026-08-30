"""Frozen-encoder linear probing. Blueprint §19.4 — `PAPER_EXACT` protocol.

The reference pipeline is fixed by §19.4: `StandardScaler` then `LogisticRegression` with L2,
`lbfgs`, `max_iter=1000`, `class_weight=None`, and `C=1.0` as the predeclared headline value. `C`,
class weighting, scaling details and threshold behaviour are unpublished, so §19.4 asks for
sensitivity runs alongside — not instead of — the headline.

Scaler and classifier are fit on training subjects only. Fitting the scaler on the full matrix is
the quiet version of leakage: it never crosses a subject boundary in any visible way, and it
still lets test-set statistics into training.

Windows are the unit of fitting; subjects are the unit of splitting. A subject contributing forty
windows contributes forty training rows, which is what §19.4 describes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .splits import Fold, window_mask

PROBE_SEED = 17  # §19.4 "fixed seed"
MAX_ITER = 1000  # PAPER_EXACT §19.4
DEFAULT_C = 1.0  # predeclared headline, §19.4


@dataclass(frozen=True)
class ProbeConfig:
    """The headline configuration is the default; every field is a §19.4 sensitivity axis."""

    C: float = DEFAULT_C
    scale: bool = True
    class_weight: str | None = None
    seed: int = PROBE_SEED

    @property
    def label(self) -> str:
        if self == ProbeConfig():
            return "headline"
        bits = [f"C={self.C}"]
        if not self.scale:
            bits.append("noscale")
        if self.class_weight:
            bits.append(str(self.class_weight))
        return ",".join(bits)


#: The predeclared §19.4 headline configuration, as a singleton default.
HEADLINE = ProbeConfig()


def make_pipeline(cfg: ProbeConfig) -> Pipeline:
    """§19.4 reference pipeline."""
    # scikit-learn 1.8 deprecates the explicit `penalty` argument; L2 is the default and is
    # what §19.4 specifies, so the semantics are unchanged.
    classifier = LogisticRegression(
        C=cfg.C,
        solver="lbfgs",
        max_iter=MAX_ITER,
        class_weight=cfg.class_weight,
        random_state=cfg.seed,
    )
    steps = [("scale", StandardScaler())] if cfg.scale else []
    return Pipeline([*steps, ("classifier", classifier)])


@dataclass
class FoldResult:
    task: str
    method: str
    repeat: int
    fold: int
    n_train: int
    n_test: int
    n_test_subjects: int
    pr_auc: float = float("nan")
    roc_auc: float = float("nan")
    macro_f1: float = float("nan")
    #: Subject-aggregated sensitivity: the same fold scored once per test subject rather than once
    #: per window, by averaging that subject's predicted probabilities. The headline metrics are
    #: window-weighted, so a subject with fourteen days of recording contributes fourteen times the
    #: weight of a subject with one. Subject-grouped splitting prevents identity leakage but not
    #: that pseudo-replication, and the two can disagree. Both are reported.
    pr_auc_subject: float = float("nan")
    roc_auc_subject: float = float("nan")
    macro_f1_subject: float = float("nan")
    #: set when a metric is undefined rather than merely poor
    note: str = ""


def _scores(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    """PR-AUC, ROC-AUC and Macro-F1. §19.4.

    A fold whose test side holds one class leaves both AUCs undefined. That is reported as NaN
    with a note and excluded from aggregation, never silently replaced by 0.5 — an imputed value
    would drag a small-minority task toward a number nobody measured.
    """
    if len(np.unique(y_true)) < 2:
        return {"note": "single-class test fold; AUCs undefined"}
    if n_classes > 2:
        return {
            "pr_auc": float(average_precision_score(
                np.eye(n_classes)[y_true], proba, average="macro"
            )),
            "roc_auc": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
        }
    return {
        "pr_auc": float(average_precision_score(y_true, proba[:, 1])),
        "roc_auc": float(roc_auc_score(y_true, proba[:, 1])),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
    }


def _by_subject(
    subjects: np.ndarray, labels: np.ndarray, proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse window-level predictions to one prediction per subject.

    A subject's probability is the mean over their windows, and their label is the label they
    already carry -- every window of a subject shares it, since the labels are subject-level
    phenotypes. Returns `(y_true, proba, pred)` at subject granularity.
    """
    unique = np.unique(subjects)
    mean_proba = np.stack([proba[subjects == s].mean(axis=0) for s in unique])
    y_true = np.array([labels[subjects == s][0] for s in unique])
    return y_true, mean_proba, mean_proba.argmax(axis=1)


def run_fold(
    fold: Fold,
    features: np.ndarray,
    window_subjects: np.ndarray,
    window_labels: np.ndarray,
    *,
    method: str,
    n_classes: int,
    cfg: ProbeConfig = HEADLINE,
) -> FoldResult:
    """Fit on the fold's training subjects, score its test subjects."""
    import warnings

    train = window_mask(window_subjects, fold.train_subjects)
    test = window_mask(window_subjects, fold.test_subjects)
    result = FoldResult(
        task=fold.task, method=method, repeat=fold.repeat, fold=fold.fold,
        n_train=int(train.sum()), n_test=int(test.sum()),
        n_test_subjects=len(fold.test_subjects),
    )
    if result.n_train == 0 or result.n_test == 0:
        result.note = "empty split"
        return result

    y_train = window_labels[train]
    if len(np.unique(y_train)) < 2:
        result.note = "single-class training fold"
        return result

    pipeline = make_pipeline(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(features[train], y_train)
    proba = pipeline.predict_proba(features[test])
    pred = pipeline.predict(features[test])

    for key, value in _scores(window_labels[test], proba, pred, n_classes).items():
        setattr(result, key, value)

    subject_scores = _scores(*_by_subject(
        window_subjects[test], window_labels[test], proba
    ), n_classes)
    for key, value in subject_scores.items():
        if key == "note":  # a single-class subject-level split must not mask the window-level note
            continue
        setattr(result, f"{key}_subject", value)
    return result


@dataclass
class ProbeRun:
    """Every fold-level score for one method on one task. §19.5 requires these unrounded."""

    task: str
    method: str
    config: ProbeConfig
    results: list[FoldResult] = field(default_factory=list)

    def scores(self, metric: str) -> np.ndarray:
        """Fold-level scores with undefined folds dropped."""
        values = np.array([getattr(r, metric) for r in self.results], dtype=float)
        return values[~np.isnan(values)]

    def summary(self) -> dict:
        out = {"task": self.task, "method": self.method, "config": self.config.label,
               "n_folds": len(self.results)}
        for metric in ("pr_auc", "roc_auc", "macro_f1",
                       "pr_auc_subject", "roc_auc_subject", "macro_f1_subject"):
            values = self.scores(metric)
            out[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            out[f"{metric}_n"] = len(values)
        dropped = sum(1 for r in self.results if r.note)
        if dropped:
            out["dropped_folds"] = dropped
            out["drop_reasons"] = sorted({r.note for r in self.results if r.note})
        return out

    def to_records(self) -> list[dict]:
        return [asdict(r) | {"config": self.config.label} for r in self.results]


def run_probe(
    folds: list[Fold],
    features: np.ndarray,
    window_subjects: np.ndarray,
    window_labels: np.ndarray,
    *,
    task: str,
    method: str,
    n_classes: int = 2,
    cfg: ProbeConfig = HEADLINE,
) -> ProbeRun:
    run = ProbeRun(task=task, method=method, config=cfg)
    for fold in folds:
        run.results.append(
            run_fold(
                fold, features, window_subjects, window_labels,
                method=method, n_classes=n_classes, cfg=cfg,
            )
        )
    return run
