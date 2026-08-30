"""The evaluation protocol. Blueprint §9.5, §19.4, §19.5.

Two properties carry the weight of every downstream number, and both fail silently:

* **no subject crosses a fold.** A window-level split leaves two days from one person on opposite
  sides and the probe scores the person, not the representation. Accuracy goes *up*, so nothing
  looks wrong.
* **folds are identical across compared methods.** If they are not, the paired tests in §19.5 are
  invalid and every confidence interval is wrong — again with no visible symptom.

Each is tested with a control that fails when the protection is removed, because a leakage test
that would pass on a broken implementation is worse than none.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencgm_stateevent.eval.probe import ProbeConfig, run_probe
from opencgm_stateevent.eval.splits import (
    build_folds,
    fold_manifest,
    repeat_seed,
    stratified_subject_folds,
    window_mask,
)
from opencgm_stateevent.eval.stats import (
    compare,
    corrected_variance,
    holm_adjust,
    macro_average_per_fold,
)


def cohort(n_subjects: int = 40, windows_each: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    subjects = np.array([f"s{i:03d}" for i in range(n_subjects)])
    labels = (np.arange(n_subjects) % 3 == 0).astype(int)
    window_subjects = np.repeat(subjects, windows_each)
    window_labels = np.repeat(labels, windows_each)
    features = rng.normal(size=(len(window_subjects), 16))
    return subjects, labels, window_subjects, window_labels, features


# --- fold construction ----------------------------------------------------------------------


def test_no_subject_appears_in_both_sides_of_any_fold():
    subjects, labels, *_ = cohort()
    for fold in build_folds("t", subjects, labels):
        assert not (set(fold.train_subjects) & set(fold.test_subjects))


def test_every_subject_is_tested_exactly_once_per_repeat():
    subjects, labels, *_ = cohort()
    folds = build_folds("t", subjects, labels)
    for repeat in range(10):
        tested = [s for f in folds if f.repeat == repeat for s in f.test_subjects]
        assert sorted(tested) == sorted(subjects)


def test_fold_count_matches_the_paper():
    subjects, labels, *_ = cohort()
    folds = build_folds("t", subjects, labels)
    assert len(folds) == 5 * 10


def test_folds_depend_only_on_task_and_repeat():
    """The property that makes paired comparison valid: folds cannot see the representation."""
    subjects, labels, *_ = cohort()
    a = fold_manifest(build_folds("task_a", subjects, labels))
    b = fold_manifest(build_folds("task_a", subjects, labels))
    assert a["hash"] == b["hash"]


def test_different_tasks_get_different_folds():
    subjects, labels, *_ = cohort()
    a = fold_manifest(build_folds("task_a", subjects, labels))
    b = fold_manifest(build_folds("task_b", subjects, labels))
    assert a["hash"] != b["hash"]


def test_repeat_seed_is_not_drawn_from_a_shared_stream():
    """Adding or reordering a task must not perturb any other task's folds."""
    assert repeat_seed("a", 0) != repeat_seed("b", 0)
    assert repeat_seed("a", 0) != repeat_seed("a", 1)
    assert repeat_seed("a", 3) == repeat_seed("a", 3)


def test_a_tiny_minority_class_still_reaches_every_fold():
    """`hall:hyperlipidemia` has 8 positives in 57 subjects. No fold may end up with zero."""
    subjects = np.array([f"s{i:03d}" for i in range(57)])
    labels = np.zeros(57, dtype=int)
    labels[:8] = 1
    for repeat in range(10):
        groups = stratified_subject_folds(subjects, labels, seed=repeat_seed("t", repeat))
        counts = [int(labels[g].sum()) for g in groups]
        assert min(counts) >= 1, f"repeat {repeat} produced a fold with no positives: {counts}"


def test_windows_are_split_by_subject_not_by_window():
    subjects, labels, window_subjects, _, _ = cohort()
    fold = build_folds("t", subjects, labels)[0]
    train = window_mask(window_subjects, fold.train_subjects)
    test = window_mask(window_subjects, fold.test_subjects)
    assert not set(window_subjects[train]) & set(window_subjects[test])
    assert (train | test).all()


# --- leakage, with controls -----------------------------------------------------------------


def test_subject_identity_features_do_not_generalise_under_grouped_splits():
    """The control that gives the leakage tests teeth.

    Features encode subject identity and nothing about the label. Under a subject-grouped split
    the probe cannot use them, so it scores near chance. Under a window-level split the same
    features score near perfect. If grouping ever broke, this gap is what would move.
    """
    subjects, labels, window_subjects, window_labels, _ = cohort()
    identity = np.eye(len(subjects))[
        np.searchsorted(subjects, window_subjects)
    ]
    folds = build_folds("t", subjects, labels)[:5]

    grouped = run_probe(
        folds, identity, window_subjects, window_labels, task="t", method="identity"
    )
    assert grouped.scores("roc_auc").mean() < 0.75

    # Same features, same classifier, but folds assigned per window rather than per subject.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(window_subjects))
    cut = len(order) // 5
    test_idx, train_idx = order[:cut], order[cut:]
    from sklearn.metrics import roc_auc_score

    from opencgm_stateevent.eval.probe import make_pipeline

    pipeline = make_pipeline(ProbeConfig())
    pipeline.fit(identity[train_idx], window_labels[train_idx])
    leaked = roc_auc_score(
        window_labels[test_idx], pipeline.predict_proba(identity[test_idx])[:, 1]
    )
    assert leaked > 0.95, "control failed: window-level split should leak"


def test_a_predictive_feature_is_still_learnable_under_grouping():
    """Complement to the leakage control: grouping must not simply destroy all signal."""
    subjects, labels, window_subjects, window_labels, _ = cohort()
    rng = np.random.default_rng(1)
    signal = (window_labels * 2.0 + rng.normal(scale=0.5, size=len(window_labels)))[:, None]
    folds = build_folds("t", subjects, labels)[:5]
    run = run_probe(folds, signal, window_subjects, window_labels, task="t", method="signal")
    assert run.scores("roc_auc").mean() > 0.9


def test_the_scaler_is_fit_on_training_windows_only():
    """Fitting the scaler on everything is leakage that never crosses a subject boundary."""
    subjects, labels, window_subjects, window_labels, features = cohort()
    fold = build_folds("t", subjects, labels)[0]
    train = window_mask(window_subjects, fold.train_subjects)

    from opencgm_stateevent.eval.probe import make_pipeline

    pipeline = make_pipeline(ProbeConfig())
    pipeline.fit(features[train], window_labels[train])
    fitted = pipeline.named_steps["scale"].mean_
    assert np.allclose(fitted, features[train].mean(axis=0))
    assert not np.allclose(fitted, features.mean(axis=0))


# --- probe behaviour ------------------------------------------------------------------------


def test_a_single_class_test_fold_yields_nan_not_a_guess():
    subjects = np.array([f"s{i:02d}" for i in range(10)])
    labels = np.zeros(10, dtype=int)
    labels[0] = 1
    window_subjects = np.repeat(subjects, 3)
    window_labels = np.repeat(labels, 3)
    features = np.random.default_rng(0).normal(size=(30, 4))
    folds = build_folds("t", subjects, labels, n_repeats=1)
    run = run_probe(folds, features, window_subjects, window_labels, task="t", method="m")
    undefined = [r for r in run.results if r.note]
    assert undefined, "expected at least one undefined fold"
    assert all(np.isnan(r.roc_auc) for r in undefined)


def test_summary_reports_how_many_folds_were_dropped():
    subjects = np.array([f"s{i:02d}" for i in range(10)])
    labels = np.zeros(10, dtype=int)
    labels[0] = 1
    ws, wl = np.repeat(subjects, 3), np.repeat(labels, 3)
    features = np.random.default_rng(0).normal(size=(30, 4))
    run = run_probe(
        build_folds("t", subjects, labels, n_repeats=1), features, ws, wl, task="t", method="m"
    )
    assert run.summary().get("dropped_folds", 0) > 0


def test_headline_config_matches_the_blueprint():
    cfg = ProbeConfig()
    classifier = __import__(
        "opencgm_stateevent.eval.probe", fromlist=["make_pipeline"]
    ).make_pipeline(cfg).named_steps["classifier"]
    assert cfg.C == 1.0
    assert classifier.l1_ratio in (None, 0)  # L2, §19.4
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 1000
    assert classifier.class_weight is None


# --- statistics -----------------------------------------------------------------------------


def test_nadeau_bengio_variance_exceeds_the_naive_variance():
    """The whole point of the correction: intervals must widen, never narrow."""
    d = np.random.default_rng(0).normal(0.02, 0.05, size=50)
    assert corrected_variance(d) > d.var(ddof=1) / len(d)


def test_corrected_interval_is_wider_than_an_uncorrected_one():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.05, size=50)
    b = a - rng.normal(0.02, 0.01, size=50)
    c = compare(a, b, task="t", metric="roc_auc", method="m", baseline="b")
    naive_half = 1.96 * np.sqrt((a - b).var(ddof=1) / 50)
    assert (c.ci_high - c.ci_low) / 2 > naive_half


def test_identical_methods_are_not_significant():
    a = np.random.default_rng(0).normal(0.7, 0.05, size=50)
    c = compare(a, a.copy(), task="t", metric="roc_auc", method="m", baseline="b")
    assert c.p_value == pytest.approx(1.0)
    assert not c.significant


def test_a_large_consistent_gain_is_significant():
    rng = np.random.default_rng(0)
    b = rng.normal(0.60, 0.02, size=50)
    a = b + 0.20
    c = compare(a, b, task="t", metric="roc_auc", method="m", baseline="b")
    assert c.mean_difference == pytest.approx(0.20, abs=1e-9)
    assert c.significant


def test_holm_only_ever_increases_p_values():
    rng = np.random.default_rng(0)
    comparisons = [
        compare(
            rng.normal(0.7, 0.05, 30), rng.normal(0.68, 0.05, 30),
            task=f"t{i}", metric="roc_auc", method="m", baseline="b",
        )
        for i in range(8)
    ]
    for before, after in zip(comparisons, holm_adjust(comparisons), strict=True):
        assert after.p_holm >= before.p_value


def test_holm_adjusted_values_are_monotone_in_rank():
    """Holm's adjusted values never decrease as the raw p-value increases."""
    rng = np.random.default_rng(1)
    comparisons = [
        compare(
            rng.normal(0.7, 0.05, 30), rng.normal(0.60, 0.05, 30),
            task=f"t{i}", metric="roc_auc", method="m", baseline="b",
        )
        for i in range(6)
    ]
    adjusted = holm_adjust(comparisons)
    by_raw = sorted(zip([c.p_value for c in comparisons],
                        [c.p_holm for c in adjusted], strict=True))
    holm_values = [h for _, h in by_raw]
    assert holm_values == sorted(holm_values)


def test_macro_average_preserves_the_fold_axis():
    per_task = {"a": np.array([0.5, 0.7, 0.9]), "b": np.array([0.1, 0.3, 0.5])}
    assert np.allclose(macro_average_per_fold(per_task), [0.3, 0.5, 0.7])


def test_macro_average_refuses_misaligned_folds():
    with pytest.raises(ValueError, match="differing fold counts"):
        macro_average_per_fold({"a": np.zeros(5), "b": np.zeros(4)})


def test_compare_refuses_unpaired_scores():
    with pytest.raises(ValueError, match="fold-aligned"):
        compare(np.zeros(5), np.zeros(4), task="t", metric="m", method="a", baseline="b")
