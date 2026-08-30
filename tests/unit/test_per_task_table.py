"""The per-task table must exist, carry 18 rows, and reconcile with the published macro figures.

The public write-up claims the model is "significantly ahead of the clinical baseline on 16 of
18 task-cohort combinations after Holm correction; CGM-JEPA on 0". This suite regenerates the
per-task table from the fold-level scores and asserts what the data actually supports. Where
the data contradicts the published claim, the test asserts the data, not the claim -- the
discrepancies are recorded in `findings/per_task.md`.

Two are known at the time of writing:

* The tally. Under the per-seed Holm-adjusted Nadeau-Bengio test, zero of 18 tasks are
  significant vs `clinical_metrics` in any seed; under the pooled 5-seed test it is also 0/18
  (and 2/18 vs `raw_masked`). Nothing in the fold data supports 16/18.

* The absolute levels. The published "ROC-AUC 0.670 +/- 0.011" (and clinical 0.652, raw 0.617,
  CGM-JEPA 0.652) does not match the 18-task per-entry macro of this data, which is 0.670
  (clinical 0.643, raw 0.607, CGM-JEPA 0.643). The published levels match, to <=0.001, the mean
  over 16 tasks excluding `shanghai_t2dm:hyperlipidemia` and `stanford:insulin_resistance`.
  The published *deltas* (+0.0269, +0.0628, +0.0000) match the full 18-task data exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE = REPO_ROOT / "reports/eval/per_task_5seed.csv"

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "reports/eval/seed17_ep120_full/fold_scores.csv").exists(),
    reason="5-seed evaluation artifacts not present in this checkout",
)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    assert TABLE.exists(), "run scripts/aggregate_per_task.py first"
    return pd.read_csv(TABLE)


@pytest.fixture(scope="module")
def rebuilt(monkeypatch_module_cwd) -> pd.DataFrame:
    from scripts.aggregate_per_task import build_table

    return build_table()


@pytest.fixture(scope="module")
def monkeypatch_module_cwd(tmp_path_factory):
    """`build_table` reads relative paths; run it from the repo root regardless of cwd."""
    import os

    old = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(old)


def test_exactly_18_task_source_rows(table: pd.DataFrame) -> None:
    """18 task-source combinations; 14 distinct dataset-task probes. Distinct counts, kept so."""
    assert len(table) == 18
    assert table.task.is_unique
    assert table.cohort.value_counts().to_dict() == {
        "cgmacros": 8, "hall": 4, "shanghai_t2dm": 3, "stanford": 3,
    }
    assert len(table[["cohort", "task_name"]].drop_duplicates()) == 14
    assert (table.n_folds == 50).all()


def test_csv_is_not_stale(table: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    """The committed CSV must be what the fold-level scores regenerate to."""
    pd.testing.assert_frame_equal(
        table, rebuilt, check_exact=False, rtol=1e-9, atol=1e-12
    )


def test_per_task_deltas_reconcile_with_published_macro(table: pd.DataFrame) -> None:
    """The mean of the 18 per-task ROC deltas is the published per-entry macro delta."""
    published = {
        "glucofm_vs_clinical_roc_auc_delta": 0.0269,
        "glucofm_vs_raw_roc_auc_delta": 0.0628,
        "cgmjepa_vs_clinical_roc_auc_delta": 0.0000,
    }
    for column, value in published.items():
        assert abs(table[column].mean() - value) < 5e-4, column


def test_absolute_levels_are_internally_consistent(table: pd.DataFrame) -> None:
    """Per-task levels must average to the per-entry macro recomputed from the summaries.

    Note the recomputed macro is 0.670, not the published 0.670; see the module docstring.
    """
    summaries = pd.concat([
        pd.read_csv(REPO_ROOT / f"reports/eval/seed{s}_ep120_full/summary.csv")
        for s in (17, 29, 43, 71, 101)
    ])
    macro = summaries[summaries.method == "opencgm_mean"].groupby("task").roc_auc_mean.mean()
    assert abs(table.glucofm_roc_auc.mean() - macro.mean()) < 1e-9
    assert abs(macro.mean() - 0.6701) < 1e-3  # NOT 0.670; published level is a 16-task subset


def test_significance_tally_is_what_the_data_supports(rebuilt: pd.DataFrame) -> None:
    """The site claims 16/18 significant vs clinical after Holm. The data says 0/18.

    Asserted from the regenerated table (fold scores -> pooled Nadeau-Bengio test -> Holm over
    the 72-test family), not from the committed CSV. Do not change the expected values here to
    match the website; change the website. The per-seed view agrees: no task is Holm-significant
    vs `clinical_metrics` in ANY of the five single-seed runs.
    """
    def tally(model: str, baseline: str) -> int:
        sig = rebuilt[f"{model}_vs_{baseline}_roc_auc_significant"]
        ahead = rebuilt[f"{model}_vs_{baseline}_roc_auc_delta"] > 0
        return int((sig & ahead).sum())

    assert tally("glucofm", "clinical") == 0  # site claims 16 -- unsupported by the data
    assert tally("glucofm", "raw") == 2
    assert tally("cgmjepa", "clinical") == 0  # site claims 0 -- this one holds
    assert (rebuilt.glucofm_vs_clinical_roc_auc_n_seeds_sig_holm == 0).all()
