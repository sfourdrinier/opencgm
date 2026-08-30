"""The numbers in the release documents must match the code and the corpus on disk.

This suite exists because they once did not. `521,584` — the CGM-JEPA comparator's parameter
count — was copied into five release-facing documents as though it were ours, including the
abstract of `paper.md` and the Hugging Face model card. `3,494` was quoted as the pretraining
corpus size in six places, one of which cited as its evidence the very `.npy` file whose shape
is `(353127, 288)`.

Prose drifts; a test does not. Every figure a reader can check is asserted here against the
thing that produces it: the model constructor, the window cache, and the source registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from opencgm_stateevent.baselines import cgm_jepa
from opencgm_stateevent.model.model import OpenCGMStateEvent
from opencgm_stateevent.sources import Lane, scan

REPO_ROOT = Path(__file__).resolve().parents[2]

# Documents a reader of the public release will check the numbers in.
RELEASE_DOCS = [
    "paper.md",
    "README.md",
    "STATE.md",
    "REPRODUCE.md",
    "app.py",
    "model_cards/glucofm_encoder.md",
    "findings/results_section.md",
    "findings/head_to_head.md",
]


def _docs() -> dict[str, str]:
    out = {}
    for rel in RELEASE_DOCS:
        path = REPO_ROOT / rel
        if path.exists():
            out[rel] = path.read_text()
    return out


# --------------------------------------------------------------------------------------
# Ground truth, computed rather than remembered
# --------------------------------------------------------------------------------------


def _model() -> OpenCGMStateEvent:
    return OpenCGMStateEvent(raw_statistics=True)


def test_trainable_parameter_count_is_732_593() -> None:
    """The figure compared against the paper's 0.72 M, and the +1.7 % claim that rides on it."""
    model = _model()
    total = sum(p.numel() for p in model.trainable_parameters())
    assert total == 732_593

    deviation = (total - 720_000) / 720_000
    assert 0.016 < deviation < 0.018, (
        f"the '+1.7 % vs 0.72 M' claim no longer holds: {deviation:.4%}"
    )


def test_released_encoder_parameter_count_is_435_633() -> None:
    """What ships as `glucofm_encoder.onnx` is the online encoder alone, not the whole model."""
    model = _model()
    encoder = sum(p.numel() for p in model.online.parameters() if p.requires_grad)
    assert encoder == 435_633


def test_parameter_decomposition_sums_to_the_total() -> None:
    """Encoder + predictor + two transition heads. The EMA target is frozen and adds nothing."""
    model = _model()

    def trainable(module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    encoder = trainable(model.online)
    predictor = trainable(model.predictor)
    transitions = trainable(model.transition_state) + trainable(model.transition_event)

    assert trainable(model.target) == 0, "the EMA target must never be trainable"
    assert encoder + predictor + transitions == 732_593
    assert (encoder, predictor, transitions) == (435_633, 132_480, 164_480)


def test_cgm_jepa_comparator_count_is_not_ours() -> None:
    """521,584 belongs to the baseline. Pinning it here is what makes the next test meaningful."""
    assert cgm_jepa.ENCODER_PARAMETERS == 521_584
    assert cgm_jepa.ENCODER_PARAMETERS != 732_593


def test_lane_a_hours_reconcile_to_the_paper_target() -> None:
    """33,736 h of 109,066 is the 30.9 % the whole reproduction is calibrated against."""
    lane_a = [s for s in scan() if s.lane is Lane.A]
    hours = sum(s.expected_hours or 0 for s in lane_a)
    assert hours == 33_736

    fraction = hours / 109_066
    assert 0.308 < fraction < 0.310, f"the 30.9 % claim no longer holds: {fraction:.3%}"


def test_pretraining_corpus_is_353_127_windows() -> None:
    """The claim `3,494 windows` cited this exact file as its evidence. The file disagreed."""
    values = REPO_ROOT / "data" / "canonical" / "windows" / "strict_seed17.values.npy"
    mask = REPO_ROOT / "data" / "canonical" / "windows" / "strict_seed17.mask.npy"
    if not values.exists():
        pytest.skip("window cache not built; run `just build-windows`")

    v = np.load(values, mmap_mode="r")
    m = np.load(mask, mmap_mode="r")
    assert v.shape == (353_127, 288)
    assert m.shape == v.shape


# --------------------------------------------------------------------------------------
# The documents must agree with the above
# --------------------------------------------------------------------------------------


def test_no_release_doc_claims_the_comparators_parameter_count_as_ours() -> None:
    """`521,584` may appear only where it is explicitly the CGM-JEPA baseline being described."""
    attributes_to_comparator = re.compile(
        r"CGM-JEPA|comparator|baseline|cruiseresearchgroup", re.I
    )
    offenders = []
    for rel, text in _docs().items():
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            if "521,584" not in line:
                continue
            # Legitimate when the attribution is local: either on the line itself, or in the
            # few lines above it -- a comparison table carries it in the column header.
            context = lines[max(0, i - 5) : i]
            if attributes_to_comparator.search(line) or any(
                attributes_to_comparator.search(c) for c in context
            ):
                continue
            offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "the comparator's parameter count is presented as ours:\n" + "\n".join(offenders)
    )


def test_no_release_doc_quotes_the_retracted_corpus_size() -> None:
    """`3,494` was never the window count. It must not come back."""
    offenders = [
        f"{rel}:{i}: {line.strip()}"
        for rel, text in _docs().items()
        for i, line in enumerate(text.splitlines(), start=1)
        if re.search(r"\b3,?494\b", line)
    ]
    assert not offenders, "the retracted corpus size is back:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    ("doc", "needle"),
    [
        ("paper.md", "732,593"),
        ("paper.md", "353,127"),
        ("model_cards/glucofm_encoder.md", "435,633"),
        ("findings/results_section.md", "732,593"),
        ("findings/results_section.md", "353,127"),
        ("findings/head_to_head.md", "732,593"),
        ("REPRODUCE.md", "732,593"),
    ],
)
def test_release_doc_states_the_correct_figure(doc: str, needle: str) -> None:
    docs = _docs()
    if doc not in docs:
        pytest.skip(f"{doc} not present")
    assert needle in docs[doc], f"{doc} no longer states {needle}"


def test_tasks_and_task_source_combinations_are_not_conflated() -> None:
    """14 dataset-task probes; 18 task-source combinations. They are different counts.

    Two cohorts are measured on two sensors each, which is where the extra four come from.
    The model card once said `macro across 14 task-source combinations`, which is neither.
    """
    offenders = [
        f"{rel}:{i}: {line.strip()}"
        for rel, text in _docs().items()
        for i, line in enumerate(text.splitlines(), start=1)
        if re.search(r"14 task[- ]source", line)
    ]
    assert not offenders, "14 counts probes, not task-source combinations:\n" + "\n".join(offenders)


def test_inferred_fork_count_matches_the_config_checker() -> None:
    """The website says "nineteen". That number must come from the config, not from memory.

    `opencgm config-check` refuses to load the reference config unless every ambiguous option
    carries an evidence tag, and reports how many are tagged INFERRED_RECONSTRUCTION. The
    website and the reference config must agree.
    """
    import subprocess

    facts = (REPO_ROOT / "web" / "lib" / "facts.ts").read_text()
    match = re.search(r"inferredForks:\s*(\d+)", facts)
    assert match, "web/lib/facts.ts no longer declares inferredForks"
    claimed = int(match.group(1))

    out = subprocess.run(
        ["uv", "run", "opencgm", "config-check", "bundle/glucofm_reference_config.yaml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    reported = re.search(r"(\d+)\s+INFERRED_RECONSTRUCTION", out.stdout + out.stderr)
    if not reported:
        pytest.skip("config-check produced no fork count")
    assert claimed == int(reported.group(1))


def test_decision_entry_count_matches_decisions_md() -> None:
    """DECISIONS.md is the register; the site must not quote a stale size for it."""
    facts = (REPO_ROOT / "web" / "lib" / "facts.ts").read_text()
    match = re.search(r"decisionEntries:\s*(\d+)", facts)
    assert match, "web/lib/facts.ts no longer declares decisionEntries"

    entries = len(re.findall(r"^## D\d{3}", (REPO_ROOT / "DECISIONS.md").read_text(), re.M))
    assert int(match.group(1)) == entries, (
        f"facts.ts says {match.group(1)} decisions; DECISIONS.md has {entries}"
    )


def test_headline_levels_are_the_mean_over_all_eighteen_task_rows() -> None:
    """The published ROC and PR levels must be the macro over every task, not a subset.

    They were not. 0.679 / 0.652 / 0.617 appeared in the paper, the site, the model card and
    the findings for months; averaging all 18 task-source rows gives 0.670 / 0.643 / 0.607.
    The published figures reproduce exactly if two tasks are dropped -- an early snapshot
    taken before `shanghai_t2dm:hyperlipidemia` and `stanford:insulin_resistance` existed.

    The error survived because every *delta* was computed over all 18 rows and reconciled
    perfectly, so each difference checked out while the levels they were differences of did
    not. This test compares the levels themselves against the evaluation CSVs.
    """
    import csv
    import statistics

    seeds = (17, 29, 43, 71, 101)
    dirs = [REPO_ROOT / "reports" / "eval" / f"seed{s}_ep120_full" / "summary.csv" for s in seeds]
    if not all(d.exists() for d in dirs):
        pytest.skip("headline evaluation runs not present")

    def macro(method: str, column: str) -> float:
        per_seed = []
        for path in dirs:
            with path.open() as fh:
                rows = [
                    float(r[column])
                    for r in csv.DictReader(fh)
                    if r["config"] == "headline" and r["method"] == method
                ]
            assert len(rows) == 18, (
                f"{path.parent.name}/{method}: {len(rows)} task rows, expected 18"
            )
            per_seed.append(statistics.mean(rows))
        return statistics.mean(per_seed)

    facts = (REPO_ROOT / "web" / "lib" / "facts.ts").read_text()

    for method, key in (
        ("opencgm_mean", "model"),
        ("clinical_metrics", "clinical"),
        ("raw_masked", "rawMasked"),
    ):
        measured = macro(method, "roc_auc_mean")
        claimed = re.search(rf"rocAuc:.*?{key}:\s*([0-9.]+)", facts, re.S)
        assert claimed, f"facts.ts no longer declares rocAuc.{key}"
        assert abs(float(claimed.group(1)) - measured) < 0.0015, (
            f"rocAuc.{key} says {claimed.group(1)}; the 18-task macro is {measured:.4f}"
        )


def test_no_document_claims_per_task_significance_that_the_data_does_not_support() -> None:
    """The site and paper claimed 16 of 18 tasks significant after Holm. It is zero.

    No task-source row separates from `clinical_metrics` at Holm-adjusted p < 0.05 in any of
    the five seeds. The macro advantage is real; the per-task claim was not.
    """
    # A line that names the old number in order to retract it is the point, not a relapse.
    retracting = re.compile(
        r"false|corrected|retract|disprove|no longer|earlier|previously|was claimed|not the",
        re.I,
    )
    offenders = [
        f"{rel}:{i}: {line.strip()}"
        for rel, text in _docs().items()
        for i, line in enumerate(text.splitlines(), start=1)
        if re.search(r"16\s*(?:of|/)\s*18", line) and not retracting.search(line)
    ]
    assert not offenders, "the retracted per-task claim is asserted again:\n" + "\n".join(offenders)
