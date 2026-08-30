"""Rights gates.

These are the tests that stop a licence mistake from reaching a public checkpoint. They must
fail loudly if anyone reclassifies a lane without thinking. Blueprint §7, AMENDMENTS A4.
"""

from __future__ import annotations

import pytest

from opencgm_stateevent.sources import FORBIDS_DISTRIBUTION, Lane, load_registry, scan

pytestmark = pytest.mark.leakage

EVAL_ONLY = {"cgmacros", "uchtt1dm", "glucofm_bench"}


def test_lane_e_sources_are_never_distributable():
    """Lane E is NC/ND/SA. A checkpoint containing them cannot be released."""
    for s in scan():
        if s.dataset_id in EVAL_ONLY:
            assert not s.distributable, f"{s.dataset_id} must not be distributable"
            assert s.lane is Lane.E


def test_strict_lane_carries_only_the_four_paper_cohorts():
    """Lane A defines the headline claim; extra members would silently inflate it."""
    lane_a = {s.dataset_id for s in scan() if s.lane is Lane.A}
    assert lane_a == {"big_ideas", "shanghai_t2dm", "stanford", "colas"}


def test_strict_lane_totals_match_the_paper_public_subset():
    """Blueprint §1.1: 285 dataset-defined records and 33,736 CGM hours."""
    lane_a = [s for s in scan() if s.lane is Lane.A]
    assert sum(s.expected_records or 0 for s in lane_a) == 285
    assert sum(s.expected_hours or 0 for s in lane_a) == 33736


def test_wear_cgm_is_recorded_as_unavailable():
    """The public claim depends on this being explicit rather than forgotten."""
    excluded = {e["dataset_id"]: e for e in load_registry().get("excluded", [])}
    assert "wear_cgm" in excluded
    assert excluded["wear_cgm"]["expected_hours"] == 75330


def test_no_source_silently_lacks_a_rights_decision():
    """`unresolved` is allowed, but only where it is stated. Silence is not."""
    for entry in load_registry()["sources"]:
        assert entry.get("license"), f"{entry['dataset_id']} has no license field"
        assert entry.get("weight_release"), f"{entry['dataset_id']} has no weight_release field"


def test_forbidden_markers_are_recognised():
    """Guards against a typo in a weight_release value silently permitting release."""
    for entry in load_registry()["sources"]:
        wr = entry["weight_release"]
        if wr.startswith("forbidden"):
            assert wr in FORBIDS_DISTRIBUTION, f"unrecognised forbidden marker: {wr}"


def test_published_heads_bundle_declares_the_licence_its_sources_impose():
    """The website and the API distribute `web/public/models/heads.json` to anyone.

    That makes it a distribution of fitted weights, so every training source's terms have to
    survive it. Eight of the 18 heads are fitted on CGMacros (`CC-BY-NC-SA-4.0`), and
    share-alike obliges an adapted work to be offered under the same licence. The bundle may
    therefore contain them only while it declares CC-BY-NC-SA-4.0 itself.

    This test fails if someone adds a share-alike source without relabelling the bundle, or
    relabels the bundle permissively while a share-alike source is still in it. See D025.
    """
    import json
    from pathlib import Path

    published = Path(__file__).resolve().parents[2] / "web" / "public" / "models" / "heads.json"
    if not published.exists():
        pytest.skip("web bundle not staged; run scripts/publish_web_assets.py")

    bundle = json.loads(published.read_text())
    sources = {h["dataset"] for h in bundle["heads"].values()}
    sharealike = {"cgmacros", "glucofm_bench"} & sources
    no_derivatives = {"uchtt1dm"} & sources

    assert not no_derivatives, (
        f"{sorted(no_derivatives)} carry no-derivatives terms; a fitted classifier "
        "may not be redistributed at all."
    )
    if sharealike:
        assert bundle.get("license") == "CC-BY-NC-SA-4.0", (
            f"heads fitted on {sorted(sharealike)} impose share-alike, but the bundle "
            f"declares {bundle.get('license')!r}."
        )


def test_encoder_is_never_licensed_by_a_source_it_did_not_train_on():
    """The share-alike term must not migrate from the heads bundle to the encoder.

    The encoder pretrained on Lane A only. CGMacros never entered it, so nothing about the
    heads bundle's CC-BY-NC-SA licence applies to the encoder, which stays CC-BY-NC-4.0.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    weights_licence = (root / "LICENSE-WEIGHTS").read_text()
    assert "NonCommercial 4.0" in weights_licence
    assert "ShareAlike" not in weights_licence, (
        "LICENSE-WEIGHTS covers the encoder; share-alike belongs to the heads bundle alone."
    )
