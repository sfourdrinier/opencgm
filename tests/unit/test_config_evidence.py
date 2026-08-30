"""The evidence discipline must be enforced by the loader, not by remembering it.

Blueprint §2: a guessed choice must never silently become "the GlucoFM recipe".
"""

from __future__ import annotations

import pytest
import yaml

from opencgm_stateevent.config import EvidenceError, EvidenceStatus, load_config

INFERRED = "INFERRED_RECONSTRUCTION"


def _write(tmp_path, obj, name: str = "cfg.yaml") -> str:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(obj))
    return str(p)


def test_untagged_ambiguous_option_is_rejected(tmp_path):
    """An untagged binning rule is exactly the failure mode §2 exists to prevent."""
    path = _write(tmp_path, {"alignment": {"binning": {"default": "floor"}}})
    with pytest.raises(EvidenceError, match="binning"):
        load_config(path)


def test_tagged_option_loads(tmp_path):
    path = _write(
        tmp_path,
        {"alignment": {"binning": {"default": "floor", "evidence_status": INFERRED}}},
    )
    cfg = load_config(path)
    assert cfg.evidence("alignment.binning") is EvidenceStatus.INFERRED_RECONSTRUCTION


def test_config_hash_is_order_independent(tmp_path):
    """Two configs with the same content must prove identical, whatever the key order."""
    a = load_config(_write(tmp_path, {"x": 1, "y": 2}, "a.yaml"))
    p2 = tmp_path / "b.yaml"
    p2.write_text("y: 2\nx: 1\n")
    b = load_config(str(p2))
    assert a.config_hash == b.config_hash


def test_config_hash_changes_with_content(tmp_path):
    """A hash that does not move on edit would silently certify the wrong run."""
    a = load_config(_write(tmp_path, {"x": 1}, "a.yaml"))
    b = load_config(_write(tmp_path, {"x": 2}, "b.yaml"))
    assert a.config_hash != b.config_hash


def test_inferred_choices_are_enumerable(tmp_path):
    """Every inferred fork must be listable, so it can reach DECISIONS.md before fan-out."""
    path = _write(
        tmp_path,
        {
            "normalization": {"type": "observed_instance_norm", "evidence_status": INFERRED},
            "alignment": {"grid_minutes": {"value": 5, "evidence_status": "PAPER_EXACT"}},
        },
    )
    cfg = load_config(path)
    inferred = cfg.inferred_choices()
    assert "normalization" in inferred
    assert not any("grid_minutes" in c for c in inferred)


def test_reference_config_is_valid_and_declares_its_assumptions():
    """The shipped reference config must pass its own gate and expose its inferred forks."""
    cfg = load_config("bundle/glucofm_reference_config.yaml")
    assert cfg.get("alignment.sequence_length") == 288
    assert cfg.get("alignment.patches") == 24
    assert cfg.get("alignment.steps_per_patch") == 12
    assert cfg.evidence("alignment.sequence_length") is EvidenceStatus.PAPER_EXACT
    # The paper leaves real ambiguity; a config claiming none would be lying.
    assert len(cfg.inferred_choices()) >= 5


def test_paper_exact_geometry_is_self_consistent():
    """288 = 24 patches x 12 positions. A config violating this cannot be the paper's."""
    cfg = load_config("bundle/glucofm_reference_config.yaml")
    assert cfg.get("alignment.patches") * cfg.get("alignment.steps_per_patch") == cfg.get(
        "alignment.sequence_length"
    )
