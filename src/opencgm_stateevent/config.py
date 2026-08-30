"""Config loading with mandatory evidence-status validation.

The blueprint's central discipline (§2): a guessed choice must never silently become
"the GlucoFM recipe". This module refuses to load a config whose ambiguous options are
untagged, so the discipline is enforced by the loader rather than by remembering.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .provenance import canonical_hash


class EvidenceStatus(StrEnum):
    """Blueprint §2. Every consequential choice carries exactly one of these."""

    PAPER_EXACT = "PAPER_EXACT"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    INFERRED_RECONSTRUCTION = "INFERRED_RECONSTRUCTION"
    PROPOSED_EXTENSION = "PROPOSED_EXTENSION"


#: Keys that must never be silently guessed. Each is either disclosed by the paper or
#: explicitly listed in blueprint §3.2 as unavailable; either way the config must say which.
REQUIRES_EVIDENCE_TAG = {
    "binning",
    "coverage_ratio",
    "interpretation",
    "normalization",
    "integer_rule",
    "optimizer",
    "schedule",
    "intra_patch_difference",
    "gate",
    "shared_physical_view_between_branches",
}


class EvidenceError(ValueError):
    """Raised when an ambiguous option is missing its evidence tag."""


class TaggedValue(BaseModel):
    """A value carrying its provenance."""

    value: Any
    evidence_status: EvidenceStatus
    alternatives: list[Any] = Field(default_factory=list)
    unresolved: str | None = None

    @field_validator("evidence_status", mode="before")
    @classmethod
    def _normalize(cls, v: Any) -> Any:
        return str(v).upper() if isinstance(v, str) else v

    def __repr__(self) -> str:  # keeps diagnostics readable in test failures
        return f"TaggedValue({self.value!r}, {self.evidence_status})"


def _walk(node: Any, path: str = "") -> list[str]:
    """Find dict nodes whose key demands an evidence tag but which carry none."""
    problems: list[str] = []
    if isinstance(node, dict):
        for key, child in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key in REQUIRES_EVIDENCE_TAG and isinstance(child, dict):
                has_tag = any(k.startswith("evidence_status") for k in child)
                if not has_tag:
                    problems.append(here)
            problems.extend(_walk(child, here))
    elif isinstance(node, list):
        for i, child in enumerate(node):
            problems.extend(_walk(child, f"{path}[{i}]"))
    return problems


class ResolvedConfig(BaseModel):
    """A loaded config plus its content hash.

    The hash is what checkpoints and cached embeddings reference, so two runs claiming the
    same configuration can be proven identical rather than assumed so.
    """

    model_config = {"extra": "allow"}

    raw: dict[str, Any]
    source_path: str | None = None
    config_hash: str

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch by dotted path, unwrapping ``{value: ...}`` tagged nodes."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        if isinstance(node, dict) and "value" in node:
            return node["value"]
        return node

    def evidence(self, dotted: str) -> EvidenceStatus | None:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        if isinstance(node, dict):
            for key in ("evidence_status", "evidence_status_for_internals"):
                if key in node:
                    return EvidenceStatus(str(node[key]).upper())
        return None

    def inferred_choices(self) -> list[str]:
        """Every INFERRED_RECONSTRUCTION path, for the run's assumption registry.

        These are exactly the forks that must appear in DECISIONS.md before any agent
        fan-out, or parallel work will resolve them inconsistently.
        """
        found: list[str] = []

        def walk(node: Any, path: str = "") -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    here = f"{path}.{key}" if path else str(key)
                    if key.startswith("evidence_status") and (
                        str(child).upper() == EvidenceStatus.INFERRED_RECONSTRUCTION
                    ):
                        found.append(path or here)
                    walk(child, here)
            elif isinstance(node, list):
                for i, child in enumerate(node):
                    walk(child, f"{path}[{i}]")

        walk(self.raw)
        return sorted(set(found))


def load_config(path: str | Path, *, strict: bool = True) -> ResolvedConfig:
    """Load a YAML config, refusing untagged ambiguous options when ``strict``."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    if strict and (problems := _walk(raw)):
        raise EvidenceError(
            "Ambiguous options missing an evidence_status tag "
            f"(blueprint §2): {', '.join(problems)}"
        )
    return ResolvedConfig(raw=raw, source_path=str(p), config_hash=canonical_hash(raw))
