"""Source registry: what we have on disk, under which rights, in which lane.

Everything here reads the filesystem. Nothing is remembered. If this module and a note
disagree, this module is right.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path("manifests/sources/registry.yaml")


class Lane(StrEnum):
    A = "A"  # strict public pretraining
    B = "B"  # downstream evaluation
    C = "C"  # public-plus candidates
    D = "D"  # PPG bridge
    E = "E"  # evaluation only: NC/ND/SA, never in a distributed checkpoint


LANE_LABEL = {
    Lane.A: "strict public pretrain",
    Lane.B: "downstream eval",
    Lane.C: "public-plus",
    Lane.D: "PPG bridge",
    Lane.E: "eval only (no distribution)",
}

#: Rights values that bar a source from any checkpoint we intend to publish.
FORBIDS_DISTRIBUTION = {"forbidden_sharealike", "forbidden_noderivatives"}


@dataclass
class SourceStatus:
    dataset_id: str
    lane: Lane
    role: str
    path: Path
    present: bool
    n_files: int
    bytes: int
    cgm_matches: int
    license: str
    weight_release: str
    expected_records: int | None
    expected_hours: int | None
    notes: list[str]

    @property
    def distributable(self) -> bool:
        return self.weight_release not in FORBIDS_DISTRIBUTION

    @property
    def gb(self) -> float:
        return self.bytes / 1e9


def _dir_stats(path: Path) -> tuple[int, int]:
    """(file count, total bytes) following the data/raw symlink."""
    n = total = 0
    if not path.exists():
        return 0, 0
    for p in path.rglob("*"):
        if p.is_file():
            n += 1
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return n, total


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def scan(
    registry_path: str | Path = DEFAULT_REGISTRY, repo_root: Path | None = None
) -> list[SourceStatus]:
    """Check every registered source against the filesystem."""
    reg = load_registry(registry_path)
    root = (repo_root or Path.cwd()) / reg.get("raw_root", "data/raw")

    out: list[SourceStatus] = []
    for entry in reg.get("sources", []):
        p = root / entry["path"]
        n_files, size = _dir_stats(p)
        glob = entry.get("cgm_glob")
        matches = len(list(p.glob(glob))) if (glob and p.exists()) else 0
        out.append(
            SourceStatus(
                dataset_id=entry["dataset_id"],
                lane=Lane(entry["lane"]),
                role=entry.get("role", "?"),
                path=p,
                present=p.exists() and n_files > 0,
                n_files=n_files,
                bytes=size,
                cgm_matches=matches,
                license=entry.get("license", "unresolved"),
                weight_release=entry.get("weight_release", "unresolved"),
                expected_records=entry.get("expected_records"),
                expected_hours=entry.get("expected_hours"),
                notes=entry.get("notes", []) or [],
            )
        )
    return out


def strict_corpus_summary(statuses: list[SourceStatus]) -> dict[str, Any]:
    """Reconcile Lane A against the paper's public-subset targets (blueprint §1.1)."""
    lane_a = [s for s in statuses if s.lane is Lane.A]
    return {
        "sources_present": sum(1 for s in lane_a if s.present),
        "sources_total": len(lane_a),
        "expected_records": sum(s.expected_records or 0 for s in lane_a),
        "expected_hours": sum(s.expected_hours or 0 for s in lane_a),
        "paper_target_records": 285,
        "paper_target_hours": 33736,
    }
