"""Hashing, environment capture, and run records.

Every artifact this project produces must be traceable to the exact bytes, code, and
configuration that made it. Blueprint §4.1, §8.1.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, streamed so multi-GB sources do not land in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj: Any) -> str:
    """Stable hash of a config-like object.

    Sorted keys and no whitespace, so the same resolved configuration always yields the
    same digest regardless of construction order. Blueprint §4.1 requires resolved configs
    to be deterministic and hashable.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_state(repo: str | Path | None = None) -> dict[str, Any]:
    """Current commit and dirty flag. Blueprint §17.5 requires both in every checkpoint."""
    cwd = str(repo) if repo else str(Path(__file__).resolve().parents[2])

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def environment() -> dict[str, Any]:
    """Environment lock detail. Blueprint §4.1: results travel with GPU/runtime details."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:  # torch is an optional extra; absence must not break provenance capture
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        env["torch"] = None
    return env


@dataclass
class RunRecord:
    """Machine-readable record written by every command. Blueprint §25."""

    command: str
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    config_hash: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    evidence_notes: list[str] = field(default_factory=list)
    git: dict[str, Any] = field(default_factory=git_state)
    env: dict[str, Any] = field(default_factory=environment)
    status: str = "running"

    def finish(self, status: str = "ok", **outputs: Any) -> RunRecord:
        self.finished_at = utc_now()
        self.status = status
        self.outputs.update(outputs)
        return self

    def write(self, reports_dir: str | Path = "reports/runs") -> Path:
        d = Path(reports_dir)
        d.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "")
        path = d / f"{stamp}_{self.command.replace(' ', '_')}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path
