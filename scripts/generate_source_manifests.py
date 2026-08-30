"""Generate per-source SHA-256 manifests.

For each source in `manifests/sources/registry.yaml` with a known `cgm_glob` or `path`, find the
actual files under `data/raw/<path>` and write a manifest next to the registry:

    manifests/sources/<dataset_id>.sha256.json

with the schema from `bundle/glucofm_dataset_manifest_template.yaml`:

  - relative_path, content_role, bytes, sha256

Why this exists: every claim of "the corpus is exactly N files / N bytes" needs a checksum against
which to verify. Without it, the corpus is just a folder; with it, the corpus is a reproducible
input. PR 1 / PR 2 of the blueprint.

Skip rules:
  * Files matching Lane E (`uchtt1dm_EVAL_ONLY_ccbyncnd`) - never commit checksums of
    restricted data.
  * Files larger than 2 GB — emit a sidecar note instead (hashing blocks for hours, and the
    raw bytes never enter a distributed checkpoint anyway).
  * `weight_release: forbidden` — same rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REGISTRY = Path("manifests/sources/registry.yaml")
OUT_DIR = Path("manifests/sources")
RAW_ROOT = Path("data/raw")
MAX_BYTES = 2 * 1024**3  # 2 GB hard cap per file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_one(src: dict) -> dict | None:
    dataset_id = src["dataset_id"]
    lane = src.get("lane", "?")
    weight_release = src.get("weight_release", "unknown")

    if weight_release in {"forbidden"}:
        return {
            "dataset_id": dataset_id, "lane": lane, "files": [],
            "skipped": "weight_release:forbidden",
        }

    raw_path = RAW_ROOT / src["path"]
    if not raw_path.exists():
        return {
            "dataset_id": dataset_id, "lane": lane, "files": [],
            "skipped": f"path missing: {raw_path}",
        }

    # Lane E sources have no manifest at all
    if lane == "E":
        return {
            "dataset_id": dataset_id, "lane": lane, "files": [],
            "skipped": "lane:E (eval-only, no manifest)",
        }

    files = []
    skipped_large = []
    for fp in sorted(raw_path.rglob("*")):
        if not fp.is_file():
            continue
        size = fp.stat().st_size
        if size > MAX_BYTES:
            skipped_large.append({"relative_path": str(fp.relative_to(raw_path)), "bytes": size})
            continue
        rel = str(fp.relative_to(raw_path))
        files.append({
            "relative_path": rel,
            "content_role": "cgm" if "cgm" in rel.lower() or rel.endswith(".csv") else "sidecar",
            "bytes": size,
            "sha256": sha256(fp),
        })

    return {
        "dataset_id": dataset_id,
        "lane": lane,
        "source_version": src.get("source_version"),
        "license": src.get("license"),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "skipped_large": skipped_large,
    }


def main() -> None:
    reg = yaml.safe_load(REGISTRY.read_text())
    sources = reg["sources"]
    summary = []
    for src in sources:
        out = build_one(src)
        if out is None:
            continue
        out_path = OUT_DIR / f"{src['dataset_id']}.sha256.json"
        out_path.write_text(json.dumps(out, indent=2))
        summary.append(
            (
                src["dataset_id"], out.get("n_files", 0),
                out.get("total_bytes", 0), out.get("skipped", "-"),
            )
        )
        print(
            f"  {src['dataset_id']:<28} lane {out.get('lane')}  "
            f"{out.get('n_files', 0):>3} files  "
            f"{out.get('total_bytes', 0) / 1e6:>8.1f} MB  {out.get('skipped', 'OK')}"
        )

    print()
    print(f"wrote {len(summary)} manifests to {OUT_DIR}/")
    print(f"  total: {sum(s[1] for s in summary)} files, {sum(s[2] for s in summary)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
