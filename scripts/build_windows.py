#!/usr/bin/env python3
"""Materialise the strict pretraining window cache from the frozen manifest.

`REPRODUCE.md` §2 told the reader to run `just build-windows`, and there was no such recipe
and no such script -- the cache had only ever been built by hand. This is that step, made
real, so a cold checkout can reach the same 353,127 windows the paper reports.

The manifest (`manifests/windows/strict_public_seed17_legal_start_fraction.json`) is frozen
and tracked in git: it names every window by session and local start time. Windows are
rebuilt from the *canonical* sessions rather than re-read from source, so the cache inherits
the row accounting and de-duplication already applied under D007.

Lane A only -- big_ideas, stanford, shanghai_t2dm, colas. Nothing else may enter the strict
corpus; the lane rules are enforced in `tests/unit/test_source_rights.py`.

    uv run python scripts/build_windows.py              # build if missing
    uv run python scripts/build_windows.py --force      # rebuild from scratch
    uv run python scripts/build_windows.py --source colas   # re-run one reader, then rebuild
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from opencgm_stateevent.data.readers import READERS  # noqa: E402
from opencgm_stateevent.train.dataset import CachePaths, build_cache  # noqa: E402

MANIFEST = REPO_ROOT / "manifests" / "windows" / "strict_public_seed17_legal_start_fraction.json"
STRICT_SOURCES = ("big_ideas", "stanford", "shanghai_t2dm", "colas")
EXPECTED_WINDOWS = 353_127


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tag", default="strict_seed17", help="cache tag under data/canonical/windows/"
    )
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    ap.add_argument(
        "--source",
        default="",
        help="restrict to one source (diagnosing a failing reader); the cache is then partial",
    )
    args = ap.parse_args()

    if not (MANIFEST.exists() or MANIFEST.with_suffix(".json.gz").exists()):
        print(f"error: frozen window manifest missing: {MANIFEST}[.gz]", file=sys.stderr)
        print("       it is tracked in git; a clean checkout has it.", file=sys.stderr)
        return 2

    paths = CachePaths.for_tag(args.tag)
    if paths.exist() and not args.force:
        meta = json.loads(paths.meta.read_text())
        print(f"cache already built: {paths.values} ({meta['n']:,} windows)")
        print("pass --force to rebuild")
        return 0

    if args.source:
        if args.source not in READERS:
            print(
                f"error: unknown source {args.source!r}; known: {', '.join(READERS)}",
                file=sys.stderr,
            )
            return 2
        sources = {args.source: READERS[args.source]}
        print(f"building PARTIAL cache from {args.source} only -- not the strict corpus")
    else:
        sources = {name: READERS[name] for name in STRICT_SOURCES}
        print(f"building strict cache from {', '.join(STRICT_SOURCES)}")

    built = build_cache(MANIFEST, sources, tag=args.tag)
    meta = json.loads(built.meta.read_text())
    counts = collections.Counter(meta["dataset"])

    print(f"\nwrote {built.values}")
    print(f"      {built.mask}")
    print(f"      {built.meta}")
    print(f"\n{meta['n']:,} windows, {len(set(meta['subject']))} subjects")
    for name, n in counts.most_common():
        print(f"  {name:>16}  {n:>8,}")

    if not args.source and meta["n"] != EXPECTED_WINDOWS:
        print(
            f"\nWARNING: expected {EXPECTED_WINDOWS:,} windows, built {meta['n']:,}. "
            "The corpus no longer matches the published figure -- do not report numbers "
            "from this cache until the difference is explained.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
