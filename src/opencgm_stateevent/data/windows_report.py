"""Build, compare, and freeze the pretraining window manifest. Blueprint §9.4, §17.4.

    uv run python -m opencgm_stateevent.data.windows_report            # compare candidates
    uv run python -m opencgm_stateevent.data.windows_report --freeze   # write the manifest

The frozen manifest defines an epoch (§17.4): one epoch is one full pass over it. Windows are
never resampled per epoch, so the manifest must be written once and hashed, not regenerated.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..provenance import RunRecord, canonical_hash
from .audit import sources
from .canonical import canonicalize
from .timestamps import segment_readings
from .windowing import SamplerCandidate, sample_sessions, summarize

STRICT = ("big_ideas", "shanghai_t2dm", "stanford", "colas")
MANIFEST_DIR = Path("manifests/windows")


def collect_segments(datasets: tuple[str, ...] = STRICT) -> tuple[list, dict]:
    """Every continuous segment in the given lane, with per-dataset counts."""
    segments: list[tuple[str, int, datetime, datetime, str]] = []
    stats: dict[str, dict[str, int]] = {}
    meta: dict[str, tuple[str, str]] = {}
    for name, fn in sources().items():
        if name not in datasets:
            continue
        total = usable = 0
        for s in canonicalize(fn()):
            for i, (b, e) in enumerate(segment_readings(s.timestamps)):
                total += 1
                start, end = s.timestamps[b], s.timestamps[e - 1]
                if (end - start).total_seconds() >= 86400:
                    usable += 1
                segments.append((s.session_id, i, start, end, s.dataset_id))
                meta[f"{s.session_id}#{i}"] = (s.canonical_subject_id, s.biological_person_id)
        stats[name] = {"segments": total, "segments_ge_24h": usable}
    return segments, {"per_dataset": stats, "identity": meta}


def compare(segments: list, seed: int = 17) -> dict[str, dict]:
    out = {}
    for cand in SamplerCandidate:
        samples = sample_sessions(segments, candidate=cand, global_seed=seed)
        s = summarize(samples)
        s["steps_per_epoch_at_batch_128"] = round(s["windows"] / 128)
        s["total_steps_120_epochs"] = round(s["windows"] / 128 * 120)
        out[cand.value] = s
    return out


def freeze(
    segments: list,
    identity: dict[str, tuple[str, str]],
    *,
    candidate: SamplerCandidate = SamplerCandidate.LEGAL_START_FRACTION,
    seed: int = 17,
) -> Path:
    """Write the immutable window manifest for one seed. Blueprint §17.4."""
    samples = sample_sessions(segments, candidate=candidate, global_seed=seed)
    rows = []
    for s in samples:
        subject, person = identity.get(f"{s.session_id}#{s.segment_index}", ("", ""))
        for start in s.starts:
            rows.append(
                {
                    "session_id": s.session_id,
                    "segment_index": s.segment_index,
                    "canonical_subject_id": subject,
                    "biological_person_id": person,
                    "start_local": start.isoformat(),
                }
            )
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate": candidate.value,
        "seed": seed,
        "n_windows": len(rows),
        "n_segments": len(samples),
        "decision": "DECISIONS.md#D005",
        "windows": rows,
    }
    digest = canonical_hash({k: v for k, v in payload.items() if k != "windows"})
    path = MANIFEST_DIR / f"strict_public_seed{seed}_{candidate.value}.json"
    path.write_text(json.dumps({**payload, "manifest_hash": digest}))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="write the immutable manifest")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rec = RunRecord(command="windows report")
    segments, info = collect_segments()

    print("segments by dataset (total, usable >=24h):")
    for name, st in info["per_dataset"].items():
        print(f"  {name:<15}{st['segments']:>6}{st['segments_ge_24h']:>7}")
    print(f"  {'TOTAL':<15}{len(segments):>6}")
    print()

    results = compare(segments, seed=args.seed)
    print(f"{'candidate':<24}{'windows':>10}{'mean/seg':>10}{'steps/ep':>10}{'120ep steps':>13}")
    for name, s in results.items():
        print(
            f"{name:<24}{s['windows']:>10,}{s['mean_windows_per_segment']:>10.1f}"
            f"{s['steps_per_epoch_at_batch_128']:>10,}{s['total_steps_120_epochs']:>13,}"
        )
    print("\nDecision: legal_start_fraction (D005). See reports/window_sampler.md")

    if args.freeze:
        path = freeze(segments, info["identity"], seed=args.seed)
        print(f"\nfrozen manifest -> {path}")
        rec.outputs["manifest"] = str(path)
    rec.finish(**{k: v["windows"] for k, v in results.items()}).write()


if __name__ == "__main__":
    main()
