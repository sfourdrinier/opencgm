"""Downstream evaluation windows. Blueprint §9.5, §9.6.

Non-overlapping 24-hour windows, built from the same canonical layer and the same binning rule as
pretraining so that a representation is never asked to generalise across a preprocessing change.

Two rules from §9.6 matter here and are easy to get subtly wrong:

* the only base exclusion is a window with zero observations. There is no hidden minimum count.
  A 15-minute Libre day is sparse, not invalid, and dropping it would quietly restrict several
  tasks to their densest subjects — a selection effect that would flatter every method equally
  and be invisible in the results.
* density is recorded, never used to filter. It belongs in the report, and in ablations if we
  choose to test thresholds explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.audit import RAW, sources
from ..data.canonical import canonicalize
from ..data.grid import build_window, non_overlapping_starts
from ..data.readers import read_cgmacros
from ..data.timestamps import SEQUENCE_LENGTH, circadian_start_index, segment_readings

CACHE = Path("data/canonical/eval")

#: Downstream sources, §6.8 and §19.3. Source roots come from `data.audit.sources` so the
#: evaluation reads exactly what pretraining read; a second copy of those paths would drift.
#: CGMacros is read once per sensor at native cadence (§19.10); the two streams stay separate and
#: are paired by subject only at split time.
def _downstream() -> dict:
    shared = sources()
    return {
        "cgmacros_dexcom": lambda: read_cgmacros(RAW / "cgmacros", "Dexcom GL"),
        "cgmacros_libre": lambda: read_cgmacros(RAW / "cgmacros", "Libre GL"),
        "hall": shared["hall"],
        "stanford": shared["stanford"],
        "shanghai_t2dm": shared["shanghai_t2dm"],
    }


DOWNSTREAM = _downstream()

#: Which label table each window source draws from.
LABEL_SOURCE = {
    "cgmacros_dexcom": "cgmacros",
    "cgmacros_libre": "cgmacros",
    "hall": "hall",
    "stanford": "stanford",
    "shanghai_t2dm": "shanghai_t2dm",
}


@dataclass
class WindowSet:
    """Non-overlapping windows for one downstream source."""

    source: str
    values: np.ndarray  # [N, 288] float32
    mask: np.ndarray  # [N, 288] bool
    circadian: np.ndarray  # [N] int64
    subjects: np.ndarray  # [N] str — the grouping key (biological person)
    sessions: np.ndarray  # [N] str
    entries: np.ndarray | None = None  # [N] str — the label join key, §19.3
    starts: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def density(self) -> np.ndarray:
        return self.mask.mean(axis=1)

    def summary(self) -> dict:
        return {
            "source": self.source,
            "windows": len(self),
            "subjects": len(np.unique(self.subjects)),
            "density_mean": float(self.density.mean()) if len(self) else 0.0,
            "density_min": float(self.density.min()) if len(self) else 0.0,
            "windows_per_subject_median": (
                float(np.median(np.bincount(
                    np.unique(self.subjects, return_inverse=True)[1]
                ))) if len(self) else 0.0
            ),
        }


def subject_key(source: str, canonical_subject_id: str) -> str:
    """Join key between a window and its label row.

    Canonical subject ids carry a dataset prefix (`hall/1636-69-001`); the label tables are keyed
    on the source's own identifier. Both CGMacros sensors map to the same participant, which is
    what lets §19.3 pair them in one split.
    """
    return canonical_subject_id.split("/", 1)[-1]


#: Sources whose labels attach to a visit rather than to a person. §19.3 counts ShanghaiT2DM as
#: "65 labeled sessions from 58 biological participants", so its label key is the entry.
ENTRY_LEVEL_SOURCES = frozenset({"shanghai_t2dm"})


def entry_key(source: str, session_id: str, canonical_subject_id: str) -> str:
    """The identity a label attaches to.

    For most sources that is the person. For an entry-level source it is `patient/visit_n`. Folds
    group by person either way, so a patient's two visits can never straddle a split (§9.5).
    """
    if source in ENTRY_LEVEL_SOURCES:
        return session_id.split("/", 1)[-1]
    return subject_key(source, canonical_subject_id)


def build_windows(source: str) -> WindowSet:
    """Every non-overlapping 24-hour window in `source` with at least one observation."""
    values, masks, circadian, subjects, sessions, entries, starts = ([] for _ in range(7))
    for session in canonicalize(DOWNSTREAM[source]()):
        ts, vs = session.timestamps, session.values_mg_dl
        for begin, end in segment_readings(ts):
            for start in non_overlapping_starts(ts[begin], ts[end - 1]):
                w = build_window(
                    ts, vs, start,
                    dataset_id=session.dataset_id,
                    canonical_subject_id=session.canonical_subject_id,
                    biological_person_id=session.biological_person_id,
                    session_id=session.session_id,
                )
                if not w.mask.any():
                    continue  # §9.6: the only base exclusion
                values.append(w.values)
                masks.append(w.mask)
                circadian.append(circadian_start_index(start))
                subjects.append(subject_key(source, session.canonical_subject_id))
                sessions.append(session.session_id)
                entries.append(
                    entry_key(source, session.session_id, session.canonical_subject_id)
                )
                starts.append(start.isoformat())

    empty = np.zeros((0, SEQUENCE_LENGTH), dtype=np.float32)
    return WindowSet(
        source=source,
        values=np.asarray(values, dtype=np.float32) if values else empty,
        mask=np.asarray(masks, dtype=bool) if masks else empty.astype(bool),
        circadian=np.asarray(circadian, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=object),
        sessions=np.asarray(sessions, dtype=object),
        entries=np.asarray(entries, dtype=object),
        starts=starts,
    )


def cache_path(source: str) -> Path:
    return CACHE / f"{source}.npz"


def load_or_build(source: str, *, rebuild: bool = False) -> WindowSet:
    path = cache_path(source)
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return WindowSet(
            source=source, values=z["values"], mask=z["mask"], circadian=z["circadian"],
            subjects=z["subjects"], sessions=z["sessions"],
            entries=z["entries"] if "entries" in z.files else z["subjects"],
            starts=list(z["starts"]),
        )
    ws = build_windows(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, values=ws.values, mask=ws.mask, circadian=ws.circadian,
        subjects=ws.subjects, sessions=ws.sessions, entries=ws.entries,
        starts=np.asarray(ws.starts, dtype=object),
    )
    return ws


def build_all(*, rebuild: bool = False) -> dict[str, WindowSet]:
    return {s: load_or_build(s, rebuild=rebuild) for s in DOWNSTREAM}


def report(sets: dict[str, WindowSet]) -> str:
    rows = [s.summary() for s in sets.values()]
    (CACHE / "windows_summary.json").write_text(json.dumps(rows, indent=2))
    header = (
        f"{'source':<18} {'windows':>8} {'subjects':>9} "
        f"{'density':>8} {'min':>6} {'med/subj':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['source']:<18} {r['windows']:>8,} {r['subjects']:>9} "
            f"{r['density_mean']:>8.3f} {r['density_min']:>6.3f} "
            f"{r['windows_per_subject_median']:>9.0f}"
        )
    return "\n".join(lines)
