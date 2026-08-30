"""Window cache and training dataset. Blueprint §8.4, §17.4.

Deterministic unaugmented windows are materialised once and memory-mapped; stochastic
augmentation is applied online (§8.4). The cache is keyed to the frozen window manifest, so an
epoch is one pass over a fixed set of windows and cannot drift between seeds.

353,127 windows at 288 float32 is about 407 MB of values plus 102 MB of mask — small enough to
memory-map and let the page cache hold it, which keeps the GPU fed without a Parquet decode on
every batch.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ..data.augmentations import augment
from ..data.canonical import canonicalize
from ..data.grid import build_window
from ..data.timestamps import SEQUENCE_LENGTH, circadian_start_index, segment_readings

CACHE_DIR = Path("data/canonical/windows")


@dataclass
class CachePaths:
    values: Path
    mask: Path
    meta: Path

    @classmethod
    def for_tag(cls, tag: str, root: Path = CACHE_DIR) -> CachePaths:
        root.mkdir(parents=True, exist_ok=True)
        return cls(root / f"{tag}.values.npy", root / f"{tag}.mask.npy", root / f"{tag}.meta.json")

    def exist(self) -> bool:
        return all(p.exists() for p in (self.values, self.mask, self.meta))


def resolve_manifest(path: Path) -> Path:
    """Return the manifest that exists, preferring the gzipped copy.

    The uncompressed JSON is 64 MB and compresses to 1 MB, so the repository ships the
    gzipped form. Either is accepted: someone who unpacked it, or who regenerated it, should
    not have to re-compress before anything will read it.
    """
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"no window manifest at {path} or {gz}")


def read_manifest(path: Path) -> dict:
    """Parse a window manifest, gzipped or not."""
    resolved = resolve_manifest(path)
    if resolved.suffix == ".gz":
        with gzip.open(resolved, "rt") as fh:
            return json.load(fh)
    return json.loads(resolved.read_text())


def manifest_sha256(path: Path) -> str:
    """SHA-256 of the manifest's *uncompressed* bytes.

    Run records have always stored the uncompressed hash. Hashing the archive instead would
    silently break every existing record's provenance chain the moment the file was packed.
    """
    resolved = resolve_manifest(path)
    raw = resolved.read_bytes()
    data = gzip.decompress(raw) if resolved.suffix == ".gz" else raw
    return hashlib.sha256(data).hexdigest()


def build_cache(manifest_path: Path, sources_fn, tag: str = "strict_seed17") -> CachePaths:
    """Materialise every window named in the frozen manifest.

    Windows are built from the canonical sessions rather than re-read from source, so the
    cache inherits the row accounting and deduplication already applied (D007).
    """
    manifest = read_manifest(Path(manifest_path))
    wanted: dict[str, list[str]] = {}
    for row in manifest["windows"]:
        wanted.setdefault(row["session_id"], []).append(row["start_local"])

    paths = CachePaths.for_tag(tag)
    n = len(manifest["windows"])
    values = np.lib.format.open_memmap(
        paths.values, mode="w+", dtype=np.float32, shape=(n, SEQUENCE_LENGTH)
    )
    masks = np.lib.format.open_memmap(
        paths.mask, mode="w+", dtype=bool, shape=(n, SEQUENCE_LENGTH)
    )
    starts = np.zeros(n, dtype=np.int16)
    subjects: list[str] = []
    people: list[str] = []
    datasets: list[str] = []

    cursor = 0
    for name, fn in sources_fn.items():
        for session in canonicalize(fn()):
            if session.session_id not in wanted:
                continue
            ts, vs = session.timestamps, session.values_mg_dl
            for iso in wanted[session.session_id]:
                start = datetime.fromisoformat(iso)
                w = build_window(
                    ts, vs, start,
                    dataset_id=session.dataset_id,
                    canonical_subject_id=session.canonical_subject_id,
                    biological_person_id=session.biological_person_id,
                    session_id=session.session_id,
                )
                values[cursor] = w.values
                masks[cursor] = w.mask
                starts[cursor] = circadian_start_index(start)
                subjects.append(session.canonical_subject_id)
                people.append(session.biological_person_id)
                datasets.append(session.dataset_id)
                cursor += 1
        del name

    values.flush()
    masks.flush()
    paths.meta.write_text(
        json.dumps(
            {
                "n": cursor,
                "manifest_hash": manifest.get("manifest_hash"),
                "seed": manifest.get("seed"),
                "candidate": manifest.get("candidate"),
                "circadian_start": starts[:cursor].tolist(),
                "subject": subjects,
                "person": people,
                "dataset": datasets,
            }
        )
    )
    return paths


def _encode(labels: list[str]) -> tuple[list[str], np.ndarray]:
    """``["a","b","a"] -> (["a","b"], array([0,1,0]))``. Order of first appearance, so the
    vocabulary is stable across runs and usable as a grouping key in split manifests."""
    vocab: list[str] = []
    index: dict[str, int] = {}
    codes = np.empty(len(labels), dtype=np.int32)
    for i, label in enumerate(labels):
        if label not in index:
            index[label] = len(vocab)
            vocab.append(label)
        codes[i] = index[label]
    return vocab, codes


class WindowDataset(Dataset):
    """Cached windows with online augmentation. Blueprint §8.4, §16."""

    def __init__(
        self,
        paths: CachePaths,
        *,
        seed: int = 17,
        augment_enabled: bool = True,
        dense_interpolation: bool = False,
        indices: np.ndarray | None = None,
        epoch: int = 0,
    ) -> None:
        meta = json.loads(paths.meta.read_text())
        self.n_total = meta["n"]
        self.values = np.load(paths.values, mmap_mode="r")
        self.mask = np.load(paths.mask, mmap_mode="r")
        self.circadian = np.asarray(meta["circadian_start"], dtype=np.int64)
        # Grouping labels are held as integer codes plus a vocabulary rather than as
        # 353,127-element string lists. Under DataLoader's spawn start method the dataset is
        # pickled to every worker, and with concurrent seeds that is tens of workers each paying
        # for the same strings. The vocabularies are what the split logic needs anyway.
        self.subject_vocab, self.subject_code = _encode(meta["subject"])
        self.person_vocab, self.person_code = _encode(meta["person"])
        self.dataset_vocab, self.dataset_code = _encode(meta["dataset"])
        self.indices = (
            np.arange(self.n_total) if indices is None else np.asarray(indices, dtype=np.int64)
        )
        self.seed = seed
        self.augment_enabled = augment_enabled
        #: Tier-1 ablation 6. The project's standing rule is that CGM is never interpolated and
        #: the physical mask is authoritative; this switch deliberately violates it so the cost
        #: of that rule can be measured rather than asserted. It must never be on outside a
        #: labelled ablation run.
        self.dense_interpolation = dense_interpolation
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def reindexed(self, indices: np.ndarray) -> WindowDataset:
        """A view over the same memory-mapped arrays in a different order.

        Shallow: the memmaps and label codes are shared, so building one per epoch costs an
        array copy of the index rather than a re-open of half a gigabyte.
        """
        view = copy.copy(self)
        view.indices = np.asarray(indices, dtype=np.int64)
        return view

    def set_epoch(self, epoch: int) -> None:
        """Augmentation varies by epoch while staying reproducible from (seed, index, epoch)."""
        self.epoch = epoch

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        values = np.array(self.values[idx], dtype=np.float32)
        mask = np.array(self.mask[idx], dtype=bool)
        if self.augment_enabled:
            rng = np.random.default_rng((self.seed * 1_000_003 + idx) * 7919 + self.epoch)
            result = augment(values, mask, rng)
            values, mask = result.values, result.mask
        if self.dense_interpolation:
            values, mask = interpolate_dense(values, mask)
        return values, mask, int(self.circadian[idx])


def interpolate_dense(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill every gap by linear interpolation and declare the window fully observed.

    Tier-1 ablation 6, and the only place in this codebase that interpolates CGM. The point is to
    measure what the mask buys: an interpolated window looks complete to the model, so the density
    weighting, the mask-aware statistics and the rate-of-change validity all become no-ops, and
    the model is told a sensor gap and a flat stretch are the same thing.

    Edges are held constant rather than extrapolated -- inventing a trend beyond the last
    observation would be a second, separate fabrication.
    """
    index = np.flatnonzero(mask)
    if len(index) == 0:
        return values, mask
    filled = np.interp(np.arange(len(values)), index, values[index]).astype(values.dtype)
    return filled, np.ones_like(mask)


def build_segments_for_manifest(sources_fn, datasets: tuple[str, ...]):
    """Helper mirroring windows_report.collect_segments, used by the cache builder."""
    out = []
    for name, fn in sources_fn.items():
        if name not in datasets:
            continue
        for s in canonicalize(fn()):
            for i, (b, e) in enumerate(segment_readings(s.timestamps)):
                out.append((s.session_id, i, s.timestamps[b], s.timestamps[e - 1], s.dataset_id))
    return out
