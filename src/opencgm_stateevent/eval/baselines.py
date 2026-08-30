"""Transparent baselines. Blueprint §19.6, items 1 and 2.

§19.6 lists ten possible comparators and says plainly not to delay release for all of them: the
must-have set is the transparent classical and raw ones. These two are that set, and they are the
comparisons that actually decide whether the representation is worth anything.

* **`clinical_metrics`** — the hand-engineered daily CGM summary any clinician would compute. If
  a learned representation cannot beat mean, SD, CV, time-in-range and MAGE, it has not earned
  its complexity. This is the bar.
* **`raw_masked`** — the flattened 288-vector with its mask and density, exactly the information
  the encoder receives. It separates "the encoder learned something" from "24 hours of glucose is
  simply informative".

Both are computed mask-aware, from observed samples only, and never interpolate. A baseline that
quietly interpolates would be competing on different information than the model, which makes the
comparison meaningless in whichever direction it lands.
"""

from __future__ import annotations

import numpy as np

from .windows import WindowSet

# Standard CGM ranges in mg/dL. Clinical convention, used here only to build baseline features.
HYPO_LEVEL_2 = 54.0
HYPO_LEVEL_1 = 70.0
TARGET_HIGH = 180.0
HYPER_LEVEL_2 = 250.0
MINUTES_PER_STEP = 5.0


def _masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window mean and standard deviation over observed samples only."""
    n = mask.sum(axis=1)
    safe = np.maximum(n, 1)
    mean = np.where(mask, values, 0.0).sum(axis=1) / safe
    var = np.where(mask, (values - mean[:, None]) ** 2, 0.0).sum(axis=1) / safe
    return {"n": n, "mean": np.where(n > 0, mean, np.nan), "sd": np.sqrt(np.maximum(var, 0.0))}


def _fraction_in(values: np.ndarray, mask: np.ndarray, low: float, high: float) -> np.ndarray:
    """Fraction of *observed* samples in `[low, high)`.

    Normalising by observed count rather than by 288 is what keeps a 15-minute Libre day
    comparable to a 5-minute Dexcom day. Dividing by 288 would make every sparse source look like
    it spent two thirds of the day outside every range.
    """
    inside = mask & (values >= low) & (values < high)
    return inside.sum(axis=1) / np.maximum(mask.sum(axis=1), 1)


def _mage(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean amplitude of glycemic excursion, simplified to excursions exceeding one SD.

    The classical definition needs turning points identified on a continuous trace. On a masked
    grid we take successive observed differences and average those whose magnitude exceeds the
    window's own standard deviation, which captures the same "large swings" quantity without
    requiring the gaps to be filled.
    """
    out = np.zeros(len(values))
    sd = _masked_stats(values, mask)["sd"]
    for i in range(len(values)):
        observed = values[i][mask[i]]
        if len(observed) < 3 or sd[i] == 0:
            continue
        deltas = np.abs(np.diff(observed))
        large = deltas[deltas > sd[i]]
        out[i] = large.mean() if len(large) else 0.0
    return out


def clinical_metrics(ws: WindowSet) -> np.ndarray:
    """Hand-engineered daily CGM summary. §19.6 baseline 1.

    Seventeen features: central tendency, variability, the standard clinical range fractions,
    excursion size, rate of change, and the window's own density and observation count — the
    latter two so the baseline is not handicapped by information the encoder does have.
    """
    v, m = ws.values.astype(np.float64), ws.mask
    stats = _masked_stats(v, m)
    mean, sd = stats["mean"], stats["sd"]

    # Rate of change between consecutive observed samples, mg/dL per minute.
    roc_mean = np.zeros(len(v))
    roc_sd = np.zeros(len(v))
    for i in range(len(v)):
        idx = np.flatnonzero(m[i])
        if len(idx) < 2:
            continue
        rates = np.diff(v[i][idx]) / (np.diff(idx) * MINUTES_PER_STEP)
        roc_mean[i], roc_sd[i] = np.abs(rates).mean(), rates.std()

    columns = [
        mean,
        sd,
        np.divide(sd, mean, out=np.zeros_like(sd), where=mean > 0),  # coefficient of variation
        np.where(stats["n"] > 0, np.nanmax(np.where(m, v, -np.inf), axis=1), np.nan),
        np.where(stats["n"] > 0, np.nanmin(np.where(m, v, np.inf), axis=1), np.nan),
        _fraction_in(v, m, 0.0, HYPO_LEVEL_2),
        _fraction_in(v, m, HYPO_LEVEL_2, HYPO_LEVEL_1),
        _fraction_in(v, m, HYPO_LEVEL_1, TARGET_HIGH),  # time in range
        _fraction_in(v, m, TARGET_HIGH, HYPER_LEVEL_2),
        _fraction_in(v, m, HYPER_LEVEL_2, np.inf),
        _mage(v, m),
        roc_mean,
        roc_sd,
        # `np.percentile` propagates NaN, and every window has unobserved positions, so these
        # two features were identically zero for every window in every dataset.
        np.nanpercentile(np.where(m, v, np.nan), 25, axis=1),
        np.nanpercentile(np.where(m, v, np.nan), 75, axis=1),
        m.mean(axis=1),  # density
        stats["n"].astype(float),
    ]
    features = np.column_stack(columns)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def raw_masked(ws: WindowSet) -> np.ndarray:
    """Flattened 288-vector with mask and density. §19.6 baseline 2.

    Unobserved positions are zero *and* flagged by the mask channel, so the classifier can tell a
    missing sample from a real one. Filling them with the window mean would be interpolation by
    another name and would hand this baseline information the encoder never gets.
    """
    values = np.where(ws.mask, ws.values, 0.0).astype(np.float64)
    return np.column_stack([values, ws.mask.astype(np.float64), ws.mask.mean(axis=1)])


BASELINES = {
    "clinical_metrics": clinical_metrics,
    "raw_masked": raw_masked,
}


def build(name: str, ws: WindowSet) -> np.ndarray:
    return BASELINES[name](ws)
