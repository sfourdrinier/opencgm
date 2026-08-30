"""CGM-aware augmentations. Blueprint §16.

Four operations, evaluated in random order. After one applies, every subsequent operation's
probability is multiplied by 0.25 — and the reduction compounds, so a third operation faces
0.0625 of its base probability (§16.1). The one-time-only reading is an ablation, not this.

Two kinds, and the distinction matters downstream:

* **value** perturbations (baseline wander, compression drop) change observed values and leave
  the mask untouched.
* **structural** perturbations (decimation, disconnection) remove observations, so they change
  the mask. Nothing may ever *add* an observation.

All randomness comes from a caller-supplied generator so a window's augmentation is a pure
function of (seed, window identity, epoch). Blueprint §17.3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .timestamps import SEQUENCE_LENGTH

SUBSEQUENT_PROBABILITY_MULTIPLIER = 0.25  # PAPER_EXACT §16.1
DECIMATION_MIN_OBSERVED_EXCLUSIVE = 200  # PAPER_EXACT §16.4


@dataclass(frozen=True)
class AugmentationResult:
    values: np.ndarray
    mask: np.ndarray
    applied: tuple[str, ...]


def baseline_wander(
    values: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sinusoidal drift. Blueprint §16.2.

    Amplitude 5-15 mg/dL and 0.5-2 cycles per 24-hour window are PAPER_EXACT; the phase is not
    specified, and a fixed phase would put every augmented window in lockstep, so it is drawn
    uniformly (INFERRED_RECONSTRUCTION, config `augmentations.baseline_wander`).

    Applied only at observed positions, so unobserved slots keep the fill value and the mask is
    unchanged. No clipping: §16.2 does not specify one, and silently clamping would hide
    implausible post-augmentation ranges instead of reporting them.
    """
    amplitude = rng.uniform(5.0, 15.0)
    cycles = rng.uniform(0.5, 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    j = np.arange(SEQUENCE_LENGTH, dtype=np.float64)
    delta = amplitude * np.sin(2.0 * np.pi * cycles * j / SEQUENCE_LENGTH + phase)
    out = values.copy()
    out[mask] = (values[mask] + delta[mask]).astype(values.dtype)
    return out, mask


def compression_drop(
    values: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """V-shaped attenuation, as when a sensor is compressed during sleep. Blueprint §16.3.

    Contiguous span of 6-12 grid positions, multiplier falling linearly from 1 to a minimum of
    0.4-0.7 and back to 1. Mask unchanged: the sensor still reports, it reports low.
    """
    length = int(rng.integers(6, 13))  # 6..12 inclusive
    start = int(rng.integers(0, SEQUENCE_LENGTH - length + 1))
    minimum = rng.uniform(0.4, 0.7)

    # Symmetric ramp down to `minimum` at the centre and back up. Handles odd and even lengths.
    ramp = np.linspace(-1.0, 1.0, length)
    multiplier = minimum + (1.0 - minimum) * np.abs(ramp)

    out = values.copy()
    span = slice(start, start + length)
    observed = mask[span]
    out[span] = np.where(observed, values[span] * multiplier, values[span]).astype(values.dtype)
    return out, mask


def decimation(
    values: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Thin a 5-minute trace to a 15-minute pattern. Blueprint §16.4.

    Applied only when more than 200 positions are observed. Keeps observations whose **absolute
    grid index** satisfies ``j % 3 == offset`` — by index, never by position in a compressed
    list of observations, which would produce a different and cadence-dependent result.
    """
    if int(mask.sum()) <= DECIMATION_MIN_OBSERVED_EXCLUSIVE:
        return values, mask
    offset = int(rng.integers(0, 3))
    j = np.arange(SEQUENCE_LENGTH)
    keep = mask & (j % 3 == offset)
    return values, keep


def disconnection(
    values: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Remove 1-3 contiguous blocks of 2-12 positions. Blueprint §16.5.

    Overlapping blocks are allowed, since nothing in the paper forbids it
    (INFERRED_RECONSTRUCTION). Removed positions take mask False; their values become
    unreadable rather than being reset, which the fill-value invariance tests then police.
    """
    n_blocks = int(rng.integers(1, 4))  # 1..3
    out_mask = mask.copy()
    for _ in range(n_blocks):
        length = int(rng.integers(2, 13))  # 2..12
        start = int(rng.integers(0, SEQUENCE_LENGTH - length + 1))
        out_mask[start : start + length] = False
    return values, out_mask


#: (name, function, base probability). Probabilities are PAPER_EXACT §16.2-§16.5.
OPERATIONS: tuple[tuple[str, Callable, float], ...] = (
    ("baseline_wander", baseline_wander, 0.25),
    ("compression_drop", compression_drop, 0.10),
    ("decimation", decimation, 0.40),
    ("disconnection", disconnection, 0.05),
)

VALUE_OPERATIONS = frozenset({"baseline_wander", "compression_drop"})
STRUCTURAL_OPERATIONS = frozenset({"decimation", "disconnection"})


def augment(
    values: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> AugmentationResult:
    """Apply the four operations in random order with compounding probability decay.

    Blueprint §16.1. The input arrays are never mutated: the cached canonical window must stay
    pristine across epochs, since augmentation is applied online.
    """
    order = rng.permutation(len(OPERATIONS))
    out_values, out_mask = values.copy(), mask.copy()
    scale = 1.0
    applied: list[str] = []

    for idx in order:
        name, fn, base = OPERATIONS[int(idx)]
        if rng.random() < base * scale:
            new_values, new_mask = fn(out_values, out_mask, rng)
            # decimation declines to act below its observation threshold; that is not an
            # application and must not trigger the probability decay.
            if new_mask is out_mask and new_values is out_values:
                continue
            out_values, out_mask = new_values, new_mask
            applied.append(name)
            scale *= SUBSEQUENT_PROBABILITY_MULTIPLIER

    return AugmentationResult(out_values, out_mask, tuple(applied))
