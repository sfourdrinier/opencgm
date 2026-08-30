"""NumPy reference implementation of the state/event front end.

Written directly from the blueprint equations in float64, with explicit loops where the
equation is a sum, and deliberately not optimised. Its job is to be obviously correct rather
than fast, so the vectorised PyTorch version can be checked against it.

The pairing is the point. Two independent readings of an ambiguous spec surface the ambiguity;
one reading silently resolves it. Anywhere the two disagree, the disagreement is a finding.

Covers blueprint §11 (normalization and statistics) and §12 (causal Gaussian decomposition).
"""

from __future__ import annotations

import numpy as np

SIGMA_MIN = 2.0  # PAPER_EXACT §12.1
SIGMA_MAX = 12.0  # PAPER_EXACT §12.1
SIGMA_INIT = 6.0  # PAPER_EXACT §12.1
RADIUS = 36  # ceil(3 * sigma_max), PAPER_EXACT §12.1
EPS = 1e-6  # INFERRED_RECONSTRUCTION §11.1
SCALE_MIN = 1e-4  # INFERRED_RECONSTRUCTION §11.1
ROC_MAX_BACK = 9  # PAPER_EXACT §11.4
STEPS_PER_PATCH = 12
PATCHES = 24


def sigma_from_rho(rho: float) -> float:
    """sigma = 2 + 10 * sigmoid(rho). Blueprint §12.1, keeps sigma inside [2, 12]."""
    return SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) / (1.0 + np.exp(-rho))


def rho_from_sigma(sigma: float) -> float:
    """Inverse, used to initialise rho so that sigma starts at 6."""
    z = (sigma - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN)
    return float(np.log(z / (1.0 - z)))


def gaussian_kernel(sigma: float, radius: int = RADIUS) -> np.ndarray:
    """One-sided normalised Gaussian weights over lags 0..R. Blueprint §12.1."""
    r = np.arange(radius + 1, dtype=np.float64)
    w = np.exp(-(r**2) / (2.0 * sigma**2))
    return w / w.sum()


def normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Observed-only per-window instance normalization. Blueprint §11.1.

    Statistics are computed over observed positions only, and the output is zeroed wherever the
    mask is false, so the fill value can never enter. No learned affine.
    """
    values = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    n = m.sum()
    if n == 0:
        return np.zeros_like(values)
    mu = (m * values).sum() / (n + EPS)
    var = (m * (values - mu) ** 2).sum() / (n + EPS)
    scale = max(np.sqrt(var + EPS), SCALE_MIN)
    return m * (values - mu) / scale


def causal_gaussian_state(
    normalized: np.ndarray, mask: np.ndarray, sigma: float, radius: int = RADIUS
) -> np.ndarray:
    """Mask-aware one-sided causal Gaussian smoother. Blueprint §12.1.

        State_j = sum_r K(r) M_{j-r} X_{j-r} / (sum_r K(r) M_{j-r} + eps)

    Strictly causal: only lags r >= 0 contribute, so a value at j+1 can never influence j.
    Negative indices are ignored rather than wrapped. The denominator is masked, so absent
    points do not drag the estimate toward the fill value; where no support exists the output
    is exactly zero rather than NaN.
    """
    x = np.asarray(normalized, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    k = gaussian_kernel(sigma, radius)
    length = x.shape[-1]
    out = np.zeros(length, dtype=np.float64)
    for j in range(length):
        num = 0.0
        den = 0.0
        for r in range(radius + 1):
            i = j - r
            if i < 0:
                break
            num += k[r] * m[i] * x[i]
            den += k[r] * m[i]
        out[j] = num / (den + EPS)
    return out


def decompose(
    values: np.ndarray, mask: np.ndarray, sigma: float = SIGMA_INIT
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full front end: normalize, smooth, take the residual. Returns (normalized, state, event).

    Event = (X - State) * M. Blueprint §12.1.
    """
    x = normalize(values, mask)
    state = causal_gaussian_state(x, mask, sigma)
    event = (x - state) * np.asarray(mask, dtype=np.float64)
    return x, state, event


def patch_density(mask: np.ndarray) -> np.ndarray:
    """d_i = (1/12) sum_{j in patch i} M_j. Blueprint §11.2 — PAPER_EXACT."""
    return np.asarray(mask, dtype=np.float64).reshape(PATCHES, STEPS_PER_PATCH).mean(axis=1)


def patch_mean_std(signal: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observed-only mean and standard deviation per one-hour patch. Blueprint §11.3.

    Empty patches yield zero for both, and the caller is expected to carry the density as the
    validity signal rather than inspecting the statistics.
    """
    s = np.asarray(signal, dtype=np.float64).reshape(PATCHES, STEPS_PER_PATCH)
    m = np.asarray(mask, dtype=np.float64).reshape(PATCHES, STEPS_PER_PATCH)
    n = m.sum(axis=1)
    mu = (m * s).sum(axis=1) / (n + EPS)
    var = (m * (s - mu[:, None]) ** 2).sum(axis=1) / (n + EPS)
    return np.where(n > 0, mu, 0.0), np.where(n > 0, np.sqrt(var), 0.0)


def rate_of_change(
    normalized: np.ndarray, mask: np.ndarray, max_back: int = ROC_MAX_BACK
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-looking rate of change. Blueprint §11.4 — PAPER_EXACT.

        b = min{ k in [1,9] : M_{j-k} = 1 },  r_j = (X_j - X_{j-b}) / b

    Units are normalized-glucose change per five-minute grid step, matching the paper's
    equation. A per-minute conversion would be an extension. Positions with no observed
    predecessor within nine steps get r = 0 and validity false.
    """
    x = np.asarray(normalized, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    length = x.shape[-1]
    roc = np.zeros(length, dtype=np.float64)
    valid = np.zeros(length, dtype=bool)
    for j in range(length):
        if not m[j]:
            continue
        for b in range(1, max_back + 1):
            i = j - b
            if i < 0:
                break
            if m[i]:
                roc[j] = (x[j] - x[i]) / b
                valid[j] = True
                break
    return roc, valid


def intra_patch_difference(
    state: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Adjacent state differences within each patch. Blueprint §11.5.

    INFERRED_RECONSTRUCTION: the paper gives a 16-dimensional trend-difference feature and a
    patch-level Diff path, but never says what sequence it differences. We difference the state
    stream and never bridge a patch boundary, so patch i's first position has no predecessor.
    Returns (PATCHES, 11) differences and their validity.
    """
    s = np.asarray(state, dtype=np.float64).reshape(PATCHES, STEPS_PER_PATCH)
    m = np.asarray(mask, dtype=bool).reshape(PATCHES, STEPS_PER_PATCH)
    diff = s[:, 1:] - s[:, :-1]
    valid = m[:, 1:] & m[:, :-1]
    return diff * valid, valid
