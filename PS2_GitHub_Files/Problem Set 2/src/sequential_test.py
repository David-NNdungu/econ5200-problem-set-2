"""One-sided Gaussian mixture sequential test; NumPy and SciPy only.

Known variance is essential. Standardize independent treatment-minus-control
observations using a pre-specified SD, never the running estimated SD.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr


def sequential_test(
    observations: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
    rho: float = 312.5,
    looks: np.ndarray | None = None,
) -> dict[str, np.ndarray | int | bool | None]:
    """Test H0: mean <= 0 against mean > 0 at specified cumulative looks.

    Under independent Gaussian observations with known SD sigma, the half-normal
    mixture is an e-process. P(ever reject a true null) <= alpha, including
    data-dependent stopping. rho is the inverse variance of the half-normal
    mixing distribution on the standardized mean and must be fixed in advance.
    Returns cumulative sample sizes, log e-values, latched decisions, and first
    crossing. The first crossing rejects; no crossing means no rejection.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("observations must be a finite nonempty 1D array")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if not (0 < alpha < 1) or not np.isfinite(rho) or rho <= 0:
        raise ValueError("require 0 < alpha < 1 and finite rho > 0")
    if looks is None:
        t = np.arange(1, x.size + 1, dtype=int)
    else:
        raw = np.asarray(looks)
        if (raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all()
                or np.any(raw != np.floor(raw))):
            raise ValueError("looks must be a nonempty array of integer counts")
        t = raw.astype(int)
        if t[0] < 1 or t[-1] > x.size or np.any(np.diff(t) <= 0):
            raise ValueError("looks must be strictly increasing within the stream")
    s = np.cumsum(x / sigma)[t - 1]
    v = rho + t
    # Integral over theta >= 0 of exp(theta*S - t*theta**2/2)
    # against a HalfNormal(scale=1/sqrt(rho)) mixing density.
    log_e = (np.log(2.0) + 0.5 * (np.log(rho) - np.log(v))
             + s**2 / (2.0 * v) + log_ndtr(s / np.sqrt(v)))
    crossing = log_e >= -np.log(alpha)
    reject_by_look = np.maximum.accumulate(crossing)
    first = int(t[np.flatnonzero(crossing)[0]]) if crossing.any() else None
    return {"n": t, "log_e": log_e, "reject_by_look": reject_by_look,
            "first_reject_n": first, "rejected": bool(crossing.any())}


def replay_portfolio(
    log: object, sigma_pair: float, seed: int, alpha: float = 0.05,
    rho: float = 312.5,
) -> list[dict[str, int | float | bool | None]]:
    """Reconstruct Gaussian paths conditional on the SAME logged final estimates.

    log must provide itertuples() with experiment_id, n_per_arm, estimate, se.
    A normal residual vector minus its mean, plus the logged estimate, is a
    conditional Gaussian path whose terminal mean equals that estimate. This
    uses extra simulated path information, not recovered real historical data.
    """
    rng = np.random.default_rng(seed)
    out = []
    for row in log.itertuples(index=False):
        n = int(row.n_per_arm)
        if n < 2 or not np.isclose(row.se, sigma_pair / np.sqrt(n)):
            raise ValueError("log standard errors do not match known sigma_pair")
        residual = rng.normal(0.0, sigma_pair, size=n)
        stream = residual - residual.mean() + row.estimate
        assert np.isclose(stream.mean(), row.estimate)
        result = sequential_test(stream, sigma_pair, alpha, rho)
        out.append({"experiment_id": int(row.experiment_id),
                    "estimate": float(stream.mean()), "n_per_arm": n,
                    "sequential_ship": bool(result["rejected"]),
                    "first_reject_n": result["first_reject_n"]})
    return out
