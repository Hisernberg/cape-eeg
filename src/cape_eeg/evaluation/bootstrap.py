"""Paired patient/component cluster bootstrap, Holm adjustment and MDE planning.

Implements section 10 of ``docs/04_Evaluation_and_Statistics.md``: the primary
inference resamples independent groups (patients or connected components) with
replacement, recomputes the paired difference of patient-mean losses in every
replicate, and reports percentile 2.5% / 97.5% limits.  Seeds are averaged as
losses, never as probabilities.  The random generator is
``numpy.random.default_rng(seed)`` and the seed is returned with every result.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm

__all__ = [
    "DEFAULT_N_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "paired_cluster_bootstrap",
    "cluster_bootstrap_statistic",
    "holm_adjust",
    "mde_paired",
]

DEFAULT_N_REPLICATES: int = 2000
DEFAULT_BOOTSTRAP_SEED: int = 20260907
_PERCENTILES: Tuple[float, float] = (2.5, 97.5)
_CHUNK_REPLICATES: int = 256

ArrayLike = Any
StatisticFn = Callable[[np.ndarray], Union[float, np.ndarray]]


def _as_group_losses(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape [G] or [G, S], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _check_replicates(n_replicates: int) -> None:
    if int(n_replicates) < 1:
        raise ValueError("n_replicates must be at least 1")


def paired_cluster_bootstrap(
    per_group_loss_a: ArrayLike,
    per_group_loss_b: ArrayLike,
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Percentile bootstrap of ``Delta = mean_g(a) - mean_g(b)`` over independent groups.

    ``per_group_loss_a`` / ``per_group_loss_b`` are per-group losses aligned
    by group, shape ``[G]`` or ``[G, S]`` for ``S`` seeds (each entry already a
    per-group mean over that group's rows).  Groups are resampled with
    replacement; inside each replicate the resampled per-group values are
    averaged over seeds and then over groups (equivalently ``mean_g mean_s``),
    and the statistic is the difference of the two arm means.  A negative
    ``point_estimate`` favours arm ``a``; superiority requires the whole
    interval below zero (``interval_wholly_below_zero``).

    Returns point estimate, both arm means, relative effect, percentile
    2.5 / 97.5 limits, bootstrap mean and SD, ``n_groups``, ``n_seeds``,
    ``n_replicates`` and ``seed``.  The replicate array itself is not returned.
    """
    a = _as_group_losses(per_group_loss_a, "per_group_loss_a")
    b = _as_group_losses(per_group_loss_b, "per_group_loss_b")
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must share a shape, got {a.shape} and {b.shape}")
    n_groups = int(a.shape[0])
    if n_groups < 2:
        raise ValueError("paired bootstrap needs at least two independent groups")
    _check_replicates(n_replicates)
    n_replicates = int(n_replicates)

    a2 = a[:, None] if a.ndim == 1 else a
    b2 = b[:, None] if b.ndim == 1 else b
    n_seeds = int(a2.shape[1])

    mean_a = float(a2.mean())
    mean_b = float(b2.mean())
    point = mean_a - mean_b

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_replicates, n_groups))
    replicates = np.empty(n_replicates, dtype=np.float64)
    for start in range(0, n_replicates, _CHUNK_REPLICATES):
        idx = draws[start : start + _CHUNK_REPLICATES]
        # [R, G, S] -> mean over seeds (axis 2) then over groups (axis 1).
        rep_a = a2[idx].mean(axis=2).mean(axis=1)
        rep_b = b2[idx].mean(axis=2).mean(axis=1)
        replicates[start : start + idx.shape[0]] = rep_a - rep_b

    ci_low, ci_high = (float(x) for x in np.percentile(replicates, _PERCENTILES))
    return {
        "statistic": "mean_g mean_s loss_a - mean_g mean_s loss_b",
        "point_estimate": point,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "relative_effect": (point / mean_b) if mean_b != 0.0 else None,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": 0.95,
        "ci_method": "percentile",
        "percentiles": list(_PERCENTILES),
        "bootstrap_mean": float(replicates.mean()),
        "bootstrap_sd": float(replicates.std(ddof=1)) if n_replicates > 1 else None,
        "n_groups": n_groups,
        "n_seeds": n_seeds,
        "n_replicates": n_replicates,
        "seed": int(seed),
        "resampling_unit": "independent group (patient / connected component)",
        "interval_wholly_below_zero": bool(ci_high < 0.0),
        "interval_crosses_zero": bool(ci_low <= 0.0 <= ci_high),
    }


def cluster_bootstrap_statistic(
    statistic_fn: StatisticFn,
    groups: ArrayLike,
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    percentiles: Tuple[float, float] = _PERCENTILES,
) -> Dict[str, Any]:
    """Generic cluster bootstrap of a row-level statistic.

    ``groups`` gives the independent-group id of every row (length ``N``).
    ``statistic_fn(row_indices)`` receives an int array of row indices (with
    repetition when a group is drawn more than once) and returns a scalar or a
    1-D array computed on the caller's row-level arrays indexed by those rows;
    the point estimate uses ``np.arange(N)``.  Row indices are pre-grouped
    once so each replicate is a vectorised gather.  Replicates in which the
    statistic is non-finite (for example an AUROC with no positives) are
    dropped from the percentile computation and counted in
    ``n_nonfinite_replicates``.
    """
    grp = np.asarray(groups)
    if grp.ndim != 1 or grp.shape[0] == 0:
        raise ValueError("groups must be a non-empty one-dimensional array")
    _check_replicates(n_replicates)
    n_replicates = int(n_replicates)
    n_rows = int(grp.shape[0])

    uniq, inverse = np.unique(grp, return_inverse=True)
    n_groups = int(uniq.size)
    sorted_rows = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    point = np.asarray(statistic_fn(np.arange(n_rows)), dtype=np.float64)
    if point.ndim > 1:
        raise ValueError("statistic_fn must return a scalar or a one-dimensional array")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_replicates, n_groups))
    values = np.empty((n_replicates,) + point.shape, dtype=np.float64)
    for r in range(n_replicates):
        chosen = draws[r]
        lens = counts[chosen]
        total = int(lens.sum())
        block_starts = np.cumsum(lens) - lens
        within = np.arange(total) - np.repeat(block_starts, lens)
        rows = sorted_rows[np.repeat(starts[chosen], lens) + within]
        values[r] = np.asarray(statistic_fn(rows), dtype=np.float64)

    finite = np.isfinite(values) if values.ndim == 1 else np.all(np.isfinite(values), axis=1)
    n_valid = int(finite.sum())
    scalar = point.ndim == 0

    def _emit(x: Optional[np.ndarray]) -> Any:
        if x is None:
            return None
        return float(x) if scalar else [float(v) for v in np.asarray(x)]

    if n_valid == 0:
        ci_low = ci_high = sd = mean = None
        status, reason = "NOT_ESTIMABLE", "statistic non-finite in every replicate"
    else:
        valid = values[finite]
        lo, hi = np.percentile(valid, percentiles, axis=0)
        ci_low, ci_high = _emit(lo), _emit(hi)
        sd = _emit(valid.std(axis=0, ddof=1)) if n_valid > 1 else None
        mean = _emit(valid.mean(axis=0))
        status, reason = "PASS", None
    return {
        "point_estimate": _emit(point) if np.all(np.isfinite(point)) else None,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": "percentile",
        "percentiles": [float(p) for p in percentiles],
        "bootstrap_mean": mean,
        "bootstrap_sd": sd,
        "n_rows": n_rows,
        "n_groups": n_groups,
        "n_replicates": n_replicates,
        "n_nonfinite_replicates": int(n_replicates - n_valid),
        "seed": int(seed),
        "resampling_unit": "independent group (patient / connected component)",
        "status": status,
        "reason": reason,
    }


def holm_adjust(pvalues: ArrayLike) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control), original order.

    With ``m`` ordered p-values ``p_(1) <= ... <= p_(m)`` the adjusted value is
    ``min(1, max_{j <= i} (m - j + 1) p_(j))``.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    if p.size == 0:
        return p
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must be finite and within [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    factors = m - np.arange(m)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(factors * p[order]))
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


def mde_paired(
    sd_delta: Union[float, ArrayLike],
    n_groups: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Union[float, np.ndarray]:
    """Planning minimum detectable paired-mean difference.

    ``MDE = (z_{1-alpha/2} + z_{power}) * sd_delta / sqrt(n_groups)`` under a
    normal approximation; at ``alpha=0.05, power=0.80`` the multiplier is about
    ``1.96 + 0.84``.  A scalar ``sd_delta`` returns a float, an array returns an
    array.  These are arithmetic scenarios, not observed power.
    """
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise ValueError("alpha and power must lie in (0, 1)")
    if int(n_groups) < 1:
        raise ValueError("n_groups must be at least 1")
    sd = np.asarray(sd_delta, dtype=np.float64)
    if np.any(sd < 0):
        raise ValueError("sd_delta must be non-negative")
    z = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    mde = z * sd / math.sqrt(int(n_groups))
    return float(mde) if mde.ndim == 0 else mde
