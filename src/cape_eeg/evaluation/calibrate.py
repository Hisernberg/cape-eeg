"""Calibration-T temperature scaling and calibration-P referral thresholds.

Section 5 and 6 of ``docs/04_Evaluation_and_Statistics.md``: a single scalar
temperature ``p_T = softmax(log(p) / T)`` with ``T`` in ``[0.25, 4]`` is fitted
on calibration-T by minimising the patient-weighted soft cross-entropy (every
group weighted equally); referral-score thresholds for the planned acceptance
coverages are frozen on calibration-P without test labels.  No isotonic
regression or classwise temperatures are offered.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from cape_eeg import metrics as M

__all__ = [
    "DEFAULT_T_MIN",
    "DEFAULT_T_MAX",
    "DEFAULT_COVERAGES",
    "apply_temperature",
    "fit_temperature",
    "temperature_effect_summary",
    "fit_referral_thresholds",
]

DEFAULT_T_MIN: float = 0.25
DEFAULT_T_MAX: float = 4.0
DEFAULT_COVERAGES: Sequence[float] = (0.80, 0.90, 0.95)

ArrayLike = Any


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(p: ArrayLike, T: float, eps: float = M.DEFAULT_EPS) -> np.ndarray:
    """Temperature-scaled probabilities ``p_T = softmax(log(p) / T)``.

    ``p`` is first passed through :func:`cape_eeg.metrics.clip_and_renormalize`
    so ``log(p)`` is finite.  ``T`` must be a finite positive scalar; ``T = 1``
    returns the clipped/renormalized input unchanged.
    """
    temperature = float(T)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be a finite positive scalar, got {T}")
    log_p = np.log(M.clip_and_renormalize(p, eps))
    return _softmax_rows(log_p / temperature)


def _group_weights(groups: np.ndarray) -> np.ndarray:
    """Row weights ``1 / (G * n_rows_in_group)`` so the weighted sum is the patient-weighted mean."""
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    return 1.0 / (counts.size * counts[inverse])


def fit_temperature(
    p: ArrayLike,
    q: ArrayLike,
    groups: ArrayLike,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    eps: float = M.DEFAULT_EPS,
    xatol: float = 1e-6,
    bound_tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """Fit the scalar temperature on calibration-T by patient-weighted soft cross-entropy.

    Objective ``J(T) = mean_g mean_{i in g} CE(q_i, softmax(log p_i / T))``
    (each group's mean cross-entropy weighted equally), minimised over
    ``T in [t_min, t_max]`` with bounded Brent search
    (``scipy.optimize.minimize_scalar``).  The objective is unimodal in ``T``
    because it is convex in ``1/T``.  The result reports ``temperature``,
    whether the optimum sits at a bound (``at_bound`` / ``bound_hit``), the
    objective at ``T = 1`` (``objective_before``), at the optimum
    (``objective_after``) and at both bounds, plus patient-weighted KL before
    and after for convenience.
    """
    if not (0.0 < t_min < t_max) or not math.isfinite(t_max):
        raise ValueError("temperature bounds must satisfy 0 < t_min < t_max < inf")
    probs = M.clip_and_renormalize(p, eps)
    targets = M._validate_targets(q)
    M._check_same_rows(targets, probs)
    n_rows = int(probs.shape[0])
    grp = M._as_group_array(groups, n_rows)
    weights = _group_weights(grp)
    log_p = np.log(probs)

    def objective(T: float) -> float:
        tempered = _softmax_rows(log_p / float(T))
        return float(weights @ M.soft_cross_entropy_rows(targets, tempered, eps))

    res = minimize_scalar(objective, bounds=(t_min, t_max), method="bounded", options={"xatol": xatol})
    candidates = {float(res.x): float(res.fun), float(t_min): objective(t_min), float(t_max): objective(t_max)}
    temperature = min(candidates, key=lambda t: candidates[t])
    objective_after = candidates[temperature]
    objective_before = objective(1.0)

    bound_hit: Optional[str] = None
    if abs(temperature - t_min) <= bound_tolerance:
        bound_hit = "lower"
    elif abs(temperature - t_max) <= bound_tolerance:
        bound_hit = "upper"

    kl_before = M.patient_weighted_mean(M.kl_rows(targets, probs, eps), grp)
    kl_after = M.patient_weighted_mean(M.kl_rows(targets, _softmax_rows(log_p / temperature), eps), grp)
    return {
        "temperature": temperature,
        "at_bound": bound_hit is not None,
        "bound_hit": bound_hit,
        "t_min": float(t_min),
        "t_max": float(t_max),
        "bound_tolerance": float(bound_tolerance),
        "objective": "patient_weighted_soft_cross_entropy",
        "weighting": "each independent group weighted equally",
        "objective_before": objective_before,
        "objective_after": objective_after,
        "objective_improvement": objective_before - objective_after,
        "objective_at_t_min": candidates[float(t_min)],
        "objective_at_t_max": candidates[float(t_max)],
        "kl_before": kl_before,
        "kl_after": kl_after,
        "n_rows": n_rows,
        "n_groups": int(np.unique(grp).size),
        "optimizer": {
            "method": "scipy.optimize.minimize_scalar(bounded Brent)",
            "success": bool(getattr(res, "success", True)),
            "n_evaluations": int(getattr(res, "nfev", -1)),
            "xatol": float(xatol),
        },
        "transform": "p_T = softmax(log(p) / T)",
    }


def temperature_effect_summary(
    p: ArrayLike, q: ArrayLike, groups: ArrayLike, T: float, eps: float = M.DEFAULT_EPS
) -> Dict[str, Any]:
    """KL, soft squared error and expected Brier before/after a frozen temperature.

    Every metric is reported both patient-weighted (``mean_g mean_i``) and
    row-weighted (``mean_i``).
    """
    probs = M.clip_and_renormalize(p, eps)
    targets = M._validate_targets(q)
    M._check_same_rows(targets, probs)
    grp = M._as_group_array(groups, probs.shape[0])
    tempered = apply_temperature(probs, T, eps)
    metrics = {
        "kl": M.kl_rows,
        "soft_squared_error": M.soft_squared_error_rows,
        "expected_brier": M.expected_brier_rows,
    }
    out: Dict[str, Any] = {"temperature": float(T), "n_rows": int(probs.shape[0]), "n_groups": int(np.unique(grp).size)}
    for name, fn in metrics.items():
        before = fn(targets, probs, eps)
        after = fn(targets, tempered, eps)
        out[name] = {
            "before_patient_weighted": M.patient_weighted_mean(before, grp),
            "after_patient_weighted": M.patient_weighted_mean(after, grp),
            "before_row_mean": float(before.mean()),
            "after_row_mean": float(after.mean()),
        }
    return out


def fit_referral_thresholds(
    score: ArrayLike, coverages: Sequence[float] = DEFAULT_COVERAGES
) -> Dict[str, Any]:
    """Freeze referral-score thresholds on calibration-P for planned acceptance coverages.

    For each planned coverage the threshold is
    :func:`cape_eeg.metrics.threshold_for_coverage`; the realised calibration
    coverage with the rule ``accept if score <= threshold`` is reported next to
    the planned value (ties can make it larger).  Scores must be finite.
    """
    s = np.asarray(score, dtype=np.float64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("score must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(s)):
        raise ValueError("calibration-P scores must be finite")
    cov = [float(c) for c in coverages]
    if not cov or any(not (0.0 < c <= 1.0) for c in cov):
        raise ValueError("coverages must lie in (0, 1]")
    records = []
    thresholds: Dict[float, float] = {}
    for c in cov:
        thr = M.threshold_for_coverage(s, c)
        accepted = M.apply_threshold(s, thr)
        n_acc = int(accepted.sum())
        thresholds[c] = thr
        records.append(
            {
                "planned_coverage": c,
                "threshold": thr,
                "calibration_coverage": n_acc / s.size,
                "n_accepted": n_acc,
                "n_referred": int(s.size - n_acc),
            }
        )
    return {
        "acceptance_rule": "accept if score <= threshold (higher score = refer)",
        "n_rows": int(s.size),
        "coverages": cov,
        "thresholds": thresholds,
        "records": records,
    }
