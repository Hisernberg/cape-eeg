"""Core distributional, hard-label, calibration, disagreement and referral metrics.

Everything here is pure numpy on float64 (scipy is used only for the Spearman
rank correlation, scikit-learn only for ranking-based AUROC / average
precision).  The conventions follow ``docs/04_Evaluation_and_Statistics.md``
and section 7 of ``docs/01_Research_Master_Plan.md``:

* the class/label order is fixed (:data:`LABELS`, :data:`CLASS_NAMES`);
* soft targets are ``q = v / v.sum(axis=1)`` and are **never** smoothed;
* predictions are clipped below at ``eps = 1e-7`` and renormalized exactly once
  before any logarithm is taken (:func:`clip_and_renormalize`);
* natural logarithms everywhere; ``KL(q || p)`` is target-to-prediction and
  terms with ``q_k = 0`` contribute exactly zero;
* a hard label exists only where exactly one class attains the maximum vote
  count; ties are excluded from hard-label metrics only and never broken by
  column order;
* observed pairwise disagreement is undefined (NaN, ``NOT_ESTIMABLE``) when a
  row has a single vote and is never clipped to the population maximum 5/6.

No function here reads data files or prints; each returns arrays or plain
dictionaries that can be serialised into the score manifest.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "LABELS",
    "CLASS_NAMES",
    "N_CLASSES",
    "DEFAULT_EPS",
    "DEFAULT_COVERAGE_GRID",
    "VOTE_COUNT_STRATA",
    "STATUS_PASS",
    "STATUS_NOT_ESTIMABLE",
    "METRIC_RECORD_KEYS",
    "SCORING_CONVENTION",
    "votes_to_targets",
    "clip_and_renormalize",
    "kl_rows",
    "soft_cross_entropy_rows",
    "soft_squared_error_rows",
    "expected_brier_rows",
    "patient_mean",
    "patient_weighted_mean",
    "unique_majority_labels",
    "hard_metrics",
    "soft_top_label_ece",
    "majority_label_ece",
    "pairwise_disagreement",
    "disagreement_metrics",
    "predictive_entropy",
    "one_minus_max_prob",
    "risk_coverage",
    "threshold_for_coverage",
    "apply_threshold",
    "referral_summary",
    "jensen_shannon_distance_rows",
    "metric_record",
]

# --------------------------------------------------------------------------- #
# Fixed conventions
# --------------------------------------------------------------------------- #

LABELS: List[str] = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
CLASS_NAMES: List[str] = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
N_CLASSES: int = len(LABELS)
DEFAULT_EPS: float = 1e-7
DEFAULT_COVERAGE_GRID: np.ndarray = np.linspace(0.5, 1.0, 51)

#: Prespecified vote-count strata ``(name, n_min, n_max)`` (inclusive bounds).
VOTE_COUNT_STRATA: Tuple[Tuple[str, int, float], ...] = (
    ("n=1", 1, 1),
    ("n=2-4", 2, 4),
    ("n=5-9", 5, 9),
    ("n>=10", 10, math.inf),
)

STATUS_PASS: str = "PASS"
STATUS_NOT_ESTIMABLE: str = "NOT_ESTIMABLE"

METRIC_RECORD_KEYS: Tuple[str, ...] = (
    "run_id",
    "method_id",
    "seed",
    "split_hash",
    "preprocess_hash",
    "protocol_hash",
    "prediction_hash",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n_rows",
    "n_patients",
    "n_components",
    "status",
    "reason",
)

#: Scoring convention recorded in the score manifest.
SCORING_CONVENTION: Dict[str, Any] = {
    "label_order": list(LABELS),
    "logarithm": "natural",
    "kl_direction": "target_to_prediction",
    "prediction_epsilon": DEFAULT_EPS,
    "prediction_clipping": "clip below at epsilon, renormalize once",
    "target_smoothing": "none",
    "hard_label_rule": "exactly one class attains the maximum vote count",
    "hard_label_ties": "excluded from hard-label metrics only",
    "disagreement_n1": "NOT_ESTIMABLE (NaN), never zero",
}

ArrayLike = Any


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _as_2d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_CLASSES:
        raise ValueError(f"{name} must have shape [N, {N_CLASSES}], got {arr.shape}")
    return arr


def _as_1d_float(x: ArrayLike, name: str, n: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")
    return arr


def _as_group_array(groups: ArrayLike, n: int) -> np.ndarray:
    arr = np.asarray(groups)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"groups must be a one-dimensional array of length {n}, got shape {arr.shape}")
    return arr


def _validate_votes(v: ArrayLike) -> np.ndarray:
    """Return votes as an integer-valued float64 array [N, 6]; raise on invalid rows."""
    arr = np.asarray(v)
    if arr.ndim != 2 or arr.shape[1] != N_CLASSES:
        raise ValueError(f"votes must have shape [N, {N_CLASSES}], got {arr.shape}")
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"votes must be numeric, got dtype {arr.dtype}")
    votes = arr.astype(np.float64)
    if not np.all(np.isfinite(votes)):
        raise ValueError("votes contain non-finite values")
    if np.any(votes < 0):
        raise ValueError("votes contain negative counts")
    if np.any(votes != np.floor(votes)):
        raise ValueError("votes contain non-integer counts")
    if np.any(votes.sum(axis=1) <= 0):
        raise ValueError("every row must have a positive total vote count (n_i = 0 is rejected)")
    return votes


def _validate_targets(q: ArrayLike) -> np.ndarray:
    targets = _as_2d_float(q, "q")
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets contain non-finite values")
    if np.any(targets < 0):
        raise ValueError("targets contain negative values")
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("every target row must sum to one (use votes_to_targets)")
    return targets


def _check_same_rows(a: np.ndarray, b: np.ndarray, name_a: str = "q", name_b: str = "p") -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"{name_a} has {a.shape[0]} rows but {name_b} has {b.shape[0]}")


def _nan_to_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    value = float(x)
    return None if math.isnan(value) else value


def _list_or_none(values: np.ndarray) -> List[Optional[float]]:
    return [_nan_to_none(x) for x in np.asarray(values, dtype=np.float64)]


def _nanmean_or_none(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    return float(finite.mean()) if finite.size else None


def _to_python(value: Any) -> Any:
    """Convert numpy scalars to Python scalars and NaN to None for manifest records."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# --------------------------------------------------------------------------- #
# Targets and predictions
# --------------------------------------------------------------------------- #


def votes_to_targets(v: ArrayLike) -> np.ndarray:
    """Soft targets ``q_ik = v_ik / n_i`` with ``n_i = sum_k v_ik``.

    Raises ``ValueError`` for a zero-sum row, negative or non-integer vote
    counts, and ``TypeError`` for non-numeric input.  No smoothing is applied.
    """
    votes = _validate_votes(v)
    return votes / votes.sum(axis=1, keepdims=True)


def clip_and_renormalize(p: ArrayLike, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Numerical scoring convention: clip ``p`` below at ``eps``, renormalize once.

    Raises ``ValueError`` if any entry is non-finite or negative, or if a row
    has a non-positive sum.  Returns a float64 array whose rows sum to one.
    """
    probs = _as_2d_float(p, "p")
    if not (0.0 < eps < 1.0 / N_CLASSES):
        raise ValueError(f"eps must lie in (0, 1/{N_CLASSES}), got {eps}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("predictions contain non-finite values")
    if np.any(probs < 0):
        raise ValueError("predictions contain negative values")
    if np.any(probs.sum(axis=1) <= 0):
        raise ValueError("every prediction row must have a positive sum")
    clipped = np.maximum(probs, eps)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _prepare_pair(q: ArrayLike, p: ArrayLike, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    targets = _validate_targets(q)
    probs = clip_and_renormalize(p, eps)
    _check_same_rows(targets, probs)
    return targets, probs


# --------------------------------------------------------------------------- #
# Soft (distributional) row metrics
# --------------------------------------------------------------------------- #


def kl_rows(q: ArrayLike, p: ArrayLike, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Primary row loss ``KL_i = sum_{k: q_ik > 0} q_ik [log q_ik - log p_ik]``.

    Natural logarithms; ``p`` is clipped/renormalized by
    :func:`clip_and_renormalize`; terms with ``q_ik = 0`` contribute exactly
    zero (``q`` is not smoothed).  Lower is better.  Returns shape ``[N]``.
    """
    targets, probs = _prepare_pair(q, p, eps)
    positive = targets > 0
    safe_q = np.where(positive, targets, 1.0)
    terms = np.where(positive, targets * (np.log(safe_q) - np.log(probs)), 0.0)
    return terms.sum(axis=1)


def soft_cross_entropy_rows(q: ArrayLike, p: ArrayLike, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Soft cross-entropy ``CE_i = -sum_k q_ik log p_ik`` (equals KL_i + H(q_i)).

    Uses the same clip/renormalize convention as :func:`kl_rows`.  Returns ``[N]``.
    """
    targets, probs = _prepare_pair(q, p, eps)
    return -(targets * np.log(probs)).sum(axis=1)


def soft_squared_error_rows(q: ArrayLike, p: ArrayLike, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Soft-target squared error ``sum_k (p_ik - q_ik)^2`` (label-distribution fidelity).

    This is not the conventional hard-label Brier score.  Returns ``[N]``.
    """
    targets, probs = _prepare_pair(q, p, eps)
    return ((probs - targets) ** 2).sum(axis=1)


def expected_brier_rows(q: ArrayLike, p: ArrayLike, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Expected categorical Brier score ``1 - 2 sum_k p_ik q_ik + sum_k p_ik^2``.

    Expected multi-class Brier score of ``p`` against a categorical label drawn
    from the expert-vote distribution ``q``.  Returns ``[N]``.
    """
    targets, probs = _prepare_pair(q, p, eps)
    return 1.0 - 2.0 * (probs * targets).sum(axis=1) + (probs ** 2).sum(axis=1)


# --------------------------------------------------------------------------- #
# Patient / group aggregation
# --------------------------------------------------------------------------- #


def patient_mean(values: ArrayLike, groups: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Per-group means of row-level values.

    ``values`` has shape ``[N]`` or ``[N, S]`` (for example one column per
    seed); ``groups`` is a length-``N`` array of hashable group identifiers
    (patients or connected components).  Returns ``(unique_groups,
    per_group_means)`` where ``unique_groups`` is sorted and
    ``per_group_means`` has shape ``[G]`` or ``[G, S]``.  NaN values propagate
    into their group's mean; mask them before calling if that is not wanted.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim not in (1, 2):
        raise ValueError(f"values must have shape [N] or [N, S], got {vals.shape}")
    n = vals.shape[0]
    grp = _as_group_array(groups, n)
    if n == 0:
        raise ValueError("values must contain at least one row")
    uniq, inverse = np.unique(grp, return_inverse=True)
    counts = np.bincount(inverse, minlength=uniq.size).astype(np.float64)
    if vals.ndim == 1:
        sums = np.bincount(inverse, weights=vals, minlength=uniq.size)
        means = sums / counts
    else:
        sums = np.zeros((uniq.size, vals.shape[1]), dtype=np.float64)
        np.add.at(sums, inverse, vals)
        means = sums / counts[:, None]
    return uniq, means


def patient_weighted_mean(values: ArrayLike, groups: ArrayLike) -> float:
    """Mean over groups of per-group means, every group weighted equally.

    For ``values`` of shape ``[N, S]`` this is ``mean_g mean_s`` of the
    per-group means (the primary estimand ``K_m = mean_a mean_s K_a,m,s``).
    """
    _, means = patient_mean(values, groups)
    return float(np.mean(means))


# --------------------------------------------------------------------------- #
# Hard labels
# --------------------------------------------------------------------------- #


def unique_majority_labels(v: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Hard label per row where exactly one class attains the maximum vote count.

    Returns ``(labels, tie_mask)``: ``labels`` is an int array ``[N]`` with the
    class index, or ``-1`` where two or more classes tie for the maximum;
    ``tie_mask`` is the boolean ``[N]`` tie indicator.  Ties are never broken
    by column order.
    """
    votes = _validate_votes(v)
    row_max = votes.max(axis=1, keepdims=True)
    n_at_max = (votes == row_max).sum(axis=1)
    tie_mask = n_at_max > 1
    labels = np.where(tie_mask, -1, votes.argmax(axis=1)).astype(np.int64)
    return labels, tie_mask


def _multiclass_mcc(cm: np.ndarray) -> Optional[float]:
    """Gorodkin's multi-class Matthews correlation from a confusion matrix; None if undefined."""
    cm = cm.astype(np.float64)
    s = cm.sum()
    c = np.trace(cm)
    t = cm.sum(axis=1)
    pr = cm.sum(axis=0)
    cov_ytyp = c * s - (t * pr).sum()
    cov_ypyp = s * s - (pr * pr).sum()
    cov_ytyt = s * s - (t * t).sum()
    denom = cov_ytyt * cov_ypyp
    if denom <= 0:
        return None
    return float(cov_ytyp / math.sqrt(denom))


def _cohen_kappa(cm: np.ndarray) -> Optional[float]:
    """Cohen's kappa ``(p_o - p_e) / (1 - p_e)``; None when ``p_e = 1``."""
    cm = cm.astype(np.float64)
    s = cm.sum()
    p_o = np.trace(cm) / s
    p_e = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / (s * s)
    if p_e >= 1.0:
        return None
    return float((p_o - p_e) / (1.0 - p_e))


def hard_metrics(v: ArrayLike, p: ArrayLike, eps: float = DEFAULT_EPS) -> Dict[str, Any]:
    """Hard-label discrimination metrics on unique-majority rows only.

    Rows whose maximum vote count is tied are excluded (and counted in
    ``n_ties_excluded``).  The predicted label is ``argmax_k p_ik``.  Per-class
    quantities are ``None`` where undefined: AUROC / AUPRC need at least one
    positive and one negative row; recall needs support; precision needs a
    predicted positive; F1 needs ``2TP + FP + FN > 0``.  Macro averages are
    taken over the defined classes only and ``n_classes_defined`` records how
    many contributed.  Balanced accuracy is the mean recall over classes with
    support.  Confusion matrices use rows = true class, columns = predicted
    class, in the fixed :data:`CLASS_NAMES` order.
    """
    votes = _validate_votes(v)
    probs = clip_and_renormalize(p, eps)
    _check_same_rows(votes, probs, "v", "p")
    labels, tie_mask = unique_majority_labels(votes)
    keep = ~tie_mask
    n_total = int(votes.shape[0])
    n_eval = int(keep.sum())
    n_ties = int(tie_mask.sum())
    result: Dict[str, Any] = {
        "class_names": list(CLASS_NAMES),
        "n_rows_total": n_total,
        "n_rows_evaluated": n_eval,
        "n_ties_excluded": n_ties,
        "prediction_rule": "argmax_k p_ik",
        "status": STATUS_PASS,
        "reason": None,
    }
    if n_eval == 0:
        result.update(
            {
                "accuracy": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "macro_precision": None,
                "macro_recall": None,
                "mcc": None,
                "cohen_kappa": None,
                "macro_auroc": None,
                "macro_auprc": None,
                "n_classes_defined": {"auroc": 0, "auprc": 0, "f1": 0, "recall": 0, "precision": 0},
                "per_class": None,
                "confusion_matrix": None,
                "confusion_matrix_row_normalized": None,
                "status": STATUS_NOT_ESTIMABLE,
                "reason": "no unique-majority rows",
            }
        )
        return result

    y_true = labels[keep]
    scores = probs[keep]
    y_pred = scores.argmax(axis=1)

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    fp = predicted - tp
    fn = support - tp
    tn = n_eval - tp - fp - fn

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        f1_den = 2.0 * tp + fp + fn
        f1 = np.where(f1_den > 0, 2.0 * tp / f1_den, np.nan)
        specificity = np.where(fp + tn > 0, tn / (fp + tn), np.nan)
        row_norm = np.where(support[:, None] > 0, cm / np.maximum(support[:, None], 1.0), np.nan)

    auroc = np.full(N_CLASSES, np.nan)
    auprc = np.full(N_CLASSES, np.nan)
    for k in range(N_CLASSES):
        positive = y_true == k
        n_pos = int(positive.sum())
        if n_pos == 0 or n_pos == n_eval:
            continue
        auroc[k] = roc_auc_score(positive.astype(np.int64), scores[:, k])
        auprc[k] = average_precision_score(positive.astype(np.int64), scores[:, k])

    result.update(
        {
            "accuracy": float(tp.sum() / n_eval),
            "balanced_accuracy": _nanmean_or_none(recall),
            "macro_f1": _nanmean_or_none(f1),
            "macro_precision": _nanmean_or_none(precision),
            "macro_recall": _nanmean_or_none(recall),
            "mcc": _multiclass_mcc(cm),
            "cohen_kappa": _cohen_kappa(cm),
            "macro_auroc": _nanmean_or_none(auroc),
            "macro_auprc": _nanmean_or_none(auprc),
            "n_classes_defined": {
                "auroc": int(np.sum(~np.isnan(auroc))),
                "auprc": int(np.sum(~np.isnan(auprc))),
                "f1": int(np.sum(~np.isnan(f1))),
                "recall": int(np.sum(~np.isnan(recall))),
                "precision": int(np.sum(~np.isnan(precision))),
            },
            "per_class": {
                "auroc": _list_or_none(auroc),
                "auprc": _list_or_none(auprc),
                "recall": _list_or_none(recall),
                "precision": _list_or_none(precision),
                "specificity": _list_or_none(specificity),
                "f1": _list_or_none(f1),
                "prevalence": [float(x) for x in support / n_eval],
                "n_positive": [int(x) for x in support],
                "n_negative": [int(n_eval - x) for x in support],
                "n_predicted": [int(x) for x in predicted],
            },
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_row_normalized": [_list_or_none(row) for row in row_norm],
        }
    )
    return result


# --------------------------------------------------------------------------- #
# Calibration (expected calibration error)
# --------------------------------------------------------------------------- #

_ECE_BINNING = "equal-count bins over predicted confidence (sorted, contiguous); independent of outcomes"
_ECE_MERGE_RULE = (
    "while some bin has fewer than min_groups_per_bin distinct groups and more than one bin "
    "remains: take the lowest-confidence such bin and merge it with the adjacent bin having "
    "fewer distinct groups (ties -> the lower-confidence neighbour; edge bins use their only "
    "neighbour)"
)


def _equal_count_bins(confidence: np.ndarray, n_bins: int) -> List[np.ndarray]:
    """Row-index chunks of (near) equal size in ascending confidence order."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    order = np.argsort(confidence, kind="stable")
    return [chunk for chunk in np.array_split(order, n_bins) if chunk.size > 0]


def _merge_sparse_bins(
    bins: List[np.ndarray], groups: np.ndarray, min_groups_per_bin: int
) -> Tuple[List[np.ndarray], int]:
    def n_groups(idx: np.ndarray) -> int:
        return int(np.unique(groups[idx]).size)

    n_merges = 0
    while len(bins) > 1:
        counts = [n_groups(b) for b in bins]
        sparse = [i for i, c in enumerate(counts) if c < min_groups_per_bin]
        if not sparse:
            break
        i = sparse[0]
        if i == 0:
            j = 1
        elif i == len(bins) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        lo, hi = min(i, j), max(i, j)
        merged = np.concatenate([bins[lo], bins[hi]])
        bins = bins[:lo] + [merged] + bins[hi + 1 :]
        n_merges += 1
    return bins, n_merges


def _ece_from_bins(
    bins: List[np.ndarray],
    confidence: np.ndarray,
    observed: np.ndarray,
    groups: Optional[np.ndarray],
) -> Dict[str, Any]:
    n = confidence.shape[0]
    lower, upper, mean_conf, mean_obs, count, n_groups, gaps = [], [], [], [], [], [], []
    for idx in bins:
        c = confidence[idx]
        o = observed[idx]
        lower.append(float(c.min()))
        upper.append(float(c.max()))
        mean_conf.append(float(c.mean()))
        mean_obs.append(float(o.mean()))
        count.append(int(idx.size))
        n_groups.append(int(np.unique(groups[idx]).size) if groups is not None else None)
        gaps.append(abs(mean_conf[-1] - mean_obs[-1]))
    weights = np.asarray(count, dtype=np.float64) / n
    gaps_arr = np.asarray(gaps, dtype=np.float64)
    return {
        "ece": float((weights * gaps_arr).sum()),
        "mce": float(gaps_arr.max()),
        "bin_lower": lower,
        "bin_upper": upper,
        "bin_mean_confidence": mean_conf,
        "bin_mean_observed": mean_obs,
        "bin_count": count,
        "bin_n_groups": n_groups,
        "n_bins_final": len(bins),
    }


def soft_top_label_ece(
    q: ArrayLike,
    p: ArrayLike,
    n_bins: int = 10,
    groups: Optional[ArrayLike] = None,
    min_groups_per_bin: int = 20,
    eps: float = DEFAULT_EPS,
) -> Dict[str, Any]:
    """Soft top-label expected calibration error.

    Confidence is ``max_k p_ik``; the observed quantity is the vote mass
    ``q_i,argmax`` that experts assigned to the predicted class.  Rows are
    placed into ``n_bins`` equal-count bins by confidence (binning depends on
    predictions only).  ``ECE = sum_b (n_b / N) |mean_conf_b - mean_obs_b|``.

    If ``groups`` is given, any bin with fewer than ``min_groups_per_bin``
    distinct groups is merged with a neighbour (rule recorded under
    ``merge_rule``).  The dictionary reports bin edges (lowest/highest
    confidence in each bin), per-bin mean confidence, mean observed vote mass,
    row count and distinct-group count, ``ece``, ``mce`` and merge statistics.
    This estimates consistency with expert-vote frequencies, not clinical truth.
    """
    targets, probs = _prepare_pair(q, p, eps)
    n = targets.shape[0]
    grp = _as_group_array(groups, n) if groups is not None else None
    predicted = probs.argmax(axis=1)
    confidence = probs[np.arange(n), predicted]
    observed = targets[np.arange(n), predicted]
    bins = _equal_count_bins(confidence, n_bins)
    n_merges = 0
    if grp is not None:
        bins, n_merges = _merge_sparse_bins(bins, grp, min_groups_per_bin)
    result = _ece_from_bins(bins, confidence, observed, grp)
    result.update(
        {
            "n_rows": int(n),
            "n_groups_total": int(np.unique(grp).size) if grp is not None else None,
            "n_bins_requested": int(n_bins),
            "n_merges": int(n_merges),
            "min_groups_per_bin": int(min_groups_per_bin) if grp is not None else None,
            "confidence_definition": "max_k p_ik",
            "observed_definition": "q_i,argmax_k p_ik (vote mass on the predicted class)",
            "binning": _ECE_BINNING,
            "merge_rule": _ECE_MERGE_RULE if grp is not None else "no merging (groups not supplied)",
        }
    )
    return result


def majority_label_ece(
    v: ArrayLike, p: ArrayLike, n_bins: int = 10, eps: float = DEFAULT_EPS
) -> Dict[str, Any]:
    """Standard top-label ECE on unique-majority rows with equal-count bins.

    Confidence is ``max_k p_ik`` and the observed quantity is the indicator
    that the predicted class equals the unique-majority label.  Tied rows are
    excluded and counted in ``n_ties_excluded``.
    """
    votes = _validate_votes(v)
    probs = clip_and_renormalize(p, eps)
    _check_same_rows(votes, probs, "v", "p")
    labels, tie_mask = unique_majority_labels(votes)
    keep = ~tie_mask
    n_eval = int(keep.sum())
    base = {
        "n_rows_total": int(votes.shape[0]),
        "n_rows_evaluated": n_eval,
        "n_ties_excluded": int(tie_mask.sum()),
        "n_bins_requested": int(n_bins),
        "confidence_definition": "max_k p_ik",
        "observed_definition": "1[argmax_k p_ik == unique majority label]",
        "binning": _ECE_BINNING,
        "status": STATUS_PASS,
        "reason": None,
    }
    if n_eval == 0:
        base.update({"ece": None, "mce": None, "n_bins_final": 0, "status": STATUS_NOT_ESTIMABLE,
                     "reason": "no unique-majority rows"})
        return base
    scores = probs[keep]
    predicted = scores.argmax(axis=1)
    confidence = scores[np.arange(n_eval), predicted]
    observed = (predicted == labels[keep]).astype(np.float64)
    bins = _equal_count_bins(confidence, n_bins)
    result = _ece_from_bins(bins, confidence, observed, None)
    result.update(base)
    return result


# --------------------------------------------------------------------------- #
# Expert disagreement
# --------------------------------------------------------------------------- #


def pairwise_disagreement(v: ArrayLike) -> np.ndarray:
    """Observed pairwise disagreement ``d_i = 1 - sum_k v_ik (v_ik - 1) / [n_i (n_i - 1)]``.

    The fraction of distinct ordered annotator pairs that disagree.  Defined
    for ``n_i > 1`` only; rows with a single vote return NaN
    (``NOT_ESTIMABLE``), never zero.  Finite-sample values may equal exactly
    1.0 and are not clipped to the six-class population maximum 5/6.
    """
    votes = _validate_votes(v)
    n = votes.sum(axis=1)
    agree = (votes * (votes - 1.0)).sum(axis=1)
    denom = n * (n - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(n > 1, 1.0 - agree / np.where(denom > 0, denom, 1.0), np.nan)
    return d


def _spearman_or_none(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    if a.size < 2:
        return None, "fewer than two rows"
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None, "constant input (rank correlation undefined)"
    res = spearmanr(a, b)
    rho = getattr(res, "statistic", None)
    if rho is None:
        rho = res[0]
    rho = float(rho)
    if math.isnan(rho):
        return None, "rank correlation undefined"
    return rho, None


def _disagreement_block(
    name: str,
    n_min: float,
    n_max: float,
    mask: np.ndarray,
    d_hat: np.ndarray,
    d_obs: np.ndarray,
) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "stratum": name,
        "n_min": n_min,
        "n_max": None if math.isinf(n_max) else n_max,
        "n_rows": int(mask.sum()),
        "n_rows_estimable": 0,
        "mean_observed": None,
        "mean_predicted": None,
        "mse": None,
        "spearman": None,
        "status": STATUS_NOT_ESTIMABLE,
        "reason": None,
    }
    if n_max < 2:
        block["reason"] = "observed disagreement is undefined for single-vote rows (n=1)"
        block["mean_predicted"] = float(d_hat[mask].mean()) if mask.any() else None
        return block
    estimable = mask & ~np.isnan(d_obs)
    m = int(estimable.sum())
    block["n_rows_estimable"] = m
    if m == 0:
        block["reason"] = "no rows with more than one vote"
        return block
    dh = d_hat[estimable]
    do = d_obs[estimable]
    block["mean_observed"] = float(do.mean())
    block["mean_predicted"] = float(dh.mean())
    block["mse"] = float(np.mean((dh - do) ** 2))
    rho, reason = _spearman_or_none(dh, do)
    block["spearman"] = rho
    block["status"] = STATUS_PASS
    block["reason"] = reason if rho is None else None
    return block


def disagreement_metrics(d_hat: ArrayLike, v: ArrayLike) -> Dict[str, Any]:
    """MSE and Spearman correlation of predicted vs observed disagreement.

    Observed ``d`` comes from :func:`pairwise_disagreement`; rows with a single
    vote are ``NOT_ESTIMABLE`` and excluded from every estimate.  Results are
    reported overall (all rows with ``n > 1``) and within the prespecified
    vote-count strata :data:`VOTE_COUNT_STRATA` (``[1]``, ``[2, 4]``,
    ``[5, 9]``, ``[10, inf)``).
    """
    votes = _validate_votes(v)
    n = votes.sum(axis=1)
    pred = _as_1d_float(d_hat, "d_hat", votes.shape[0])
    if not np.all(np.isfinite(pred)):
        raise ValueError("d_hat contains non-finite values")
    obs = pairwise_disagreement(votes)
    overall = _disagreement_block("overall (n>1)", 2, math.inf, n > 1, pred, obs)
    strata = [
        _disagreement_block(name, lo, hi, (n >= lo) & (n <= hi), pred, obs)
        for name, lo, hi in VOTE_COUNT_STRATA
    ]
    return {
        "definition": "d = 1 - sum_k v_k (v_k - 1) / [n (n - 1)], n > 1",
        "n_rows_total": int(votes.shape[0]),
        "n_rows_not_estimable": int((n == 1).sum()),
        "overall": overall,
        "strata": strata,
    }


# --------------------------------------------------------------------------- #
# Referral scores and risk-coverage analysis
# --------------------------------------------------------------------------- #


def predictive_entropy(p: ArrayLike, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Shannon entropy ``H_i = -sum_k p_ik log p_ik`` (natural log) after clip/renormalize."""
    probs = clip_and_renormalize(p, eps)
    return -(probs * np.log(probs)).sum(axis=1)


def one_minus_max_prob(p: ArrayLike, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Referral score ``1 - max_k p_ik`` after clip/renormalize."""
    probs = clip_and_renormalize(p, eps)
    return 1.0 - probs.max(axis=1)


def _n_accepted(coverage: np.ndarray, n: int) -> np.ndarray:
    """Number of accepted rows for a coverage fraction: ``clip(round(c * N), 1, N)``."""
    return np.clip(np.round(coverage * n), 1, n).astype(np.int64)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def risk_coverage(
    score: ArrayLike,
    risk_rows: ArrayLike,
    coverage_grid: ArrayLike = DEFAULT_COVERAGE_GRID,
    groups: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """Selective risk as a function of acceptance coverage.

    ``score`` is a referral/uncertainty score (higher = refer); rows with the
    LOWER score are accepted first.  For each coverage ``c`` on the grid the
    ``round(c * N)`` lowest-score rows (at least one) are accepted and the mean
    ``risk_rows`` over accepted rows is reported (``risk``); with ``groups`` the
    patient-weighted accepted risk (``risk_patient_weighted``) is added.
    ``aurc`` is the trapezoid integral of ``risk`` over the supplied grid
    (default ``[0.5, 1.0]``) and ``aurc_normalized`` divides by the grid span.
    ``score_threshold`` holds the highest accepted score at each coverage.
    """
    s = _as_1d_float(score, "score")
    n = s.shape[0]
    r = _as_1d_float(risk_rows, "risk_rows", n)
    if n == 0:
        raise ValueError("score must contain at least one row")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(r)):
        raise ValueError("score and risk_rows must be finite")
    grid = _as_1d_float(coverage_grid, "coverage_grid")
    if grid.size == 0 or np.any(grid <= 0) or np.any(grid > 1) or np.any(np.diff(grid) <= 0):
        raise ValueError("coverage_grid must be strictly increasing within (0, 1]")
    grp = _as_group_array(groups, n) if groups is not None else None

    order = np.argsort(s, kind="stable")
    sorted_score = s[order]
    sorted_risk = r[order]
    cumulative = np.cumsum(sorted_risk)
    n_acc = _n_accepted(grid, n)
    risk = cumulative[n_acc - 1] / n_acc
    thresholds = sorted_score[n_acc - 1]
    risk_pw: Optional[List[float]] = None
    if grp is not None:
        sorted_groups = grp[order]
        risk_pw = [patient_weighted_mean(sorted_risk[:k], sorted_groups[:k]) for k in n_acc]

    aurc = _trapezoid(risk, grid)
    span = float(grid[-1] - grid[0])
    return {
        "acceptance_rule": "accept the coverage fraction of rows with the lowest score",
        "n_rows": int(n),
        "n_groups": int(np.unique(grp).size) if grp is not None else None,
        "coverage_grid": [float(c) for c in grid],
        "n_accepted": [int(k) for k in n_acc],
        "actual_coverage": [float(k / n) for k in n_acc],
        "risk": [float(x) for x in risk],
        "risk_patient_weighted": risk_pw,
        "score_threshold": [float(t) for t in thresholds],
        "full_cohort_risk": float(r.mean()),
        "full_cohort_risk_patient_weighted": patient_weighted_mean(r, grp) if grp is not None else None,
        "aurc": aurc,
        "aurc_normalized": (aurc / span) if span > 0 else None,
        "integration_range": [float(grid[0]), float(grid[-1])],
        "integration_method": "trapezoid",
    }


def threshold_for_coverage(score: ArrayLike, coverage: float) -> float:
    """Score threshold accepting the ``coverage`` fraction of lowest-score rows.

    Intended for calibration-P: with ``k = clip(round(coverage * N), 1, N)`` the
    threshold is the ``k``-th smallest score, and :func:`apply_threshold`
    accepts rows with ``score <= threshold``.  Ties at the threshold may make
    the realised coverage exceed the planned value; report both.
    """
    s = _as_1d_float(score, "score")
    if s.size == 0:
        raise ValueError("score must contain at least one row")
    if not np.all(np.isfinite(s)):
        raise ValueError("score must be finite")
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must lie in (0, 1], got {coverage}")
    k = int(_n_accepted(np.asarray([coverage], dtype=np.float64), s.size)[0])
    return float(np.sort(s, kind="stable")[k - 1])


def apply_threshold(score: ArrayLike, threshold: float) -> np.ndarray:
    """Accepted-row mask ``score <= threshold``; NaN scores are never accepted (referred)."""
    s = _as_1d_float(score, "score")
    with np.errstate(invalid="ignore"):
        return s <= float(threshold)


def referral_summary(
    score: ArrayLike,
    threshold: float,
    risk_rows: ArrayLike,
    groups: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """Operating-point report for a frozen threshold applied to a cohort.

    Reports the realised acceptance coverage, the number and fraction of rows
    referred, the mean risk over accepted rows, over referred rows and over the
    whole cohort (row-weighted, and patient-weighted when ``groups`` is given).
    A low accepted risk after referring many rows is not improved overall
    classification, which is why the whole-cohort risk is reported alongside.
    """
    s = _as_1d_float(score, "score")
    n = s.shape[0]
    r = _as_1d_float(risk_rows, "risk_rows", n)
    if n == 0:
        raise ValueError("score must contain at least one row")
    grp = _as_group_array(groups, n) if groups is not None else None
    accepted = apply_threshold(s, threshold)
    n_acc = int(accepted.sum())
    n_ref = n - n_acc
    out: Dict[str, Any] = {
        "threshold": float(threshold),
        "acceptance_rule": "accept if score <= threshold; NaN scores are referred",
        "n_rows": int(n),
        "n_accepted": n_acc,
        "n_referred": n_ref,
        "actual_coverage": n_acc / n,
        "fraction_referred": n_ref / n,
        "accepted_risk": float(r[accepted].mean()) if n_acc else None,
        "referred_risk": float(r[~accepted].mean()) if n_ref else None,
        "whole_cohort_risk": float(r.mean()),
        "accepted_risk_patient_weighted": None,
        "whole_cohort_risk_patient_weighted": None,
        "n_groups": None,
        "n_groups_accepted": None,
        "status": STATUS_PASS if n_acc else STATUS_NOT_ESTIMABLE,
        "reason": None if n_acc else "no rows accepted at this threshold",
    }
    if grp is not None:
        out["n_groups"] = int(np.unique(grp).size)
        out["whole_cohort_risk_patient_weighted"] = patient_weighted_mean(r, grp)
        if n_acc:
            out["n_groups_accepted"] = int(np.unique(grp[accepted]).size)
            out["accepted_risk_patient_weighted"] = patient_weighted_mean(r[accepted], grp[accepted])
    return out


# --------------------------------------------------------------------------- #
# Prediction shift
# --------------------------------------------------------------------------- #


def jensen_shannon_distance_rows(p1: ArrayLike, p2: ArrayLike, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Row-wise Jensen-Shannon distance ``sqrt(0.5 KL(p1||m) + 0.5 KL(p2||m))``, ``m = (p1+p2)/2``.

    Natural logarithms, so the maximum is ``sqrt(log 2)``.  Both inputs pass
    through :func:`clip_and_renormalize`.  Returns ``[N]``.
    """
    a = clip_and_renormalize(p1, eps)
    b = clip_and_renormalize(p2, eps)
    _check_same_rows(a, b, "p1", "p2")
    m = 0.5 * (a + b)
    log_m = np.log(m)
    js = 0.5 * (a * (np.log(a) - log_m)).sum(axis=1) + 0.5 * (b * (np.log(b) - log_m)).sum(axis=1)
    return np.sqrt(np.maximum(js, 0.0))


# --------------------------------------------------------------------------- #
# Manifest records
# --------------------------------------------------------------------------- #


def metric_record(**kwargs: Any) -> Dict[str, Any]:
    """Build one score-manifest record with exactly :data:`METRIC_RECORD_KEYS`.

    Missing fields are ``None``; ``status`` defaults to ``"PASS"``.  Numpy
    scalars are converted to Python scalars and NaN values become ``None``
    (unrun values are null, never a number).  Unknown keys raise ``ValueError``.
    """
    unknown = set(kwargs) - set(METRIC_RECORD_KEYS)
    if unknown:
        raise ValueError(f"unknown metric record keys: {sorted(unknown)}")
    record: Dict[str, Any] = {key: None for key in METRIC_RECORD_KEYS}
    record["status"] = STATUS_PASS
    for key, value in kwargs.items():
        record[key] = _to_python(value)
    if record["status"] is None:
        record["status"] = STATUS_PASS
    return record
