"""Deterministic synthetic tests for cape_eeg.evaluation.calibrate."""

import math

import numpy as np
import pytest

from cape_eeg import metrics as M
from cape_eeg.evaluation import calibrate as C


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _synthetic(t0, n=600, seed=1, n_groups=40, scale=1.5):
    """Return (miscalibrated p, calibrated soft target q, groups).

    ``p = softmax(z / t0)`` so that ``softmax(log p / T)`` recovers ``q =
    softmax(z)`` exactly at ``T = 1 / t0`` -- up to the 1e-7 clip of the
    scoring convention, which truncates ``log p`` below about -16 and therefore
    distorts recovery when ``scale / t0`` is large.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, scale, size=(n, 6))
    q = _softmax(z)
    p = _softmax(z / t0)
    groups = rng.integers(0, n_groups, size=n)
    return p, q, groups


def test_apply_temperature_identity_and_limits():
    p, _, _ = _synthetic(0.5)
    np.testing.assert_allclose(C.apply_temperature(p, 1.0), M.clip_and_renormalize(p), atol=1e-12)
    hot = C.apply_temperature(p, 1e6)
    np.testing.assert_allclose(hot, 1 / 6, atol=1e-4)
    cold = C.apply_temperature(p, 1e-3)
    assert np.array_equal(cold.argmax(axis=1), M.clip_and_renormalize(p).argmax(axis=1))
    assert cold.max(axis=1).mean() > 0.99 and cold.max(axis=1).min() > 0.5
    np.testing.assert_allclose(C.apply_temperature(p, 2.0).sum(axis=1), 1.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            C.apply_temperature(p, bad)


def test_fit_temperature_recovers_known_temperature():
    p, q, groups = _synthetic(0.5)
    out = C.fit_temperature(p, q, groups)
    assert math.isclose(out["temperature"], 2.0, abs_tol=2e-3)
    assert out["at_bound"] is False and out["bound_hit"] is None
    assert out["objective_after"] < out["objective_before"]
    assert out["kl_after"] < out["kl_before"] and out["kl_after"] < 1e-4
    assert out["n_rows"] == 600 and out["n_groups"] == 40
    assert out["objective"] == "patient_weighted_soft_cross_entropy"
    # The fitted objective is the patient-weighted CE at T.
    tempered = C.apply_temperature(p, out["temperature"])
    ref = M.patient_weighted_mean(M.soft_cross_entropy_rows(q, tempered), groups)
    assert math.isclose(out["objective_after"], ref, rel_tol=1e-9)
    # Underconfident inputs move the other way.
    p2, q2, g2 = _synthetic(2.0, seed=3)
    assert math.isclose(C.fit_temperature(p2, q2, g2)["temperature"], 0.5, abs_tol=2e-3)


def test_fit_temperature_is_patient_weighted():
    rng = np.random.default_rng(8)
    z = rng.normal(0.0, 1.5, size=(300, 6))
    q = _softmax(z)
    # One big group needs T=2, many small single-row groups need T=0.5.
    p = np.vstack([_softmax(z[:200] / 0.5), _softmax(z[200:] / 2.0)])
    groups = np.concatenate([np.zeros(200, dtype=int), np.arange(1, 101)])
    patient = C.fit_temperature(p, q, groups)["temperature"]
    rows = C.fit_temperature(p, q, np.arange(300))["temperature"]
    assert patient < rows  # 100 of 101 groups pull towards 0.5; 200 of 300 rows pull towards 2


def test_fit_temperature_reports_bounds():
    p_hi, q_hi, g_hi = _synthetic(0.1)  # optimum near 10 > 4
    upper = C.fit_temperature(p_hi, q_hi, g_hi)
    assert upper["at_bound"] is True and upper["bound_hit"] == "upper"
    assert math.isclose(upper["temperature"], 4.0, abs_tol=1e-3)
    assert math.isclose(upper["objective_after"], upper["objective_at_t_max"])
    p_lo, q_lo, g_lo = _synthetic(10.0)  # optimum near 0.1 < 0.25
    lower = C.fit_temperature(p_lo, q_lo, g_lo)
    assert lower["at_bound"] is True and lower["bound_hit"] == "lower"
    assert math.isclose(lower["temperature"], 0.25, abs_tol=1e-3)
    # With a modest logit scale the clip does not bite and wide bounds recover T = 10.
    p_w, q_w, g_w = _synthetic(0.1, scale=0.2)
    wide = C.fit_temperature(p_w, q_w, g_w, t_min=0.25, t_max=20.0)
    assert wide["at_bound"] is False and math.isclose(wide["temperature"], 10.0, abs_tol=0.05)
    with pytest.raises(ValueError):
        C.fit_temperature(p_hi, q_hi, g_hi, t_min=2.0, t_max=1.0)
    with pytest.raises(ValueError):
        C.fit_temperature(p_hi, q_hi, g_hi[:-1])


def test_temperature_effect_summary_reports_before_after():
    p, q, groups = _synthetic(0.5)
    out = C.temperature_effect_summary(p, q, groups, 2.0)
    for key in ("kl", "soft_squared_error", "expected_brier"):
        block = out[key]
        assert block["after_patient_weighted"] < block["before_patient_weighted"]
        assert block["after_row_mean"] < block["before_row_mean"]
    assert out["n_groups"] == 40 and out["temperature"] == 2.0


def test_fit_referral_thresholds_on_calibration_p():
    rng = np.random.default_rng(6)
    score = rng.uniform(size=1000)
    out = C.fit_referral_thresholds(score)
    assert out["coverages"] == [0.8, 0.9, 0.95] and out["n_rows"] == 1000
    thr = [r["threshold"] for r in out["records"]]
    assert thr[0] < thr[1] < thr[2]
    for rec in out["records"]:
        assert math.isclose(rec["calibration_coverage"], rec["planned_coverage"], abs_tol=1 / 1000)
        assert rec["n_accepted"] + rec["n_referred"] == 1000
        assert out["thresholds"][rec["planned_coverage"]] == rec["threshold"]
    # Applying the frozen threshold to a new cohort yields a nearby, not identical, coverage.
    new_score = rng.uniform(size=5000)
    accepted = M.apply_threshold(new_score, out["thresholds"][0.9])
    assert 0.85 < accepted.mean() < 0.95
    with pytest.raises(ValueError):
        C.fit_referral_thresholds(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        C.fit_referral_thresholds(score, coverages=(1.5,))
