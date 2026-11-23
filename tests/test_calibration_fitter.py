"""Unit tests for the Phase 1 Lindblad RB calibration fitter."""

from __future__ import annotations

import numpy as np
import pytest

from qtwin.calibration import (
    fit_lindblad_to_rb,
    lindblad_clifford_alpha,
    predict_rb_survival,
)


def test_lindblad_clifford_alpha_matches_exact_formula() -> None:
    """Verify QuTiP PTM alpha matches analytical (exp(-t/T1) + 2*exp(-t/T2)) / 3."""
    t1_us = 70.0
    t2_us = 45.0
    t_cliff_us = 0.05  # 50 ns

    alpha_num = lindblad_clifford_alpha(t1_us=t1_us, t2_us=t2_us, clifford_duration_us=t_cliff_us)
    alpha_exact = (np.exp(-t_cliff_us / t1_us) + 2.0 * np.exp(-t_cliff_us / t2_us)) / 3.0

    assert abs(alpha_num - alpha_exact) < 1e-7


def test_fit_t2_recovers_ground_truth_parameters() -> None:
    """Verify fit_t2 accurately recovers synthetic T2, SPAM, and EPC."""
    t1_true = 80.0
    t2_true = 42.0
    t_cliff_us = 0.05
    amp_true = 0.48
    off_true = 0.50

    lengths = np.array([1, 50, 150, 300, 600, 1000, 1500], dtype=float)
    p0_clean, _, epc_true = predict_rb_survival(
        lengths,
        t1_us=t1_true,
        t2_us=t2_true,
        clifford_duration_us=t_cliff_us,
        amplitude=amp_true,
        offset=off_true,
    )
    # Small deterministic noise
    rng = np.random.default_rng(123)
    sigma = np.full_like(p0_clean, 0.002)
    p0_noisy = p0_clean + rng.normal(0.0, 0.001, size=len(lengths))

    fit = fit_lindblad_to_rb(
        lengths,
        p0_noisy,
        sigma,
        clifford_duration_us=t_cliff_us,
        mode="fit_t2",
        t1_us_init=t1_true,
        t2_us_init=65.0,  # deliberately perturbed initial guess
        fit_spam=True,
    )

    assert fit.success
    assert 0.0 < fit.t2_us <= 2.0 * fit.t1_us
    assert abs(fit.t2_us - t2_true) / t2_true < 0.05
    assert abs(fit.predicted_epc - epc_true) / epc_true < 0.05
    assert fit.t2_us_err is not None and np.isfinite(fit.t2_us_err) and fit.t2_us_err > 0.0
    assert fit.rmse < 0.003


def test_fit_residual_recovers_excess_gate_error() -> None:
    """Verify fit_residual accurately recovers excess coherent/depolarizing gate error."""
    t1_rep = 60.0
    t2_rep = 50.0
    t_cliff_us = 0.035
    eps_true = 8.0e-4

    lengths = np.array([1, 25, 75, 150, 300, 500, 800], dtype=float)
    p0_clean, _, epc_true = predict_rb_survival(
        lengths,
        t1_us=t1_rep,
        t2_us=t2_rep,
        clifford_duration_us=t_cliff_us,
        amplitude=0.49,
        offset=0.50,
        residual_error=eps_true,
    )
    sigma = np.full_like(p0_clean, 0.003)

    fit = fit_lindblad_to_rb(
        lengths,
        p0_clean,
        sigma,
        clifford_duration_us=t_cliff_us,
        mode="fit_residual",
        t1_us_init=t1_rep,
        t2_us_init=t2_rep,
        fit_spam=True,
    )

    assert fit.success
    assert abs(fit.residual_error - eps_true) < 1e-5
    assert abs(fit.predicted_epc - epc_true) < 1e-5
    assert fit.residual_error_err is not None and np.isfinite(fit.residual_error_err)


def test_fit_t1_t2_enforces_physical_bound() -> None:
    """Ensure joint T1/T2 fitting always obeys 0 < T2 <= 2 * T1."""
    lengths = np.array([1, 100, 300, 600, 1000], dtype=float)
    p0, _, _ = predict_rb_survival(
        lengths,
        t1_us=50.0,
        t2_us=95.0,  # near 2*T1 boundary
        clifford_duration_us=0.04,
        amplitude=0.5,
        offset=0.5,
    )
    sigma = np.full_like(p0, 0.002)

    fit = fit_lindblad_to_rb(
        lengths,
        p0,
        sigma,
        clifford_duration_us=0.04,
        mode="fit_t1_t2",
        t1_us_init=55.0,
        t2_us_init=80.0,
        fit_spam=True,
    )

    assert fit.success
    assert fit.t1_us > 0.0
    assert 0.0 < fit.t2_us <= 2.0 * fit.t1_us * (1.0 + 1e-9)
    assert fit.t1_us_err is not None and fit.t2_us_err is not None


def test_invalid_mode_raises() -> None:
    """Reject unknown fitting modes cleanly."""
    with pytest.raises(ValueError, match="Unsupported fit mode"):
        fit_lindblad_to_rb(
            np.array([1, 10, 50]),
            np.array([0.99, 0.95, 0.80]),
            np.array([0.01, 0.01, 0.01]),
            clifford_duration_us=0.05,
            mode="invalid_mode",  # type: ignore[arg-type]
        )
