"""Unit tests for the Phase 2 2-qubit ZZ crosstalk extraction engine."""

from __future__ import annotations

import numpy as np
import pytest

from qtwin.calibration import (
    fit_crosstalk_to_srb,
    lindblad_two_qubit_srb_alphas,
    predict_srb_curves,
)
from qtwin.physics import SingleQubitParams, TwoQubitParams


def test_two_qubit_lindblad_srb_alphas_match_analytical() -> None:
    """Verify QuTiP 2q mesolve SRB alphas match exact Gambetta/ZZ closed-form expressions."""
    t1_a, t2_a = 80.0, 55.0
    t1_b, t2_b = 70.0, 50.0
    f_zz = 0.30  # MHz
    t_cliff = 0.04  # 40 ns

    params = TwoQubitParams(
        qubit0=SingleQubitParams(freq_mhz=0.0, t1_us=t1_a, t2_us=t2_a),
        qubit1=SingleQubitParams(freq_mhz=0.0, t1_us=t1_b, t2_us=t2_b),
        zz_coupling_mhz=f_zz,
    )
    a_ia, a_ib, a_sa, a_sb = lindblad_two_qubit_srb_alphas(params, t_cliff)

    cos2_half = np.cos(np.pi * f_zz * t_cliff) ** 2
    exact_ia = (np.exp(-t_cliff / t1_a) + 2.0 * np.exp(-t_cliff / t2_a)) / 3.0
    exact_ib = (np.exp(-t_cliff / t1_b) + 2.0 * np.exp(-t_cliff / t2_b)) / 3.0
    exact_sa = (np.exp(-t_cliff / t1_a) + 2.0 * np.exp(-t_cliff / t2_a) * cos2_half) / 3.0
    exact_sb = (np.exp(-t_cliff / t1_b) + 2.0 * np.exp(-t_cliff / t2_b) * cos2_half) / 3.0

    assert abs(a_ia - exact_ia) < 1e-6
    assert abs(a_ib - exact_ib) < 1e-6
    assert abs(a_sa - exact_sa) < 1e-6
    assert abs(a_sb - exact_sb) < 1e-6
    assert a_sa < a_ia and a_sb < a_ib


def test_fit_crosstalk_recovers_synthetic_zz_coupling() -> None:
    """Verify fit_crosstalk_to_srb accurately recovers injected f_ZZ and local errors."""
    t1_a, t2_a = 80.0, 60.0
    t1_b, t2_b = 75.0, 55.0
    t_cliff = 0.04  # 40 ns
    zz_true = 0.35  # MHz
    eps_a_true = 3.0e-4
    eps_b_true = 2.0e-4

    lengths = np.array([1, 50, 150, 300, 600, 1000], dtype=float)
    synth = predict_srb_curves(
        lengths,
        t1_a_us=t1_a,
        t2_a_us=t2_a,
        t1_b_us=t1_b,
        t2_b_us=t2_b,
        clifford_duration_us=t_cliff,
        zz_coupling_mhz=zz_true,
        local_error_a=eps_a_true,
        local_error_b=eps_b_true,
        amp_a=0.48,
        off_a=0.50,
        amp_b=0.49,
        off_b=0.50,
    )

    rng = np.random.default_rng(7)
    sigma = np.full_like(lengths, 0.002)
    fit = fit_crosstalk_to_srb(
        lengths,
        synth["p0_iso_a"] + rng.normal(0.0, 0.0005, size=len(lengths)),
        sigma,
        synth["p0_iso_b"] + rng.normal(0.0, 0.0005, size=len(lengths)),
        sigma,
        synth["p0_sim_a"] + rng.normal(0.0, 0.0005, size=len(lengths)),
        sigma,
        synth["p0_sim_b"] + rng.normal(0.0, 0.0005, size=len(lengths)),
        sigma,
        t1_a_us=t1_a,
        t2_a_us=t2_a,
        t1_b_us=t1_b,
        t2_b_us=t2_b,
        clifford_duration_us=t_cliff,
        mode="fit_zz",
        zz_init_mhz=0.10,
    )

    assert fit.success
    assert abs(fit.zz_coupling_mhz - zz_true) / zz_true < 0.05
    assert fit.zz_coupling_mhz_err is not None and np.isfinite(fit.zz_coupling_mhz_err)
    assert abs(fit.local_error_a - eps_a_true) < 5e-5
    assert abs(fit.local_error_b - eps_b_true) < 5e-5
    assert fit.rmse_sim < 0.003


def test_zero_crosstalk_returns_near_zero_zz() -> None:
    """When Isolated and Simultaneous RB match, extracted f_ZZ should be near zero."""
    lengths = np.array([1, 50, 150, 300, 600], dtype=float)
    synth = predict_srb_curves(
        lengths,
        t1_a_us=80.0,
        t2_a_us=60.0,
        t1_b_us=80.0,
        t2_b_us=60.0,
        clifford_duration_us=0.04,
        zz_coupling_mhz=0.0,
        local_error_a=2e-4,
        local_error_b=2e-4,
    )
    sigma = np.full_like(lengths, 0.002)
    fit = fit_crosstalk_to_srb(
        lengths,
        synth["p0_iso_a"],
        sigma,
        synth["p0_iso_b"],
        sigma,
        synth["p0_sim_a"],
        sigma,
        synth["p0_sim_b"],
        sigma,
        t1_a_us=80.0,
        t2_a_us=60.0,
        t1_b_us=80.0,
        t2_b_us=60.0,
        clifford_duration_us=0.04,
        mode="fit_zz",
    )

    assert fit.success
    assert fit.zz_coupling_mhz < 0.02


def test_invalid_crosstalk_mode_raises() -> None:
    lengths = np.array([1, 10, 50], dtype=float)
    y = np.array([0.99, 0.95, 0.85])
    s = np.array([0.01, 0.01, 0.01])
    with pytest.raises(ValueError, match="Unsupported crosstalk fit mode"):
        fit_crosstalk_to_srb(
            lengths, y, s, y, s, y, s, y, s,
            t1_a_us=80.0, t2_a_us=60.0, t1_b_us=80.0, t2_b_us=60.0,
            clifford_duration_us=0.04,
            mode="bad_mode",  # type: ignore[arg-type]
        )
