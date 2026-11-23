"""Lindblad open-system noise fitting against 1-qubit Randomized Benchmarking data.

Connects the Phase 0 QuTiP Lindblad simulator (`qtwin.physics.lindblad`) directly to
measured RB survival curves P_0(m) = A * alpha^m + B:

1. Given (T1, T2) and an effective physical Clifford duration t_clifford (in us),
   we evolve cardinal single-qubit states through QuTiP's `mesolve` to obtain the
   Pauli Transfer Matrix diagonal traces (D_X, D_Y, D_Z) and exact depolarizing
   parameter:
       alpha_lindblad = (D_X + D_Y + D_Z) / 3
   which equals (exp(-t/T1) + 2*exp(-t/T2)) / 3 for pure T1/T2 Lindblad noise.
2. An optional residual depolarizing/coherent error term `residual_error >= 0`
   captures excess gate error beyond coherence limits:
       alpha_total = alpha_lindblad * (1 - 2 * residual_error),
       EPC = (1 - alpha_total) / 2.
3. `scipy.optimize.least_squares` fits physical parameters under strict physical
   bounds (T1 > 0, 0 < T2 <= 2 * T1) and estimates 1-sigma parameter uncertainties
   from the Jacobian covariance matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import qutip as qt
from scipy.optimize import least_squares

from qtwin.physics import (
    SingleQubitParams,
    excited_state,
    ground_state,
    plus_state,
    simulate_single_qubit,
)

FitMode = Literal["fit_t2", "fit_residual", "fit_t1_t2"]


@dataclass(frozen=True)
class LindbladFitResult:
    """Fitted Lindblad model parameters, uncertainties, and goodness-of-fit metrics."""

    mode: str
    t1_us: float
    t1_us_err: float | None
    t2_us: float
    t2_us_err: float | None
    residual_error: float
    residual_error_err: float | None
    amplitude: float
    amplitude_err: float | None
    offset: float
    offset_err: float | None
    clifford_duration_us: float
    predicted_alpha: float
    predicted_epc: float
    rmse: float
    max_abs_err: float
    chi2: float
    dof: int
    reduced_chi2: float
    p0_pred: list[float]
    success: bool
    message: str


def _plus_i_state() -> qt.Qobj:
    """Return |+i> = (|0> + i|1>) / sqrt(2)."""
    return (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()


def lindblad_clifford_alpha(
    t1_us: float | None,
    t2_us: float | None,
    clifford_duration_us: float,
    residual_error: float = 0.0,
) -> float:
    """Compute the 1-qubit RB depolarizing parameter alpha via QuTiP Lindblad evolution.

    Evolves |0>, |1>, |+>, and |+i> for ``clifford_duration_us`` using
    ``simulate_single_qubit`` in the rotating frame (freq_mhz=0) and evaluates
    the Pauli Transfer Matrix diagonal average alpha_lindblad = (D_X + D_Y + D_Z) / 3.

    Args:
        t1_us: Energy relaxation time T1 in microseconds.
        t2_us: Coherence time T2 in microseconds (must satisfy T2 <= 2 * T1).
        clifford_duration_us: Effective physical duration of one Clifford gate in us.
        residual_error: Additional per-Clifford depolarizing/coherent error (>= 0).

    Returns:
        Total RB depolarizing parameter alpha in [0, 1].
    """
    if clifford_duration_us < 0.0:
        raise ValueError(f"clifford_duration_us must be non-negative, got {clifford_duration_us}.")

    params = SingleQubitParams(freq_mhz=0.0, t1_us=t1_us, t2_us=t2_us)
    times = np.array([0.0, clifford_duration_us], dtype=float)

    sz = qt.sigmaz()
    sx = qt.sigmax()
    sy = qt.sigmay()

    res_0 = simulate_single_qubit(params, ground_state(), times, e_ops=[sz], store_states=False)
    res_1 = simulate_single_qubit(params, excited_state(), times, e_ops=[sz], store_states=False)
    res_x = simulate_single_qubit(params, plus_state(), times, e_ops=[sx], store_states=False)
    res_y = simulate_single_qubit(params, _plus_i_state(), times, e_ops=[sy], store_states=False)

    d_z = 0.5 * (float(res_0.expect[0][-1]) - float(res_1.expect[0][-1]))
    d_x = float(res_x.expect[0][-1])
    d_y = float(res_y.expect[0][-1])

    alpha_lindblad = (d_x + d_y + d_z) / 3.0
    alpha_total = alpha_lindblad * (1.0 - 2.0 * residual_error)
    return float(np.clip(alpha_total, 0.0, 1.0))


def predict_rb_survival(
    lengths: np.ndarray,
    *,
    t1_us: float | None,
    t2_us: float | None,
    clifford_duration_us: float,
    amplitude: float = 0.5,
    offset: float = 0.5,
    residual_error: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Predict RB survival probabilities P_0(m) = A * alpha^m + B from Lindblad physics.

    Returns:
        Tuple ``(p0_pred, alpha, epc)`` where ``epc = (1 - alpha) / 2``.
    """
    m = np.asarray(lengths, dtype=float)
    alpha = lindblad_clifford_alpha(
        t1_us=t1_us,
        t2_us=t2_us,
        clifford_duration_us=clifford_duration_us,
        residual_error=residual_error,
    )
    p0_pred = amplitude * np.power(alpha, m) + offset
    epc = 0.5 * (1.0 - alpha)
    return p0_pred, alpha, float(epc)


def compute_goodness_of_fit(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    y_err: np.ndarray,
    n_params: int,
) -> dict[str, float | int]:
    """Compute RMSE, max absolute error, chi-squared, and reduced chi-squared."""
    diff = y_pred - y_obs
    sigma = np.maximum(np.asarray(y_err, dtype=float), 1e-9)
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    chi2 = float(np.sum((diff / sigma) ** 2))
    dof = max(int(len(y_obs) - n_params), 1)
    reduced_chi2 = float(chi2 / dof)
    return {
        "rmse": rmse,
        "max_abs_err": max_abs,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
    }


def _covariance_uncertainties(
    jac: np.ndarray,
    residuals: np.ndarray,
    dof: int,
) -> np.ndarray:
    """Compute 1-sigma parameter standard errors from the least-squares Jacobian."""
    jtj = jac.T @ jac
    cov_unscaled = np.linalg.pinv(jtj)
    s2 = float(np.sum(residuals**2) / max(dof, 1))
    diag = np.maximum(np.diag(cov_unscaled * s2), 0.0)
    return np.sqrt(diag)


def fit_lindblad_to_rb(
    lengths: np.ndarray,
    p0_mean: np.ndarray,
    p0_std: np.ndarray,
    clifford_duration_us: float,
    *,
    mode: FitMode = "fit_t2",
    t1_us_init: float = 80.0,
    t2_us_init: float = 50.0,
    residual_error_init: float = 1e-4,
    fit_spam: bool = True,
) -> LindbladFitResult:
    """Fit the QuTiP Lindblad RB model to measured survival probabilities P_0(m).

    Supported modes:
    - ``"fit_t2"``: Fixes T1 to ``t1_us_init`` and fits effective T2 in (0, 2*T1]
      (plus SPAM A, B if ``fit_spam=True``). Well-posed and identifiable from 1q RB.
    - ``"fit_residual"``: Fixes (T1, T2) to backend-reported ``(t1_us_init, t2_us_init)``
      and fits the excess residual gate error ``residual_error >= 0`` (plus SPAM A, B).
    - ``"fit_t1_t2"``: Jointly fits T1 > 0 and ratio eta = T2 / (2*T1) in (0, 1]
      (strictly enforcing 0 < T2 <= 2*T1) with mild soft-prior stabilization on T1.

    Args:
        lengths: 1D array of Clifford sequence lengths m.
        p0_mean: Measured mean survival probabilities P_0(m).
        p0_std: 1-sigma uncertainties on P_0(m).
        clifford_duration_us: Effective physical duration of one Clifford gate in us.
        mode: Parameterization mode (``"fit_t2"``, ``"fit_residual"``, or ``"fit_t1_t2"``).
        t1_us_init: Initial (or fixed) T1 in microseconds.
        t2_us_init: Initial (or fixed) T2 in microseconds.
        residual_error_init: Initial residual error for ``"fit_residual"`` mode.
        fit_spam: If True, also fit SPAM amplitude A and offset B; else fix A=0.5, B=0.5.

    Returns:
        ``LindbladFitResult`` with fitted parameters, 1-sigma uncertainties, and metrics.
    """
    m = np.asarray(lengths, dtype=float)
    y = np.asarray(p0_mean, dtype=float)
    sigma = np.maximum(np.asarray(p0_std, dtype=float), 1e-6)

    if t1_us_init <= 0.0:
        raise ValueError(f"t1_us_init must be positive, got {t1_us_init}.")
    t2_clamped = float(np.clip(t2_us_init, 1e-2, 2.0 * t1_us_init * (1.0 - 1e-9)))

    # Reasonable SPAM initial guesses from endpoints
    b_init = 0.5
    a_init = float(np.clip(y[0] - b_init, 0.05, 0.55))

    if mode == "fit_t2":
        x0 = [t2_clamped] + ([a_init, b_init] if fit_spam else [])
        lb = [1e-2] + ([0.05, 0.2] if fit_spam else [])
        ub = [2.0 * t1_us_init] + ([0.6, 0.8] if fit_spam else [])

        def _res_t2(theta: np.ndarray) -> np.ndarray:
            t2_val = float(theta[0])
            amp = float(theta[1]) if fit_spam else 0.5
            off = float(theta[2]) if fit_spam else 0.5
            pred, _, _ = predict_rb_survival(
                m,
                t1_us=t1_us_init,
                t2_us=t2_val,
                clifford_duration_us=clifford_duration_us,
                amplitude=amp,
                offset=off,
                residual_error=0.0,
            )
            return (pred - y) / sigma

        opt = least_squares(_res_t2, x0=x0, bounds=(lb, ub), method="trf")
        dof = max(len(m) - len(x0), 1)
        errs = _covariance_uncertainties(opt.jac, opt.fun, dof)

        t1_fit, t1_err = float(t1_us_init), None
        t2_fit, t2_err = float(opt.x[0]), float(errs[0])
        res_fit, res_err = 0.0, None
        amp_fit = float(opt.x[1]) if fit_spam else 0.5
        amp_err = float(errs[1]) if fit_spam else None
        off_fit = float(opt.x[2]) if fit_spam else 0.5
        off_err = float(errs[2]) if fit_spam else None

    elif mode == "fit_residual":
        x0 = [max(residual_error_init, 1e-6)] + ([a_init, b_init] if fit_spam else [])
        lb = [0.0] + ([0.05, 0.2] if fit_spam else [])
        ub = [0.25] + ([0.6, 0.8] if fit_spam else [])

        def _res_eps(theta: np.ndarray) -> np.ndarray:
            eps = float(theta[0])
            amp = float(theta[1]) if fit_spam else 0.5
            off = float(theta[2]) if fit_spam else 0.5
            pred, _, _ = predict_rb_survival(
                m,
                t1_us=t1_us_init,
                t2_us=t2_clamped,
                clifford_duration_us=clifford_duration_us,
                amplitude=amp,
                offset=off,
                residual_error=eps,
            )
            return (pred - y) / sigma

        opt = least_squares(_res_eps, x0=x0, bounds=(lb, ub), method="trf")
        dof = max(len(m) - len(x0), 1)
        errs = _covariance_uncertainties(opt.jac, opt.fun, dof)

        t1_fit, t1_err = float(t1_us_init), None
        t2_fit, t2_err = float(t2_clamped), None
        res_fit, res_err = float(opt.x[0]), float(errs[0])
        amp_fit = float(opt.x[1]) if fit_spam else 0.5
        amp_err = float(errs[1]) if fit_spam else None
        off_fit = float(opt.x[2]) if fit_spam else 0.5
        off_err = float(errs[2]) if fit_spam else None

    elif mode == "fit_t1_t2":
        eta_init = float(np.clip(t2_clamped / (2.0 * t1_us_init), 0.05, 0.99))
        x0 = [t1_us_init, eta_init] + ([a_init, b_init] if fit_spam else [])
        lb = [1e-1, 1e-3] + ([0.05, 0.2] if fit_spam else [])
        ub = [1e4, 1.0] + ([0.6, 0.8] if fit_spam else [])

        def _res_joint(theta: np.ndarray) -> np.ndarray:
            t1_val = float(theta[0])
            eta_val = float(theta[1])
            t2_val = 2.0 * t1_val * eta_val
            amp = float(theta[2]) if fit_spam else 0.5
            off = float(theta[3]) if fit_spam else 0.5
            pred, _, _ = predict_rb_survival(
                m,
                t1_us=t1_val,
                t2_us=t2_val,
                clifford_duration_us=clifford_duration_us,
                amplitude=amp,
                offset=off,
                residual_error=0.0,
            )
            # Soft 20% prior on T1 to break T1/T2 degeneracy in 1q RB
            prior_t1 = (t1_val - t1_us_init) / (0.2 * t1_us_init)
            return np.concatenate([(pred - y) / sigma, [prior_t1]])

        opt = least_squares(_res_joint, x0=x0, bounds=(lb, ub), method="trf")
        dof = max(len(m) - len(x0), 1)
        errs = _covariance_uncertainties(opt.jac, opt.fun, dof)

        t1_fit = float(opt.x[0])
        eta_fit = float(opt.x[1])
        t2_fit = 2.0 * t1_fit * eta_fit
        t1_err = float(errs[0])
        # First-order error propagation for T2 = 2 * T1 * eta
        t2_err = float(2.0 * np.hypot(eta_fit * errs[0], t1_fit * errs[1]))
        res_fit, res_err = 0.0, None
        amp_fit = float(opt.x[2]) if fit_spam else 0.5
        amp_err = float(errs[2]) if fit_spam else None
        off_fit = float(opt.x[3]) if fit_spam else 0.5
        off_err = float(errs[3]) if fit_spam else None
    else:
        raise ValueError(f"Unsupported fit mode: {mode!r}")

    p0_pred, alpha_pred, epc_pred = predict_rb_survival(
        m,
        t1_us=t1_fit,
        t2_us=t2_fit,
        clifford_duration_us=clifford_duration_us,
        amplitude=amp_fit,
        offset=off_fit,
        residual_error=res_fit,
    )
    gof = compute_goodness_of_fit(y, p0_pred, sigma, n_params=len(x0))

    return LindbladFitResult(
        mode=mode,
        t1_us=t1_fit,
        t1_us_err=t1_err,
        t2_us=t2_fit,
        t2_us_err=t2_err,
        residual_error=res_fit,
        residual_error_err=res_err,
        amplitude=amp_fit,
        amplitude_err=amp_err,
        offset=off_fit,
        offset_err=off_err,
        clifford_duration_us=clifford_duration_us,
        predicted_alpha=alpha_pred,
        predicted_epc=epc_pred,
        rmse=float(gof["rmse"]),
        max_abs_err=float(gof["max_abs_err"]),
        chi2=float(gof["chi2"]),
        dof=int(gof["dof"]),
        reduced_chi2=float(gof["reduced_chi2"]),
        p0_pred=[float(v) for v in p0_pred],
        success=bool(opt.success),
        message=str(opt.message),
    )


def fit_rb_json(
    json_path: Path | str,
    *,
    mode: FitMode = "fit_t2",
) -> dict[str, Any]:
    """Load an RB result JSON, compute reported-calibration baseline, and fit Lindblad model."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    lengths = np.asarray(data["lengths"], dtype=float)
    p0_mean = np.asarray(data["p0_mean"], dtype=float)
    p0_std = np.asarray(data["p0_std"], dtype=float)

    cal = data.get("calibration", {})
    t1_rep = float(cal["t1_us"]) if cal.get("t1_us") is not None else 80.0
    t2_rep = float(cal["t2_us"]) if cal.get("t2_us") is not None else 50.0
    sx_ns = float(cal["sx_duration_ns"]) if cal.get("sx_duration_ns") is not None else 35.0
    avg_gates = float(data.get("avg_physical_gates_per_clifford", 1.0))
    clifford_duration_us = (sx_ns * 1e-3) * avg_gates

    # Fit Lindblad model to measured survival curve
    fit_res = fit_lindblad_to_rb(
        lengths,
        p0_mean,
        p0_std,
        clifford_duration_us=clifford_duration_us,
        mode=mode,
        t1_us_init=t1_rep,
        t2_us_init=t2_rep,
        fit_spam=True,
    )

    # Backend-reported T1/T2 coherence-limited baseline (using fitted SPAM A, B for fair decay comparison)
    baseline_pred, baseline_alpha, baseline_epc = predict_rb_survival(
        lengths,
        t1_us=t1_rep,
        t2_us=min(t2_rep, 2.0 * t1_rep),
        clifford_duration_us=clifford_duration_us,
        amplitude=fit_res.amplitude,
        offset=fit_res.offset,
        residual_error=0.0,
    )
    baseline_gof = compute_goodness_of_fit(p0_mean, baseline_pred, p0_std, n_params=2)

    return {
        "raw": data,
        "clifford_duration_us": clifford_duration_us,
        "baseline": {
            "t1_us": t1_rep,
            "t2_us": t2_rep,
            "predicted_alpha": baseline_alpha,
            "predicted_epc": baseline_epc,
            "p0_pred": [float(v) for v in baseline_pred],
            **baseline_gof,
        },
        "fit": fit_res,
    }
