"""Single-qubit Lindblad RB survival-curve fitter using QuTiP mesolve and scipy least_squares."""

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
    return (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()


def lindblad_clifford_alpha(
    t1_us: float | None,
    t2_us: float | None,
    clifford_duration_us: float,
    residual_error: float = 0.0,
) -> float:
    """Compute 1q RB depolarizing parameter alpha from QuTiP PTM diagonal traces."""
    if clifford_duration_us < 0.0:
        raise ValueError(f"clifford_duration_us must be non-negative, got {clifford_duration_us}.")

    params = SingleQubitParams(freq_mhz=0.0, t1_us=t1_us, t2_us=t2_us)
    times = np.array([0.0, clifford_duration_us], dtype=float)

    sz, sx, sy = qt.sigmaz(), qt.sigmax(), qt.sigmay()

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
    """Predict P_0(m) = A * alpha^m + B; returns (p0_pred, alpha, epc)."""
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
    """Fit 1q Lindblad RB model under physical bounds (T1 > 0, T2 <= 2*T1)."""
    m = np.asarray(lengths, dtype=float)
    y = np.asarray(p0_mean, dtype=float)
    sigma = np.maximum(np.asarray(p0_std, dtype=float), 1e-6)

    if t1_us_init <= 0.0:
        raise ValueError(f"t1_us_init must be positive, got {t1_us_init}.")
    t2_clamped = float(np.clip(t2_us_init, 1e-2, 2.0 * t1_us_init * (1.0 - 1e-9)))

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
            # 20% soft prior on T1 to break T1/T2 degeneracy in 1q RB
            prior_t1 = (t1_val - t1_us_init) / (0.2 * t1_us_init)
            return np.concatenate([(pred - y) / sigma, [prior_t1]])

        opt = least_squares(_res_joint, x0=x0, bounds=(lb, ub), method="trf")
        dof = max(len(m) - len(x0), 1)
        errs = _covariance_uncertainties(opt.jac, opt.fun, dof)

        t1_fit = float(opt.x[0])
        eta_fit = float(opt.x[1])
        t2_fit = 2.0 * t1_fit * eta_fit
        t1_err = float(errs[0])
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
    """Load an RB JSON, evaluate reported-calibration baseline, and fit Lindblad model."""
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
