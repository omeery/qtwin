"""2-Qubit Correlated ZZ Crosstalk Fitting Engine using QuTiP's 2-qubit Lindblad solver.

Models the degradation of Simultaneous 1-qubit RB decay rates (alpha_sim) relative to
Isolated 1-qubit RB (alpha_iso) via our QuTiP 2-qubit open quantum system engine
(`TwoQubitParams`, `simulate_two_qubit`):

- In Isolated RB (spectator idle in |0>), H_ZZ = zeta_ZZ |11><11| has zero effect,
  so each qubit k decays at alpha_iso_k = alpha_1q(T1_k, T2_k, t_cliff) * (1 - 2 * eps_loc_k).
- In Simultaneous RB (spectator twirled over C_1, average state I/2), evolving
  cardinal states of qubit k tensored with I/2 through QuTiP `simulate_two_qubit`
  under H = zeta_ZZ (n_a x n_b) yields the exact reduced depolarizing parameter:
      alpha_sim_lindblad_k(f_zz) = (exp(-t/T1_k) + 2*exp(-t/T2_k) * cos^2(pi * f_zz * t)) / 3
  and total simultaneous decay:
      alpha_sim_k = alpha_sim_lindblad_k(f_zz) * (1 - 2 * eps_loc_k) * (1 - 2 * eps_corr).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import qutip as qt
from scipy.optimize import least_squares

from qtwin.calibration.rb_model import compute_goodness_of_fit
from qtwin.physics import (
    SingleQubitParams,
    TwoQubitParams,
    excited_state,
    ground_state,
    plus_state,
    simulate_two_qubit,
)

CrosstalkFitMode = Literal["fit_zz", "fit_zz_and_corr"]


@dataclass(frozen=True)
class CrosstalkFitResult:
    """Fitted 2-qubit correlated Lindblad crosstalk parameters and diagnostics."""

    mode: str
    zz_coupling_mhz: float
    zz_coupling_mhz_err: float | None
    correlated_error: float
    correlated_error_err: float | None
    local_error_a: float
    local_error_a_err: float | None
    local_error_b: float
    local_error_b_err: float | None
    clifford_duration_us: float
    # Predicted RB metrics
    pred_epc_iso_a: float
    pred_epc_iso_b: float
    pred_epc_sim_a: float
    pred_epc_sim_b: float
    pred_delta_epc_a: float
    pred_delta_epc_b: float
    # Predicted curves
    p0_iso_a_pred: list[float]
    p0_iso_b_pred: list[float]
    p0_sim_a_pred: list[float]
    p0_sim_b_pred: list[float]
    # Combined goodness-of-fit across all 4 curves (iso_a, iso_b, sim_a, sim_b)
    rmse_total: float
    rmse_sim: float
    chi2_total: float
    dof_total: int
    reduced_chi2_total: float
    reduced_chi2_sim: float
    success: bool
    message: str


def _plus_i_state() -> qt.Qobj:
    return (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()


def lindblad_two_qubit_srb_alphas(
    params: TwoQubitParams,
    clifford_duration_us: float,
) -> tuple[float, float, float, float]:
    """Compute (alpha_iso_a, alpha_iso_b, alpha_sim_a, alpha_sim_b) via QuTiP 2q `mesolve`.

    For Isolated RB on qubit a, spectator b is initialized in |0><0|.
    For Simultaneous RB on qubit a, spectator b is twirled (maximally mixed I/2).
    """
    times = np.array([0.0, clifford_duration_us], dtype=float)
    eye = qt.qeye(2)
    rho_0 = qt.ket2dm(ground_state())
    rho_mix = 0.5 * eye

    sz_a, sx_a, sy_a = (
        qt.tensor(qt.sigmaz(), eye),
        qt.tensor(qt.sigmax(), eye),
        qt.tensor(qt.sigmay(), eye),
    )
    sz_b, sx_b, sy_b = (
        qt.tensor(eye, qt.sigmaz()),
        qt.tensor(eye, qt.sigmax()),
        qt.tensor(eye, qt.sigmay()),
    )

    def _alpha_for_qubit(target: int, spectator_state: qt.Qobj) -> float:
        cardinals = [
            (qt.ket2dm(ground_state()), sz_a if target == 0 else sz_b),
            (qt.ket2dm(excited_state()), sz_a if target == 0 else sz_b),
            (qt.ket2dm(plus_state()), sx_a if target == 0 else sx_b),
            (qt.ket2dm(_plus_i_state()), sy_a if target == 0 else sy_b),
        ]
        vals: list[float] = []
        for rho_t, obs in cardinals:
            rho_init = qt.tensor(rho_t, spectator_state) if target == 0 else qt.tensor(spectator_state, rho_t)
            res = simulate_two_qubit(params, rho_init, times, e_ops=[obs], store_states=False)
            vals.append(float(res.expect[0][-1]))
        d_z = 0.5 * (vals[0] - vals[1])
        d_x = vals[2]
        d_y = vals[3]
        return float(np.clip((d_x + d_y + d_z) / 3.0, 0.0, 1.0))

    alpha_iso_a = _alpha_for_qubit(0, rho_0)
    alpha_iso_b = _alpha_for_qubit(1, rho_0)
    alpha_sim_a = _alpha_for_qubit(0, rho_mix)
    alpha_sim_b = _alpha_for_qubit(1, rho_mix)
    return alpha_iso_a, alpha_iso_b, alpha_sim_a, alpha_sim_b


def predict_srb_curves(
    lengths: np.ndarray,
    *,
    t1_a_us: float,
    t2_a_us: float,
    t1_b_us: float,
    t2_b_us: float,
    clifford_duration_us: float,
    zz_coupling_mhz: float = 0.0,
    local_error_a: float = 0.0,
    local_error_b: float = 0.0,
    correlated_error: float = 0.0,
    amp_a: float = 0.5,
    off_a: float = 0.5,
    amp_b: float = 0.5,
    off_b: float = 0.5,
) -> dict[str, Any]:
    """Predict Isolated and Simultaneous RB survival curves from QuTiP 2q Lindblad dynamics."""
    m = np.asarray(lengths, dtype=float)
    params = TwoQubitParams(
        qubit0=SingleQubitParams(freq_mhz=0.0, t1_us=t1_a_us, t2_us=min(t2_a_us, 2.0 * t1_a_us)),
        qubit1=SingleQubitParams(freq_mhz=0.0, t1_us=t1_b_us, t2_us=min(t2_b_us, 2.0 * t1_b_us)),
        zz_coupling_mhz=zz_coupling_mhz,
    )
    a_iso_a_l, a_iso_b_l, a_sim_a_l, a_sim_b_l = lindblad_two_qubit_srb_alphas(
        params, clifford_duration_us
    )

    alpha_iso_a = float(np.clip(a_iso_a_l * (1.0 - 2.0 * local_error_a), 0.0, 1.0))
    alpha_iso_b = float(np.clip(a_iso_b_l * (1.0 - 2.0 * local_error_b), 0.0, 1.0))
    alpha_sim_a = float(
        np.clip(a_sim_a_l * (1.0 - 2.0 * local_error_a) * (1.0 - 2.0 * correlated_error), 0.0, 1.0)
    )
    alpha_sim_b = float(
        np.clip(a_sim_b_l * (1.0 - 2.0 * local_error_b) * (1.0 - 2.0 * correlated_error), 0.0, 1.0)
    )

    return {
        "alpha_iso_a": alpha_iso_a,
        "alpha_iso_b": alpha_iso_b,
        "alpha_sim_a": alpha_sim_a,
        "alpha_sim_b": alpha_sim_b,
        "epc_iso_a": 0.5 * (1.0 - alpha_iso_a),
        "epc_iso_b": 0.5 * (1.0 - alpha_iso_b),
        "epc_sim_a": 0.5 * (1.0 - alpha_sim_a),
        "epc_sim_b": 0.5 * (1.0 - alpha_sim_b),
        "p0_iso_a": amp_a * np.power(alpha_iso_a, m) + off_a,
        "p0_iso_b": amp_b * np.power(alpha_iso_b, m) + off_b,
        "p0_sim_a": amp_a * np.power(alpha_sim_a, m) + off_a,
        "p0_sim_b": amp_b * np.power(alpha_sim_b, m) + off_b,
    }


def _cov_uncertainties(jac: np.ndarray, residuals: np.ndarray, dof: int) -> np.ndarray:
    cov = np.linalg.pinv(jac.T @ jac) * float(np.sum(residuals**2) / max(dof, 1))
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def fit_crosstalk_to_srb(
    lengths: np.ndarray,
    p0_iso_a: np.ndarray,
    p0_iso_a_std: np.ndarray,
    p0_iso_b: np.ndarray,
    p0_iso_b_std: np.ndarray,
    p0_sim_a: np.ndarray,
    p0_sim_a_std: np.ndarray,
    p0_sim_b: np.ndarray,
    p0_sim_b_std: np.ndarray,
    *,
    t1_a_us: float,
    t2_a_us: float,
    t1_b_us: float,
    t2_b_us: float,
    clifford_duration_us: float,
    mode: CrosstalkFitMode = "fit_zz",
    zz_init_mhz: float = 0.2,
) -> CrosstalkFitResult:
    """Jointly fit local residual errors and 2-qubit ZZ crosstalk to Isolated + Simultaneous RB.

    Parameters fitted:
    - ``mode="fit_zz"``: ``(zz_mhz, eps_loc_a, eps_loc_b, amp_a, off_a, amp_b, off_b)``
      with ``eps_corr = 0``. Cleanly isolates static/effective ZZ coupling in MHz.
    - ``mode="fit_zz_and_corr"``: adds ``eps_corr >= 0`` for correlated 2q depolarizing noise.
    """
    m = np.asarray(lengths, dtype=float)
    y_ia, s_ia = np.asarray(p0_iso_a, dtype=float), np.maximum(np.asarray(p0_iso_a_std, dtype=float), 1e-6)
    y_ib, s_ib = np.asarray(p0_iso_b, dtype=float), np.maximum(np.asarray(p0_iso_b_std, dtype=float), 1e-6)
    y_sa, s_sa = np.asarray(p0_sim_a, dtype=float), np.maximum(np.asarray(p0_sim_a_std, dtype=float), 1e-6)
    y_sb, s_sb = np.asarray(p0_sim_b, dtype=float), np.maximum(np.asarray(p0_sim_b_std, dtype=float), 1e-6)

    amp_a0 = float(np.clip(y_ia[0] - 0.5, 0.1, 0.55))
    amp_b0 = float(np.clip(y_ib[0] - 0.5, 0.1, 0.55))

    if mode == "fit_zz":
        # [zz_mhz, eps_loc_a, eps_loc_b, amp_a, off_a, amp_b, off_b]
        x0 = [max(zz_init_mhz, 1e-3), 1e-4, 1e-4, amp_a0, 0.5, amp_b0, 0.5]
        lb = [0.0, 0.0, 0.0, 0.05, 0.2, 0.05, 0.2]
        ub = [10.0, 0.15, 0.15, 0.6, 0.8, 0.6, 0.8]
    elif mode == "fit_zz_and_corr":
        # [zz_mhz, eps_corr, eps_loc_a, eps_loc_b, amp_a, off_a, amp_b, off_b]
        x0 = [max(zz_init_mhz, 1e-3), 1e-4, 1e-4, 1e-4, amp_a0, 0.5, amp_b0, 0.5]
        lb = [0.0, 0.0, 0.0, 0.0, 0.05, 0.2, 0.05, 0.2]
        ub = [10.0, 0.15, 0.15, 0.15, 0.6, 0.8, 0.6, 0.8]
    else:
        raise ValueError(f"Unsupported crosstalk fit mode: {mode!r}")

    def _unpack(theta: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
        if mode == "fit_zz":
            return (
                float(theta[0]),
                0.0,
                float(theta[1]),
                float(theta[2]),
                float(theta[3]),
                float(theta[4]),
                float(theta[5]),
                float(theta[6]),
            )
        return (
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            float(theta[3]),
            float(theta[4]),
            float(theta[5]),
            float(theta[6]),
            float(theta[7]),
        )

    def _residuals(theta: np.ndarray) -> np.ndarray:
        zz, eps_c, eps_a, eps_b, aa, oa, ab, ob = _unpack(theta)
        pred = predict_srb_curves(
            m,
            t1_a_us=t1_a_us,
            t2_a_us=t2_a_us,
            t1_b_us=t1_b_us,
            t2_b_us=t2_b_us,
            clifford_duration_us=clifford_duration_us,
            zz_coupling_mhz=zz,
            local_error_a=eps_a,
            local_error_b=eps_b,
            correlated_error=eps_c,
            amp_a=aa,
            off_a=oa,
            amp_b=ab,
            off_b=ob,
        )
        return np.concatenate(
            [
                (pred["p0_iso_a"] - y_ia) / s_ia,
                (pred["p0_iso_b"] - y_ib) / s_ib,
                (pred["p0_sim_a"] - y_sa) / s_sa,
                (pred["p0_sim_b"] - y_sb) / s_sb,
            ]
        )

    opt = least_squares(_residuals, x0=x0, bounds=(lb, ub), method="trf")
    n_pts = 4 * len(m)
    dof = max(n_pts - len(x0), 1)
    errs = _cov_uncertainties(opt.jac, opt.fun, dof)

    zz, eps_c, eps_a, eps_b, aa, oa, ab, ob = _unpack(opt.x)
    if mode == "fit_zz":
        zz_err, eps_c_err, eps_a_err, eps_b_err = float(errs[0]), None, float(errs[1]), float(errs[2])
    else:
        zz_err, eps_c_err, eps_a_err, eps_b_err = (
            float(errs[0]),
            float(errs[1]),
            float(errs[2]),
            float(errs[3]),
        )

    pred = predict_srb_curves(
        m,
        t1_a_us=t1_a_us,
        t2_a_us=t2_a_us,
        t1_b_us=t1_b_us,
        t2_b_us=t2_b_us,
        clifford_duration_us=clifford_duration_us,
        zz_coupling_mhz=zz,
        local_error_a=eps_a,
        local_error_b=eps_b,
        correlated_error=eps_c,
        amp_a=aa,
        off_a=oa,
        amp_b=ab,
        off_b=ob,
    )

    y_all = np.concatenate([y_ia, y_ib, y_sa, y_sb])
    s_all = np.concatenate([s_ia, s_ib, s_sa, s_sb])
    p_all = np.concatenate([pred["p0_iso_a"], pred["p0_iso_b"], pred["p0_sim_a"], pred["p0_sim_b"]])
    gof_all = compute_goodness_of_fit(y_all, p_all, s_all, n_params=len(x0))

    y_sim = np.concatenate([y_sa, y_sb])
    s_sim = np.concatenate([s_sa, s_sb])
    p_sim = np.concatenate([pred["p0_sim_a"], pred["p0_sim_b"]])
    gof_sim = compute_goodness_of_fit(y_sim, p_sim, s_sim, n_params=1 if mode == "fit_zz" else 2)

    return CrosstalkFitResult(
        mode=mode,
        zz_coupling_mhz=zz,
        zz_coupling_mhz_err=zz_err,
        correlated_error=eps_c,
        correlated_error_err=eps_c_err,
        local_error_a=eps_a,
        local_error_a_err=eps_a_err,
        local_error_b=eps_b,
        local_error_b_err=eps_b_err,
        clifford_duration_us=clifford_duration_us,
        pred_epc_iso_a=float(pred["epc_iso_a"]),
        pred_epc_iso_b=float(pred["epc_iso_b"]),
        pred_epc_sim_a=float(pred["epc_sim_a"]),
        pred_epc_sim_b=float(pred["epc_sim_b"]),
        pred_delta_epc_a=float(pred["epc_sim_a"] - pred["epc_iso_a"]),
        pred_delta_epc_b=float(pred["epc_sim_b"] - pred["epc_iso_b"]),
        p0_iso_a_pred=[float(v) for v in pred["p0_iso_a"]],
        p0_iso_b_pred=[float(v) for v in pred["p0_iso_b"]],
        p0_sim_a_pred=[float(v) for v in pred["p0_sim_a"]],
        p0_sim_b_pred=[float(v) for v in pred["p0_sim_b"]],
        rmse_total=float(gof_all["rmse"]),
        rmse_sim=float(gof_sim["rmse"]),
        chi2_total=float(gof_all["chi2"]),
        dof_total=int(gof_all["dof"]),
        reduced_chi2_total=float(gof_all["reduced_chi2"]),
        reduced_chi2_sim=float(gof_sim["reduced_chi2"]),
        success=bool(opt.success),
        message=str(opt.message),
    )


def fit_srb_json(
    json_path: Path | str,
    *,
    mode: CrosstalkFitMode = "fit_zz",
) -> dict[str, Any]:
    """Load an SRB JSON, compute uncorrelated baseline (ZZ=0), and fit correlated Lindblad model."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    lengths = np.asarray(data["lengths"], dtype=float)
    cal_a = data["calibration_a"]
    cal_b = data["calibration_b"]

    t1_a = float(cal_a["t1_us"]) if cal_a.get("t1_us") is not None else 80.0
    t2_a = float(cal_a["t2_us"]) if cal_a.get("t2_us") is not None else 50.0
    t1_b = float(cal_b["t1_us"]) if cal_b.get("t1_us") is not None else 80.0
    t2_b = float(cal_b["t2_us"]) if cal_b.get("t2_us") is not None else 50.0
    sx_ns = float(cal_a["sx_duration_ns"]) if cal_a.get("sx_duration_ns") is not None else 35.0
    t_cliff_us = (sx_ns * 1e-3) * float(data.get("avg_physical_gates_per_clifford", 1.0))

    fit_res = fit_crosstalk_to_srb(
        lengths,
        np.asarray(data["p0_iso_a"], dtype=float),
        np.asarray(data["p0_iso_a_std"], dtype=float),
        np.asarray(data["p0_iso_b"], dtype=float),
        np.asarray(data["p0_iso_b_std"], dtype=float),
        np.asarray(data["p0_sim_a"], dtype=float),
        np.asarray(data["p0_sim_a_std"], dtype=float),
        np.asarray(data["p0_sim_b"], dtype=float),
        np.asarray(data["p0_sim_b_std"], dtype=float),
        t1_a_us=t1_a,
        t2_a_us=t2_a,
        t1_b_us=t1_b,
        t2_b_us=t2_b,
        clifford_duration_us=t_cliff_us,
        mode=mode,
    )

    # Uncorrelated baseline (zz_coupling_mhz = 0, correlated_error = 0, using fitted local errors & SPAM)
    # Use isolated SPAM from raw fits for clean comparison
    uncorr = predict_srb_curves(
        lengths,
        t1_a_us=t1_a,
        t2_a_us=t2_a,
        t1_b_us=t1_b,
        t2_b_us=t2_b,
        clifford_duration_us=t_cliff_us,
        zz_coupling_mhz=0.0,
        local_error_a=fit_res.local_error_a,
        local_error_b=fit_res.local_error_b,
        correlated_error=0.0,
        amp_a=data["iso_a"]["amplitude"],
        off_a=data["iso_a"]["offset"],
        amp_b=data["iso_b"]["amplitude"],
        off_b=data["iso_b"]["offset"],
    )
    y_sim = np.concatenate([data["p0_sim_a"], data["p0_sim_b"]])
    s_sim = np.concatenate([data["p0_sim_a_std"], data["p0_sim_b_std"]])
    p_uncorr_sim = np.concatenate([uncorr["p0_sim_a"], uncorr["p0_sim_b"]])
    uncorr_sim_gof = compute_goodness_of_fit(y_sim, p_uncorr_sim, s_sim, n_params=0)

    return {
        "raw": data,
        "clifford_duration_us": t_cliff_us,
        "uncorrelated_baseline": {
            "zz_coupling_mhz": 0.0,
            "epc_sim_a": float(uncorr["epc_sim_a"]),
            "epc_sim_b": float(uncorr["epc_sim_b"]),
            "p0_sim_a_pred": [float(v) for v in uncorr["p0_sim_a"]],
            "p0_sim_b_pred": [float(v) for v in uncorr["p0_sim_b"]],
            "rmse_sim": float(uncorr_sim_gof["rmse"]),
            "reduced_chi2_sim": float(uncorr_sim_gof["reduced_chi2"]),
        },
        "fit": fit_res,
    }
