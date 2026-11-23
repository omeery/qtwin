"""Phase 1 Validation: Lindblad Noise Fitting & Prediction-vs-Measurement Benchmark.

Runs the QuTiP Lindblad RB fitter on:
1. Local Qiskit Aer dry-run RB dataset (ground-truth verification).
2. Live IBM Quantum hardware RB dataset (reported-calibration baseline vs.
   fitted Lindblad model vs. actual QPU measurement).

Generates publication-grade comparison plots (with residual panels) under
``outputs/plots/`` and a Markdown validation report at
``outputs/reports/phase1_validation.md``.

Usage:
    python scripts/run_phase1_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qtwin.calibration import fit_rb_json, predict_rb_survival

ROOT_DIR = Path(__file__).resolve().parents[1]
HARDWARE_DIR = ROOT_DIR / "outputs" / "hardware"
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"


def _find_latest_rb_json(mode: str) -> Path | None:
    """Find the most recent Phase 1 RB JSON (containing ``p0_mean``) for a given mode."""
    if not HARDWARE_DIR.exists():
        return None
    candidates = sorted(HARDWARE_DIR.glob(f"rb_1q_{mode}_*.json"), reverse=True)
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if "p0_mean" in data and "calibration" in data:
                return path
        except Exception:
            continue
    return None


def _plot_comparison(
    analysis_t2: dict[str, Any],
    analysis_res: dict[str, Any],
    *,
    title: str,
    baseline_label: str,
    out_path: Path,
) -> None:
    """Create a 2-panel publication-grade RB comparison figure with residuals."""
    raw = analysis_t2["raw"]
    base = analysis_t2["baseline"]
    fit_t2 = analysis_t2["fit"]
    fit_res = analysis_res["fit"]

    # Choose the primary fitted model with lower RMSE (typically identical or very close)
    primary_fit = fit_res if fit_res.rmse <= fit_t2.rmse else fit_t2
    primary_label = (
        f"Fitted Lindblad + residual ($\\epsilon_{{\\mathrm{{res}}}}={fit_res.residual_error:.2e}$)"
        if primary_fit is fit_res
        else f"Fitted Lindblad ($T_2={fit_t2.t2_us:.1f}\\,\\mu$s)"
    )

    m_obs = np.asarray(raw["lengths"], dtype=float)
    y_obs = np.asarray(raw["p0_mean"], dtype=float)
    y_err = np.asarray(raw["p0_std"], dtype=float)

    m_dense = np.linspace(0.0, float(np.max(m_obs)), 300)
    y_base_dense, _, _ = predict_rb_survival(
        m_dense,
        t1_us=base["t1_us"],
        t2_us=min(base["t2_us"], 2.0 * base["t1_us"]),
        clifford_duration_us=analysis_t2["clifford_duration_us"],
        amplitude=primary_fit.amplitude,
        offset=primary_fit.offset,
        residual_error=0.0,
    )
    y_fit_dense, _, _ = predict_rb_survival(
        m_dense,
        t1_us=primary_fit.t1_us,
        t2_us=primary_fit.t2_us,
        clifford_duration_us=analysis_t2["clifford_duration_us"],
        amplitude=primary_fit.amplitude,
        offset=primary_fit.offset,
        residual_error=primary_fit.residual_error,
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.4),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3]},
    )

    # Top panel: survival curves
    ax1.errorbar(
        m_obs,
        y_obs,
        yerr=y_err,
        fmt="o",
        color="#111827",
        ecolor="#4b5563",
        capsize=3.5,
        markersize=5.5,
        zorder=5,
        label="Measured RB $P_0(m)$",
    )
    ax1.plot(
        m_dense,
        y_base_dense,
        "--",
        color="#2563eb",
        linewidth=1.9,
        label=f"{baseline_label} (EPC={base['predicted_epc']:.2e})",
    )
    ax1.plot(
        m_dense,
        y_fit_dense,
        "-",
        color="#dc2626",
        linewidth=2.2,
        label=f"{primary_label} (EPC={primary_fit.predicted_epc:.2e})",
    )

    epc_meas = raw["epc"]
    epc_std_str = f" ± {raw['epc_std']:.2e}" if raw.get("epc_std") is not None else ""
    metrics_box = (
        f"Measured EPC: {epc_meas:.2e}{epc_std_str}\n"
        f"Baseline EPC: {base['predicted_epc']:.2e} (RMSE={base['rmse']:.4f})\n"
        f"Fitted EPC:   {primary_fit.predicted_epc:.2e} (RMSE={primary_fit.rmse:.4f}, "
        f"$\\chi^2_\\nu$={primary_fit.reduced_chi2:.2f})"
    )
    ax1.text(
        0.03,
        0.06,
        metrics_box,
        transform=ax1.transAxes,
        fontsize=8.5,
        family="monospace",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f9fafb", "edgecolor": "#d1d5db"},
    )
    ax1.set_title(title)
    ax1.set_ylabel(r"Ground-State Survival Probability $P_0(m)$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    # Bottom panel: residuals
    res_base = y_obs - np.asarray(base["p0_pred"], dtype=float)
    res_fit = y_obs - np.asarray(primary_fit.p0_pred, dtype=float)

    ax2.axhline(0.0, color="#6b7280", linestyle=":", linewidth=1.2)
    ax2.plot(m_obs, res_base, "s--", color="#2563eb", markersize=4.5, alpha=0.8, label="Baseline residual")
    ax2.errorbar(
        m_obs,
        res_fit,
        yerr=y_err,
        fmt="o-",
        color="#dc2626",
        ecolor="#f87171",
        capsize=3,
        markersize=4.5,
        label="Fitted Lindblad residual",
    )
    ax2.set_xlabel("Clifford Sequence Length $m$")
    ax2.set_ylabel(r"$P_{0,\mathrm{obs}} - P_{0,\mathrm{pred}}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _format_val(val: float | None, err: float | None = None, fmt: str = ".4g") -> str:
    if val is None:
        return "N/A"
    if err is not None:
        return f"{val:{fmt}} ± {err:{fmt}}"
    return f"{val:{fmt}}"


def _build_report_section(
    heading: str,
    json_path: Path,
    analysis_t2: dict[str, Any],
    analysis_res: dict[str, Any],
    plot_rel_path: str,
) -> str:
    raw = analysis_t2["raw"]
    cal = raw.get("calibration", {})
    base = analysis_t2["baseline"]
    fit_t2 = analysis_t2["fit"]
    fit_res = analysis_res["fit"]

    lines = [
        f"## {heading}",
        "",
        f"- **Backend:** `{raw['backend_name']}` (Qubit `{raw['qubit']}`)",
        f"- **Source JSON:** `{json_path.name}`",
        f"- **Job IDs:** `{raw.get('job_ids', [])}`",
        f"- **RB Settings:** lengths `{raw['lengths']}`, samples `{raw['num_samples']}`, shots `{raw['shots']}`",
        f"- **Effective Clifford Duration:** `{analysis_t2['clifford_duration_us']*1e3:.2f} ns` "
        f"(`{raw['avg_physical_gates_per_clifford']:.3f}` physical gates/Clifford × `{cal.get('sx_duration_ns')} ns`)",
        f"- **Reported Calibration:** $T_1 = {cal.get('t1_us'):.2f}\\,\\mu\\text{{s}}$, "
        f"$T_2 = {cal.get('t2_us'):.2f}\\,\\mu\\text{{s}}$, "
        f"`sx_err` = `{cal.get('sx_error')}`, `readout_err` = `{cal.get('readout_error')}`",
        "",
        "| Model | $T_1$ ($\\mu$s) | $T_2$ ($\\mu$s) | Residual $\\epsilon_{\\mathrm{res}}$ | Predicted EPC | RMSE | Reduced $\\chi^2_\\nu$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Measured (Qiskit RB)** | — | — | — | {_format_val(raw['epc'], raw.get('epc_std'), '.3e')} | — | — |",
        f"| **Reported $T_1/T_2$ Lindblad Baseline** | {base['t1_us']:.2f} | {base['t2_us']:.2f} | 0 | {base['predicted_epc']:.3e} | {base['rmse']:.5f} | {base['reduced_chi2']:.2f} |",
        f"| **Fitted Lindblad (Effective $T_2$)** | {fit_t2.t1_us:.2f} (fixed) | {_format_val(fit_t2.t2_us, fit_t2.t2_us_err, '.2f')} | 0 | {fit_t2.predicted_epc:.3e} | {fit_t2.rmse:.5f} | {fit_t2.reduced_chi2:.2f} |",
        f"| **Fitted Lindblad ($T_1/T_2$ + Residual)** | {fit_res.t1_us:.2f} (fixed) | {fit_res.t2_us:.2f} (fixed) | {_format_val(fit_res.residual_error, fit_res.residual_error_err, '.2e')} | {fit_res.predicted_epc:.3e} | {fit_res.rmse:.5f} | {fit_res.reduced_chi2:.2f} |",
        "",
        f"![{heading}]({plot_rel_path})",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run Phase 1 validation across Aer dry-run and live IBM Quantum datasets."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    aer_json = _find_latest_rb_json("dry_run_aer")
    ibm_json = _find_latest_rb_json("ibm_quantum")

    report_parts = [
        "# Phase 1 Validation Report — Lindblad Noise Fitting & Hardware Benchmark",
        "",
        "This report validates the QuTiP Lindblad RB calibration fitter (`qtwin.calibration.rb_model`) "
        "against both a synthetic ground-truth Qiskit Aer thermal-relaxation simulator and live IBM Quantum hardware.",
        "",
    ]

    if aer_json is not None:
        aer_t2 = fit_rb_json(aer_json, mode="fit_t2")
        aer_res = fit_rb_json(aer_json, mode="fit_residual")
        aer_plot = PLOTS_DIR / "phase1_aer_validation.png"
        _plot_comparison(
            aer_t2,
            aer_res,
            title="Phase 1 Ground-Truth Verification — Qiskit Aer Thermal Relaxation RB",
            baseline_label="Ground-truth $T_1/T_2$ Lindblad",
            out_path=aer_plot,
        )
        report_parts.append(
            _build_report_section(
                "1. Ground-Truth Verification (Qiskit Aer Dry-Run)",
                aer_json,
                aer_t2,
                aer_res,
                "../plots/phase1_aer_validation.png",
            )
        )
        print(f"[Aer] Measured EPC={aer_t2['raw']['epc']:.3e} | Baseline EPC={aer_t2['baseline']['predicted_epc']:.3e} | Fitted EPC={aer_t2['fit'].predicted_epc:.3e} | RMSE={aer_t2['fit'].rmse:.5f}")

    if ibm_json is not None:
        ibm_t2 = fit_rb_json(ibm_json, mode="fit_t2")
        ibm_res = fit_rb_json(ibm_json, mode="fit_residual")
        ibm_plot = PLOTS_DIR / "phase1_ibm_validation.png"
        backend_name = ibm_t2["raw"]["backend_name"]
        qubit = ibm_t2["raw"]["qubit"]
        _plot_comparison(
            ibm_t2,
            ibm_res,
            title=f"Phase 1 Live Hardware Benchmark — IBM Quantum `{backend_name}` (Qubit {qubit})",
            baseline_label="IBM reported $T_1/T_2$ Lindblad",
            out_path=ibm_plot,
        )
        report_parts.append(
            _build_report_section(
                f"2. Live IBM Quantum Hardware Benchmark (`{backend_name}`, Qubit {qubit})",
                ibm_json,
                ibm_t2,
                ibm_res,
                "../plots/phase1_ibm_validation.png",
            )
        )
        print(
            f"[IBM {backend_name}] Measured EPC={ibm_t2['raw']['epc']:.3e} | "
            f"IBM-Reported Lindblad Baseline EPC={ibm_t2['baseline']['predicted_epc']:.3e} (RMSE={ibm_t2['baseline']['rmse']:.5f}) | "
            f"Fitted (T2) EPC={ibm_t2['fit'].predicted_epc:.3e} (RMSE={ibm_t2['fit'].rmse:.5f}, chi2_nu={ibm_t2['fit'].reduced_chi2:.2f}) | "
            f"Fitted (Residual) EPC={ibm_res['fit'].predicted_epc:.3e} (RMSE={ibm_res['fit'].rmse:.5f}, chi2_nu={ibm_res['fit'].reduced_chi2:.2f})"
        )
    else:
        report_parts.append(
            "## 2. Live IBM Quantum Hardware Benchmark\n\n"
            "_No completed live IBM Quantum RB JSON found yet in `outputs/hardware/`._\n"
        )

    report_path = REPORTS_DIR / "phase1_validation.md"
    report_path.write_text("\n".join(report_parts), encoding="utf-8")
    print(f"Saved validation report: {report_path}")


if __name__ == "__main__":
    main()
