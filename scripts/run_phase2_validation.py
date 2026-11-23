"""Phase 2 Validation: 2-Qubit Simultaneous RB & Correlated ZZ Crosstalk Extraction.

Runs the QuTiP 2-qubit Lindblad crosstalk fitter (`qtwin.calibration.crosstalk_model`)
on:
1. Local Qiskit Aer dry-run SRB dataset with known synthetic ZZ coupling (ground truth).
2. Live IBM Quantum `ibm_fez` SRB dataset on a physically connected qubit pair.

Generates publication-grade comparison plots under ``outputs/plots/`` and a concise
Markdown validation report at ``outputs/reports/phase2_crosstalk_validation.md``.

Usage:
    python scripts/run_phase2_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qtwin.calibration import fit_srb_json, predict_srb_curves

ROOT_DIR = Path(__file__).resolve().parents[1]
HARDWARE_DIR = ROOT_DIR / "outputs" / "hardware"
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"


def _find_latest_srb_json(mode: str) -> Path | None:
    if not HARDWARE_DIR.exists():
        return None
    candidates = sorted(HARDWARE_DIR.glob(f"srb_2q_{mode}_*.json"), reverse=True)
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if "p0_iso_a" in data and "p0_sim_a" in data:
                return path
        except Exception:
            continue
    return None


def _plot_crosstalk_comparison(
    analysis: dict[str, Any],
    *,
    title: str,
    out_path: Path,
) -> None:
    """Create a 2x2 publication-grade figure: Q_a & Q_b survival curves + residuals."""
    raw = analysis["raw"]
    uncorr = analysis["uncorrelated_baseline"]
    fit = analysis["fit"]
    qa, qb = raw["qubit_pair"]
    m_obs = np.asarray(raw["lengths"], dtype=float)
    m_dense = np.linspace(0.0, float(np.max(m_obs)), 300)

    cal_a, cal_b = raw["calibration_a"], raw["calibration_b"]
    t1_a = float(cal_a["t1_us"]) if cal_a.get("t1_us") is not None else 80.0
    t2_a = float(cal_a["t2_us"]) if cal_a.get("t2_us") is not None else 50.0
    t1_b = float(cal_b["t1_us"]) if cal_b.get("t1_us") is not None else 80.0
    t2_b = float(cal_b["t2_us"]) if cal_b.get("t2_us") is not None else 50.0
    t_cliff = analysis["clifford_duration_us"]

    dense_uncorr = predict_srb_curves(
        m_dense,
        t1_a_us=t1_a,
        t2_a_us=t2_a,
        t1_b_us=t1_b,
        t2_b_us=t2_b,
        clifford_duration_us=t_cliff,
        zz_coupling_mhz=0.0,
        local_error_a=fit.local_error_a,
        local_error_b=fit.local_error_b,
        amp_a=raw["iso_a"]["amplitude"],
        off_a=raw["iso_a"]["offset"],
        amp_b=raw["iso_b"]["amplitude"],
        off_b=raw["iso_b"]["offset"],
    )
    dense_fit = predict_srb_curves(
        m_dense,
        t1_a_us=t1_a,
        t2_a_us=t2_a,
        t1_b_us=t1_b,
        t2_b_us=t2_b,
        clifford_duration_us=t_cliff,
        zz_coupling_mhz=fit.zz_coupling_mhz,
        local_error_a=fit.local_error_a,
        local_error_b=fit.local_error_b,
        correlated_error=fit.correlated_error,
        amp_a=raw["iso_a"]["amplitude"],
        off_a=raw["iso_a"]["offset"],
        amp_b=raw["iso_b"]["amplitude"],
        off_b=raw["iso_b"]["offset"],
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 6.8),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3]},
    )

    configs = [
        (
            0,
            f"Qubit {qa}",
            np.asarray(raw["p0_iso_a"]),
            np.asarray(raw["p0_iso_a_std"]),
            np.asarray(raw["p0_sim_a"]),
            np.asarray(raw["p0_sim_a_std"]),
            dense_uncorr["p0_sim_a"],
            dense_fit["p0_sim_a"],
            np.asarray(uncorr["p0_sim_a_pred"]),
            np.asarray(fit.p0_sim_a_pred),
            raw["iso_a"]["epc"],
            raw["sim_a"]["epc"],
            raw["delta_epc_a"],
        ),
        (
            1,
            f"Qubit {qb}",
            np.asarray(raw["p0_iso_b"]),
            np.asarray(raw["p0_iso_b_std"]),
            np.asarray(raw["p0_sim_b"]),
            np.asarray(raw["p0_sim_b_std"]),
            dense_uncorr["p0_sim_b"],
            dense_fit["p0_sim_b"],
            np.asarray(uncorr["p0_sim_b_pred"]),
            np.asarray(fit.p0_sim_b_pred),
            raw["iso_b"]["epc"],
            raw["sim_b"]["epc"],
            raw["delta_epc_b"],
        ),
    ]

    for col, qlabel, y_iso, s_iso, y_sim, s_sim, d_uncorr, d_fit, p_uncorr, p_fit, epc_i, epc_s, depc in configs:
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.errorbar(
            m_obs,
            y_iso,
            yerr=s_iso,
            fmt="o",
            color="#2563eb",
            ecolor="#93c5fd",
            capsize=3,
            markersize=5,
            label=f"Isolated RB (EPC={epc_i:.2e})",
        )
        ax_top.errorbar(
            m_obs,
            y_sim,
            yerr=s_sim,
            fmt="s",
            color="#111827",
            ecolor="#6b7280",
            capsize=3,
            markersize=5,
            zorder=5,
            label=f"Simultaneous RB (EPC={epc_s:.2e})",
        )
        ax_top.plot(
            m_dense,
            d_uncorr,
            "--",
            color="#2563eb",
            linewidth=1.6,
            alpha=0.8,
            label=r"Uncorrelated baseline ($\zeta_{ZZ}=0$)",
        )
        ax_top.plot(
            m_dense,
            d_fit,
            "-",
            color="#dc2626",
            linewidth=2.1,
            label=f"Correlated Lindblad ($f_{{ZZ}}={fit.zz_coupling_mhz:.3f}$ MHz)",
        )
        ax_top.set_title(f"{qlabel} ($\\Delta\\mathrm{{EPC}} = {depc:+.2e}$)")
        if col == 0:
            ax_top.set_ylabel(r"Survival Probability $P_0(m)$")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="lower left", fontsize=8.0, frameon=True)

        # Bottom panel: Simultaneous RB residuals
        ax_bot.axhline(0.0, color="#6b7280", linestyle=":", linewidth=1.1)
        ax_bot.plot(
            m_obs,
            y_sim - p_uncorr,
            "s--",
            color="#2563eb",
            markersize=4,
            alpha=0.8,
            label="Uncorrelated residual",
        )
        ax_bot.errorbar(
            m_obs,
            y_sim - p_fit,
            yerr=s_sim,
            fmt="o-",
            color="#dc2626",
            ecolor="#f87171",
            capsize=2.5,
            markersize=4,
            label="Correlated Lindblad residual",
        )
        ax_bot.set_xlabel("Clifford Sequence Length $m$")
        if col == 0:
            ax_bot.set_ylabel(r"$P_{0,\mathrm{sim}} - P_{0,\mathrm{pred}}$")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="lower left", fontsize=7.5, frameon=True)

    zz_err_str = f" ± {fit.zz_coupling_mhz_err:.3f}" if fit.zz_coupling_mhz_err is not None else ""
    fig.suptitle(
        f"{title}\n"
        f"Extracted $f_{{ZZ}} = {fit.zz_coupling_mhz:.3f}{zz_err_str}$ MHz | "
        f"Uncorr Sim RMSE={uncorr['rmse_sim']:.4f} ($\\chi^2_\\nu$={uncorr['reduced_chi2_sim']:.2f}) -> "
        f"Correlated Sim RMSE={fit.rmse_sim:.4f} ($\\chi^2_\\nu$={fit.reduced_chi2_sim:.2f})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _format_val(val: float | None, err: float | None = None, fmt: str = ".3e") -> str:
    if val is None:
        return "N/A"
    if err is not None:
        return f"{val:{fmt}} ± {err:{fmt}}"
    return f"{val:{fmt}}"


def _build_section(
    heading: str,
    json_path: Path,
    analysis: dict[str, Any],
    plot_rel_path: str,
) -> str:
    raw = analysis["raw"]
    uncorr = analysis["uncorrelated_baseline"]
    fit = analysis["fit"]
    qa, qb = raw["qubit_pair"]
    cal_a, cal_b = raw["calibration_a"], raw["calibration_b"]

    lines = [
        f"## {heading}",
        "",
        f"- **Backend:** `{raw['backend_name']}` | **Qubit Pair:** `({qa}, {qb})`",
        f"- **Source JSON:** `{json_path.name}` | **Job IDs:** `{raw.get('job_ids', [])}`",
        f"- **SRB Settings:** lengths `{raw['lengths']}`, samples `{raw['num_samples']}`, shots `{raw['shots']}`",
        f"- **Effective Clifford Duration:** `{analysis['clifford_duration_us']*1e3:.2f} ns` "
        f"(`{raw['avg_physical_gates_per_clifford']:.3f}` phys gates/Clifford)",
        f"- **Reported Pair Calibration:** "
        f"Q{qa} ($T_1={cal_a.get('t1_us'):.2f}\\,\\mu$s, $T_2={cal_a.get('t2_us'):.2f}\\,\\mu$s), "
        f"Q{qb} ($T_1={cal_b.get('t1_us'):.2f}\\,\\mu$s, $T_2={cal_b.get('t2_us'):.2f}\\,\\mu$s)",
    ]
    if raw.get("synthetic_zz_mhz") is not None:
        lines.append(f"- **Injected Ground-Truth $f_{{ZZ}}$:** `{raw['synthetic_zz_mhz']:.3f} MHz`")

    lines.extend(
        [
            "",
            "### Measured Isolated vs. Simultaneous RB",
            "",
            "| Qubit | $\\mathrm{EPC}_{\\mathrm{iso}}$ | $\\mathrm{EPC}_{\\mathrm{sim}}$ | $\\Delta\\mathrm{EPC} = \\mathrm{EPC}_{\\mathrm{sim}} - \\mathrm{EPC}_{\\mathrm{iso}}$ |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Q{qa}** | {_format_val(raw['iso_a']['epc'], raw['iso_a'].get('epc_std'))} | {_format_val(raw['sim_a']['epc'], raw['sim_a'].get('epc_std'))} | `{raw['delta_epc_a']:+.3e}` |",
            f"| **Q{qb}** | {_format_val(raw['iso_b']['epc'], raw['iso_b'].get('epc_std'))} | {_format_val(raw['sim_b']['epc'], raw['sim_b'].get('epc_std'))} | `{raw['delta_epc_b']:+.3e}` |",
            "",
            "### Uncorrelated Baseline vs. Correlated 2-Qubit Lindblad Fit",
            "",
            "| Model | Extracted $f_{ZZ}$ (MHz) | Pred $\\Delta\\mathrm{EPC}$ (Q"
            + f"{qa} / Q{qb}) | Sim RMSE | Sim Reduced $\\chi^2_\\nu$ |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Uncorrelated Baseline ($f_{{ZZ}}=0$)** | `0.000` | `0 / 0` | `{uncorr['rmse_sim']:.5f}` | `{uncorr['reduced_chi2_sim']:.2f}` |",
            f"| **Correlated Lindblad (`fit_zz`)** | {_format_val(fit.zz_coupling_mhz, fit.zz_coupling_mhz_err, '.4f')} | `{fit.pred_delta_epc_a:+.2e}` / `{fit.pred_delta_epc_b:+.2e}` | `{fit.rmse_sim:.5f}` | `{fit.reduced_chi2_sim:.2f}` |",
            "",
            f"![{heading}]({plot_rel_path})",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    aer_json = _find_latest_srb_json("dry_run_aer")
    ibm_json = _find_latest_srb_json("ibm_quantum")

    parts = [
        "# Phase 2 Validation Report — 2-Qubit Simultaneous RB & Correlated ZZ Crosstalk",
        "",
        "Validates the QuTiP 2-qubit Lindblad crosstalk fitter (`qtwin.calibration.crosstalk_model`) "
        "on synthetic Qiskit Aer ground truth and live IBM Quantum hardware.",
        "",
    ]

    if aer_json is not None:
        aer_res = fit_srb_json(aer_json, mode="fit_zz")
        _plot_crosstalk_comparison(
            aer_res,
            title="Phase 2 Ground-Truth Verification — Qiskit Aer 2-Qubit SRB (Synthetic ZZ)",
            out_path=PLOTS_DIR / "phase2_aer_crosstalk.png",
        )
        parts.append(
            _build_section(
                "1. Ground-Truth Verification (Qiskit Aer Dry-Run)",
                aer_json,
                aer_res,
                "../plots/phase2_aer_crosstalk.png",
            )
        )
        fit = aer_res["fit"]
        uncorr = aer_res["uncorrelated_baseline"]
        print(
            f"[Aer SRB] True ZZ={aer_res['raw']['synthetic_zz_mhz']:.3f} MHz | "
            f"Extracted ZZ={fit.zz_coupling_mhz:.4f} +/- {fit.zz_coupling_mhz_err:.4f} MHz | "
            f"Sim RMSE: {uncorr['rmse_sim']:.5f} -> {fit.rmse_sim:.5f} | "
            f"Sim chi2_nu: {uncorr['reduced_chi2_sim']:.2f} -> {fit.reduced_chi2_sim:.2f}"
        )

    if ibm_json is not None:
        ibm_res = fit_srb_json(ibm_json, mode="fit_zz")
        backend_name = ibm_res["raw"]["backend_name"]
        qa, qb = ibm_res["raw"]["qubit_pair"]
        _plot_crosstalk_comparison(
            ibm_res,
            title=f"Phase 2 Live Hardware Benchmark — IBM Quantum {backend_name} Pair ({qa}, {qb})",
            out_path=PLOTS_DIR / "phase2_ibm_crosstalk.png",
        )
        parts.append(
            _build_section(
                f"2. Live IBM Quantum Hardware Benchmark (`{backend_name}`, Pair `({qa}, {qb})`)",
                ibm_json,
                ibm_res,
                "../plots/phase2_ibm_crosstalk.png",
            )
        )
        fit = ibm_res["fit"]
        uncorr = ibm_res["uncorrelated_baseline"]
        print(
            f"[IBM {backend_name} ({qa},{qb})] "
            f"Delta_EPC=({ibm_res['raw']['delta_epc_a']:+.2e}, {ibm_res['raw']['delta_epc_b']:+.2e}) | "
            f"Extracted ZZ={fit.zz_coupling_mhz:.4f} +/- {fit.zz_coupling_mhz_err:.4f} MHz | "
            f"Sim RMSE: {uncorr['rmse_sim']:.5f} -> {fit.rmse_sim:.5f} | "
            f"Sim chi2_nu: {uncorr['reduced_chi2_sim']:.2f} -> {fit.reduced_chi2_sim:.2f}"
        )
    else:
        parts.append(
            "## 2. Live IBM Quantum Hardware Benchmark\n\n"
            "_No completed live IBM Quantum SRB JSON found yet in `outputs/hardware/`._\n"
        )

    report_path = REPORTS_DIR / "phase2_crosstalk_validation.md"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved Phase 2 report: {report_path}")


if __name__ == "__main__":
    main()
