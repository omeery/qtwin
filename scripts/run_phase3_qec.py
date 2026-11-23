"""Phase 3 QEC Validation: Hardware-Calibrated Stim Surface Code + PyMatching Benchmark.

Loads our Phase 1/2 `ibm_fez` calibrated Lindblad + ZZ crosstalk parameters,
runs the 4-regime surface code distance scaling benchmark (d = 3, 5, 7) and
the f_ZZ crosstalk sweep (d = 3, 5), and saves:
- Publication-grade 2-panel plot: ``outputs/plots/phase3_qec_logical_error.png``
- Concise Markdown report:        ``outputs/reports/phase3_qec_impact.md``

Usage:
    python scripts/run_phase3_qec.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qtwin.qec import (
    QECBenchmarkPoint,
    compute_suppression_factor,
    lindblad_to_circuit_noise,
    load_ibm_fez_calibration,
    run_four_regime_benchmark,
    run_zz_crosstalk_sweep,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

REGIME_STYLE = {
    "ibm_reported_baseline": ("IBM-Reported $T_1/T_2$ Baseline", "#6b7280", "o--"),
    "uncorrelated_twin": ("Uncorrelated Digital Twin ($f_{ZZ}=0$)", "#2563eb", "s-"),
    "correlated_naive_decoder": ("Correlated HW + Naive Decoder", "#dc2626", "^-."),
    "correlated_noise_aware_decoder": ("Correlated HW + Noise-Aware Decoder", "#059669", "D-"),
}


def _plot_phase3(
    regime_pts: list[QECBenchmarkPoint],
    sweep_pts: list[QECBenchmarkPoint],
    hw_zz_mhz: float,
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=150)

    # Panel A: LER vs distance d across the 4 regimes
    for regime_key, (label, color, fmt) in REGIME_STYLE.items():
        pts = sorted([p for p in regime_pts if p.regime == regime_key], key=lambda p: p.distance)
        ds = [p.distance for p in pts]
        lers = [max(p.ler, 1e-6) for p in pts]
        errs = [p.ler_std for p in pts]
        ax1.errorbar(
            ds,
            lers,
            yerr=errs,
            fmt=fmt,
            color=color,
            capsize=3.5,
            linewidth=2.0,
            markersize=6.0,
            label=label,
        )

    ax1.set_yscale("log")
    ax1.set_xticks([3, 5, 7])
    ax1.set_xlabel("Rotated Surface Code Distance $d$ ($\\mathrm{rounds} = d$)")
    ax1.set_ylabel("Logical Error Rate (LER)")
    ax1.set_title("A. Surface Code Scaling Under `ibm_fez` Calibrated Noise")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="best", fontsize=8.5, frameon=True)

    # Panel B: LER vs f_ZZ crosstalk strength (Naive vs Noise-Aware at d=3 and d=5)
    sweep_styles = [
        (3, "naive_decoder", "$d=3$ Naive Decoder", "#ef4444", "o--"),
        (3, "noise_aware_decoder", "$d=3$ Noise-Aware Decoder", "#10b981", "o-"),
        (5, "naive_decoder", "$d=5$ Naive Decoder", "#991b1b", "s--"),
        (5, "noise_aware_decoder", "$d=5$ Noise-Aware Decoder", "#047857", "s-"),
    ]
    for d, reg, label, color, fmt in sweep_styles:
        pts = sorted(
            [p for p in sweep_pts if p.distance == d and p.regime == reg],
            key=lambda p: p.zz_coupling_mhz,
        )
        zzs = [p.zz_coupling_mhz for p in pts]
        lers = [max(p.ler, 1e-6) for p in pts]
        errs = [p.ler_std for p in pts]
        ax2.errorbar(
            zzs,
            lers,
            yerr=errs,
            fmt=fmt,
            color=color,
            capsize=3.0,
            linewidth=2.0,
            markersize=5.5,
            label=label,
        )

    ax2.axvline(
        hw_zz_mhz,
        color="#4b5563",
        linestyle=":",
        linewidth=1.5,
        label=f"`ibm_fez` extracted $f_{{ZZ}} = {hw_zz_mhz:.3f}$ MHz",
    )
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Correlated Crosstalk Strength $f_{ZZ}$ (MHz)")
    ax2.set_ylabel("Logical Error Rate (LER)")
    ax2.set_title("B. Crosstalk Sweep: Naive vs. Digital-Twin Noise-Aware Decoder")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=8.0, frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_report(
    cal_source: str,
    t1_us: float,
    t2_us: float,
    eps_res: float,
    zz_mhz: float,
    cycle_us: float,
    noise_corr: object,
    regime_pts: list[QECBenchmarkPoint],
    sweep_pts: list[QECBenchmarkPoint],
    out_path: Path,
) -> None:
    by_rd = {(p.regime, p.distance): p for p in regime_pts}

    lines = [
        "# Phase 3 QEC Validation Report — `ibm_fez` Digital Twin → Stim + PyMatching",
        "",
        "## 1. Calibrated Physical Inputs & Pauli-Twirled Circuit Noise",
        "",
        f"- **Calibration Source:** `{cal_source}` (`ibm_fez` pair `(0, 1)` + Phase 1/2 fits)",
        f"- **Physical Parameters:** $T_1 = {t1_us:.2f}\\,\\mu$s, $T_2 = {t2_us:.2f}\\,\\mu$s, "
        f"$\\epsilon_{{\\mathrm{{res}}}} = {eps_res:.2e}$, $f_{{ZZ}} = {zz_mhz:.4f}$ MHz, "
        f"$t_{{\\mathrm{{cycle}}}} = {cycle_us:.2f}\\,\\mu$s",
        f"- **Derived Circuit Pauli Noise:** $p_X = p_Y = {noise_corr.p_x:.3e}$, $p_Z = {noise_corr.p_z:.3e}$, "
        f"$p_{{2q}} = {noise_corr.p_2q_depol:.3e}$, $p_{{\\mathrm{{meas}}}} = {noise_corr.p_meas:.3e}$, "
        f"$p_{{IZ}} = p_{{ZI}} = p_{{ZZ}} = {noise_corr.p_zz:.3e}$",
        "",
        "## 2. Four-Regime Rotated Surface Code Benchmark ($d \\in \\{3, 5, 7\\}$, 60,000 shots/point)",
        "",
        "| Regime | LER ($d=3$) | LER ($d=5$) | LER ($d=7$) | $\\Lambda_{3\\to 5}$ | $\\Lambda_{5\\to 7}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key, (label, _, _) in REGIME_STYLE.items():
        p3 = by_rd[(key, 3)]
        p5 = by_rd[(key, 5)]
        p7 = by_rd[(key, 7)]
        lam35 = compute_suppression_factor(p3.ler, p5.ler)
        lam57 = compute_suppression_factor(p5.ler, p7.ler)
        lam35_s = f"{lam35:.2f}" if lam35 is not None else "N/A"
        lam57_s = f"{lam57:.2f}" if lam57 is not None else "N/A"
        clean_label = label.replace("$", "")
        lines.append(
            f"| **{clean_label}** | `{p3.ler:.4e} ± {p3.ler_std:.1e}` | "
            f"`{p5.ler:.4e} ± {p5.ler_std:.1e}` | `{p7.ler:.4e} ± {p7.ler_std:.1e}` | "
            f"`{lam35_s}` | `{lam57_s}` |"
        )

    # Relative LER reduction from Noise-Aware decoder at ibm_fez f_ZZ and across sweep
    lines.extend(
        [
            "",
            "## 3. Crosstalk Sweep: Naive vs. Digital-Twin Noise-Aware Decoder (60,000 shots/point)",
            "",
            "| $f_{ZZ}$ (MHz) | $d=3$ Naive LER | $d=3$ Noise-Aware LER | $d=3$ LER Reduction | $d=5$ Naive LER | $d=5$ Noise-Aware LER | $d=5$ LER Reduction |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    zz_vals = sorted({p.zz_coupling_mhz for p in sweep_pts})
    by_sw = {(p.distance, p.regime, p.zz_coupling_mhz): p for p in sweep_pts}
    for f_zz in zz_vals:
        n3 = by_sw[(3, "naive_decoder", f_zz)]
        a3 = by_sw[(3, "noise_aware_decoder", f_zz)]
        n5 = by_sw[(5, "naive_decoder", f_zz)]
        a5 = by_sw[(5, "noise_aware_decoder", f_zz)]
        red3 = 100.0 * (n3.ler - a3.ler) / n3.ler if n3.ler > 0 else 0.0
        red5 = 100.0 * (n5.ler - a5.ler) / n5.ler if n5.ler > 0 else 0.0
        lines.append(
            f"| `{f_zz:.4f}` | `{n3.ler:.4e}` | `{a3.ler:.4e}` | `{red3:+.1f}%` | "
            f"`{n5.ler:.4e}` | `{a5.ler:.4e}` | `{red5:+.1f}%` |"
        )

    lines.extend(
        [
            "",
            "![Phase 3 QEC Logical Error Benchmark](../plots/phase3_qec_logical_error.png)",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    cal = load_ibm_fez_calibration(ROOT_DIR / "outputs" / "hardware")
    noise_corr = lindblad_to_circuit_noise(cal)

    shots = 60_000
    regime_pts = run_four_regime_benchmark(cal, distances=(3, 5, 7), shots=shots, seed=42)
    sweep_pts = run_zz_crosstalk_sweep(
        cal,
        zz_values_mhz=(0.0, 0.1, 0.2, 0.2864, 0.5, 0.8),
        distances=(3, 5),
        shots=shots,
        seed=1000,
    )

    plot_path = PLOTS_DIR / "phase3_qec_logical_error.png"
    report_path = REPORTS_DIR / "phase3_qec_impact.md"
    _plot_phase3(regime_pts, sweep_pts, cal.zz_coupling_mhz, plot_path)
    _write_report(
        cal.source,
        cal.t1_us,
        cal.t2_us,
        cal.residual_error,
        cal.zz_coupling_mhz,
        cal.cycle_duration_us,
        noise_corr,
        regime_pts,
        sweep_pts,
        report_path,
    )
    elapsed = time.perf_counter() - t0

    print(f"Phase 3 QEC benchmark completed in {elapsed:.2f}s (shots={shots}/point)")
    for pt in regime_pts:
        print(
            f"  [{pt.regime:30s}] d={pt.distance} | LER={pt.ler:.4e} +/- {pt.ler_std:.1e} ({pt.num_logical_errors}/{pt.shots})"
        )
    print(f"Saved plot:   {plot_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
