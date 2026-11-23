"""Generate Phase 0 physics verification plots for T1, T2 Ramsey, and two-qubit ZZ dynamics.

Usage:
    python scripts/plot_physics_verification.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

from qtwin.physics import (
    SingleQubitParams,
    TwoQubitParams,
    simulate_ramsey_decay,
    simulate_t1_relaxation,
    simulate_zz_phase_accumulation,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots"


def plot_t1_relaxation(output_dir: Path) -> Path:
    """Plot numerical T1 relaxation against exact analytical exp(-t / T1)."""
    t1_us = 60.0
    params = SingleQubitParams(freq_mhz=0.0, t1_us=t1_us, t2_us=None)
    times_us = np.linspace(0.0, 250.0, 400)

    p1_num, _ = simulate_t1_relaxation(params, times_us)
    p1_exact = np.exp(-times_us / t1_us)
    max_err = float(np.max(np.abs(p1_num - p1_exact)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.plot(times_us, p1_exact, color="#1f2937", linewidth=2.2, label=r"Analytical $e^{-t/T_1}$")
    ax.plot(
        times_us[::12],
        p1_num[::12],
        "o",
        color="#2563eb",
        markersize=4.5,
        label="QuTiP Lindblad (mesolve)",
    )
    ax.set_title(f"Single-Qubit $T_1$ Relaxation ($T_1 = {t1_us:.0f}\\,\\mu$s)")
    ax.set_xlabel(r"Time $t$ ($\mu$s)")
    ax.set_ylabel(r"Excited-State Population $P_1(t)$")
    ax.text(
        0.97,
        0.60,
        f"Max abs error: {max_err:.2e}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f3f4f6", "edgecolor": "#d1d5db"},
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()

    out_path = output_dir / "t1_relaxation.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_t2_ramsey_decay(output_dir: Path) -> Path:
    """Plot numerical T2 Ramsey decay against exact analytical exp(-t / T2) cos(Delta t)."""
    t1_us = 80.0
    t2_us = 45.0
    detuning_mhz = 0.08  # 80 kHz detuning -> period 12.5 us
    params = SingleQubitParams(freq_mhz=detuning_mhz, t1_us=t1_us, t2_us=t2_us)
    times_us = np.linspace(0.0, 120.0, 600)

    sx_num, _, _ = simulate_ramsey_decay(params, times_us)
    envelope = np.exp(-times_us / t2_us)
    sx_exact = envelope * np.cos(2.0 * np.pi * detuning_mhz * times_us)
    max_err = float(np.max(np.abs(sx_num - sx_exact)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.plot(times_us, envelope, "--", color="#9ca3af", linewidth=1.2, label=r"$\pm e^{-t/T_2}$ envelope")
    ax.plot(times_us, -envelope, "--", color="#9ca3af", linewidth=1.2)
    ax.plot(
        times_us,
        sx_exact,
        color="#1f2937",
        linewidth=2.0,
        label=r"Analytical $e^{-t/T_2}\cos(\Delta t)$",
    )
    ax.plot(
        times_us[::10],
        sx_num[::10],
        "o",
        color="#dc2626",
        markersize=3.8,
        label="QuTiP Lindblad (mesolve)",
    )
    ax.set_title(
        f"Single-Qubit $T_2$ Ramsey Decay "
        f"($T_1={t1_us:.0f}\\,\\mu$s, $T_2={t2_us:.0f}\\,\\mu$s, $\\Delta/2\\pi={detuning_mhz*1e3:.0f}$ kHz)"
    )
    ax.set_xlabel(r"Time $t$ ($\mu$s)")
    ax.set_ylabel(r"Coherence $\langle \sigma_x(t) \rangle$")
    ax.text(
        0.97,
        0.08,
        f"Max abs error: {max_err:.2e}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f3f4f6", "edgecolor": "#d1d5db"},
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out_path = output_dir / "t2_ramsey_decay.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_two_qubit_zz_phase(output_dir: Path) -> Path:
    """Plot conditional Ramsey oscillations and differential ZZ phase accumulation."""
    zz_mhz = 0.15  # 150 kHz static ZZ coupling
    params = TwoQubitParams(
        qubit0=SingleQubitParams(freq_mhz=0.0, t1_us=100.0, t2_us=70.0),
        qubit1=SingleQubitParams(freq_mhz=0.0, t1_us=None, t2_us=None),
        zz_coupling_mhz=zz_mhz,
    )
    times_us = np.linspace(0.0, 20.0, 400)

    data = simulate_zz_phase_accumulation(params, times_us)
    exact_phase = 2.0 * np.pi * zz_mhz * times_us
    max_phase_err = float(np.max(np.abs(data["zz_phase_diff_rad"] - exact_phase)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.8), dpi=150, sharex=True)

    ax1.plot(
        times_us,
        data["sigma_x_q1_0"],
        color="#2563eb",
        linewidth=2.0,
        label=r"Spectator $Q_1 = |0\rangle$",
    )
    ax1.plot(
        times_us,
        data["sigma_x_q1_1"],
        color="#ea580c",
        linewidth=2.0,
        label=r"Spectator $Q_1 = |1\rangle$",
    )
    ax1.set_title(
        f"Two-Qubit Conditional Ramsey & $ZZ$ Phase Accumulation ($f_{{ZZ}} = {zz_mhz*1e3:.0f}$ kHz)"
    )
    ax1.set_ylabel(r"$\langle \sigma_x^{(0)}(t) \rangle$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", frameon=True)

    ax2.plot(
        times_us,
        exact_phase / np.pi,
        color="#1f2937",
        linewidth=2.2,
        label=r"Analytical $\zeta_{ZZ} t / \pi$",
    )
    ax2.plot(
        times_us[::12],
        data["zz_phase_diff_rad"][::12] / np.pi,
        "o",
        color="#059669",
        markersize=4.2,
        label="QuTiP Lindblad (unwrapped)",
    )
    ax2.set_xlabel(r"Time $t$ ($\mu$s)")
    ax2.set_ylabel(r"Accumulated $ZZ$ Phase ($\Delta\phi / \pi$)")
    ax2.text(
        0.03,
        0.82,
        f"Max phase error: {max_phase_err:.2e} rad",
        transform=ax2.transAxes,
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f3f4f6", "edgecolor": "#d1d5db"},
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    out_path = output_dir / "two_qubit_zz_phase.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    """Generate and save all Phase 0 verification plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_t1_relaxation(OUTPUT_DIR),
        plot_t2_ramsey_decay(OUTPUT_DIR),
        plot_two_qubit_zz_phase(OUTPUT_DIR),
    ]
    print("Saved Phase 0 physics verification plots:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
