"""Lindblad-to-Pauli-twirl noise bridge for Stim circuit-level QEC simulations.

Converts continuous-time open quantum system parameters calibrated in Phases 0–2
(T1, T2, residual gate error eps_res, gate/cycle durations, and extracted static
ZZ crosstalk f_ZZ) into strictly valid discrete single-qubit and two-qubit Pauli
channel probabilities for Stim.

Physics conventions:
1. Single-qubit Pauli Twirl Approximation (PTA) of amplitude damping (T1) + pure
   dephasing (T2) composed with residual depolarizing error eps_res over duration t:
       D_Z = exp(-t / T1) * (1 - 2 * eps_res)
       D_X = D_Y = exp(-t / T2) * (1 - 2 * eps_res)
       p_X = p_Y = (1 - D_Z) / 4
       p_Z = (1 + D_Z - 2 * D_X) / 4
   Guaranteed non-negative and sum <= 3/4 whenever 0 < T2 <= 2 * T1 and 0 <= eps_res <= 0.5.

2. Two-qubit Pauli twirl of coherent static ZZ crosstalk U_ZZ(theta) = exp(-i theta |11><11|)
   with theta = 2 * pi * f_ZZ * t_cycle:
       p_IZ = p_ZI = p_ZZ = sin^2(theta / 2) / 4
   Guaranteed non-negative with p_IZ + p_ZI + p_ZZ <= 3/4.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HardwareCalibrationInput:
    """Calibrated physical parameters extracted from Phase 1/2 hardware experiments."""

    t1_us: float
    t2_us: float
    gate_duration_ns: float
    cycle_duration_us: float = 0.25
    residual_error: float = 0.0
    zz_coupling_mhz: float = 0.0
    readout_error: float = 0.005
    source: str = "manual"


@dataclass(frozen=True)
class CircuitNoiseParams:
    """Discrete Stim-compatible Pauli error probabilities derived from Lindblad physics."""

    # Per-round idle/memory single-qubit Pauli channel on data qubits
    p_x: float
    p_y: float
    p_z: float
    # Two-qubit entangling gate (CX/CZ) depolarizing probability
    p_2q_depol: float
    # Measurement and reset bit-flip probabilities
    p_meas: float
    p_reset: float
    # Correlated two-qubit ZZ crosstalk channel on adjacent data qubits per cycle
    p_iz: float
    p_zi: float
    p_zz: float
    # Source physical metadata
    zz_coupling_mhz: float
    cycle_duration_us: float

    def __post_init__(self) -> None:
        for name in ("p_x", "p_y", "p_z", "p_2q_depol", "p_meas", "p_reset", "p_iz", "p_zi", "p_zz"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0) or not math.isfinite(val):
                raise ValueError(f"Invalid probability {name}={val}; must be finite in [0, 1].")
        if self.p_x + self.p_y + self.p_z > 1.0:
            raise ValueError(
                f"Single-qubit Pauli probabilities sum to {self.p_x + self.p_y + self.p_z} > 1."
            )
        if self.p_iz + self.p_zi + self.p_zz > 1.0:
            raise ValueError(
                f"Two-qubit ZZ Pauli probabilities sum to {self.p_iz + self.p_zi + self.p_zz} > 1."
            )

    @property
    def total_1q_error(self) -> float:
        """Total single-qubit Pauli error probability p_x + p_y + p_z."""
        return self.p_x + self.p_y + self.p_z

    @property
    def total_zz_error(self) -> float:
        """Total two-qubit ZZ-induced Pauli error probability p_iz + p_zi + p_zz."""
        return self.p_iz + self.p_zi + self.p_zz


def pauli_twirl_single_qubit(
    t1_us: float,
    t2_us: float,
    duration_us: float,
    residual_error: float = 0.0,
) -> tuple[float, float, float]:
    """Compute exact Pauli-twirled (p_X, p_Y, p_Z) for T1/T2 + residual depolarizing error."""
    if t1_us <= 0.0 or t2_us <= 0.0:
        raise ValueError(f"T1 and T2 must be positive, got T1={t1_us}, T2={t2_us}.")
    if t2_us > 2.0 * t1_us * (1.0 + 1e-12):
        raise ValueError(f"Physical bound violated: T2 ({t2_us}) > 2 * T1 ({2.0 * t1_us}).")
    if duration_us < 0.0:
        raise ValueError(f"duration_us must be non-negative, got {duration_us}.")
    if not (0.0 <= residual_error <= 0.5):
        raise ValueError(f"residual_error must be in [0, 0.5], got {residual_error}.")

    t2_eff = min(t2_us, 2.0 * t1_us)
    dep_factor = 1.0 - 2.0 * residual_error

    d_z = math.exp(-duration_us / t1_us) * dep_factor
    d_x = math.exp(-duration_us / t2_eff) * dep_factor

    p_x = max(0.0, 0.25 * (1.0 - d_z))
    p_y = p_x
    p_z = max(0.0, 0.25 * (1.0 + d_z - 2.0 * d_x))

    total = p_x + p_y + p_z
    if total > 1.0:
        scale = 1.0 / total
        p_x, p_y, p_z = p_x * scale, p_y * scale, p_z * scale

    return float(p_x), float(p_y), float(p_z)


def pauli_twirl_zz_crosstalk(
    zz_coupling_mhz: float,
    duration_us: float,
) -> tuple[float, float, float]:
    """Compute exact Pauli-twirled (p_IZ, p_ZI, p_ZZ) for U_ZZ = exp(-i * 2*pi*f_ZZ*t |11><11|)."""
    if zz_coupling_mhz < 0.0:
        raise ValueError(f"zz_coupling_mhz must be non-negative, got {zz_coupling_mhz}.")
    if duration_us < 0.0:
        raise ValueError(f"duration_us must be non-negative, got {duration_us}.")

    theta = 2.0 * math.pi * zz_coupling_mhz * duration_us
    p_component = 0.25 * (math.sin(0.5 * theta) ** 2)
    return float(p_component), float(p_component), float(p_component)


def lindblad_to_circuit_noise(cal: HardwareCalibrationInput) -> CircuitNoiseParams:
    """Map calibrated physical Lindblad + ZZ parameters to Stim circuit-level Pauli noise."""
    # Single-qubit data idle/memory Pauli error over one QEC stabilizer cycle
    p_x, p_y, p_z = pauli_twirl_single_qubit(
        t1_us=cal.t1_us,
        t2_us=cal.t2_us,
        duration_us=cal.cycle_duration_us,
        residual_error=cal.residual_error,
    )

    # Single-gate coherence + residual error to parameterize 2-qubit entangling gate noise
    gate_us = cal.gate_duration_ns * 1e-3
    gx, gy, gz = pauli_twirl_single_qubit(
        t1_us=cal.t1_us,
        t2_us=cal.t2_us,
        duration_us=gate_us,
        residual_error=cal.residual_error,
    )
    # 2q entangling gate error ~ 2x single-qubit gate error (both qubits participate)
    p_2q = float(np.clip(2.0 * (gx + gy + gz), 0.0, 0.75))

    p_iz, p_zi, p_zz = pauli_twirl_zz_crosstalk(
        zz_coupling_mhz=cal.zz_coupling_mhz,
        duration_us=cal.cycle_duration_us,
    )

    p_meas = float(np.clip(cal.readout_error, 0.0, 0.5))
    # Reset error typically comparable to ~0.5x readout error on superconducting hardware
    p_reset = float(np.clip(0.5 * cal.readout_error, 0.0, 0.5))

    return CircuitNoiseParams(
        p_x=p_x,
        p_y=p_y,
        p_z=p_z,
        p_2q_depol=p_2q,
        p_meas=p_meas,
        p_reset=p_reset,
        p_iz=p_iz,
        p_zi=p_zi,
        p_zz=p_zz,
        zz_coupling_mhz=cal.zz_coupling_mhz,
        cycle_duration_us=cal.cycle_duration_us,
    )


def load_ibm_fez_calibration(
    hardware_dir: Path | str = Path("outputs/hardware"),
    *,
    cycle_duration_us: float = 0.10,
    fitted_residual_error: float = 1.02e-3,
    extracted_zz_mhz: float = 0.2864,
) -> HardwareCalibrationInput:
    """Load `ibm_fez` calibration metadata from Phase 1/2 JSON outputs if available."""
    hw_path = Path(hardware_dir)
    t1_us = 56.77
    t2_us = 49.63
    sx_ns = 24.0
    readout_err = 0.005  # default fallback
    source = "ibm_fez_phase1_phase2_summary"

    if hw_path.exists():
        # Prefer Phase 2 SRB JSON (has both Q0 and Q1 calibration), else Phase 1 RB JSON
        srb_files = sorted(hw_path.glob("srb_2q_ibm_quantum_ibm_fez_*.json"), reverse=True)
        if srb_files:
            with srb_files[0].open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            cal_a = data.get("calibration_a", {})
            cal_b = data.get("calibration_b", {})
            # Pair-averaged coherence & readout for representative 2D grid parameters
            t1_vals = [v for v in (cal_a.get("t1_us"), cal_b.get("t1_us")) if v]
            t2_vals = [v for v in (cal_a.get("t2_us"), cal_b.get("t2_us")) if v]
            ro_vals = [v for v in (cal_a.get("readout_error"), cal_b.get("readout_error")) if v is not None]
            if t1_vals:
                t1_us = float(np.mean(t1_vals))
            if t2_vals:
                t2_us = float(np.mean(t2_vals))
            if cal_a.get("sx_duration_ns"):
                sx_ns = float(cal_a["sx_duration_ns"])
            if ro_vals:
                readout_err = float(np.mean(ro_vals))
            source = srb_files[0].name

    return HardwareCalibrationInput(
        t1_us=t1_us,
        t2_us=min(t2_us, 2.0 * t1_us),
        gate_duration_ns=sx_ns,
        cycle_duration_us=cycle_duration_us,
        residual_error=fitted_residual_error,
        zz_coupling_mhz=extracted_zz_mhz,
        readout_error=readout_err,
        source=source,
    )
