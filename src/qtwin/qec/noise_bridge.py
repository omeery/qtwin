"""Lindblad-to-Pauli twirl bridge mapping continuous-time noise to Stim Pauli channels."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HardwareCalibrationInput:
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
    p_x: float
    p_y: float
    p_z: float
    p_2q_depol: float
    p_meas: float
    p_reset: float
    p_iz: float
    p_zi: float
    p_zz: float
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
        return self.p_x + self.p_y + self.p_z

    @property
    def total_zz_error(self) -> float:
        return self.p_iz + self.p_zi + self.p_zz


def pauli_twirl_single_qubit(
    t1_us: float,
    t2_us: float,
    duration_us: float,
    residual_error: float = 0.0,
) -> tuple[float, float, float]:
    """Pauli-twirled (p_X, p_Y, p_Z) for T1/T2 relaxation composed with residual_error."""
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
    """Pauli-twirled (p_IZ, p_ZI, p_ZZ) for U_ZZ = exp(-i * 2*pi*f_ZZ*t |11><11|)."""
    if zz_coupling_mhz < 0.0:
        raise ValueError(f"zz_coupling_mhz must be non-negative, got {zz_coupling_mhz}.")
    if duration_us < 0.0:
        raise ValueError(f"duration_us must be non-negative, got {duration_us}.")

    theta = 2.0 * math.pi * zz_coupling_mhz * duration_us
    p_component = 0.25 * (math.sin(0.5 * theta) ** 2)
    return float(p_component), float(p_component), float(p_component)


def lindblad_to_circuit_noise(cal: HardwareCalibrationInput) -> CircuitNoiseParams:
    """Convert calibrated Lindblad + ZZ parameters to discrete Stim circuit Pauli noise."""
    p_x, p_y, p_z = pauli_twirl_single_qubit(
        t1_us=cal.t1_us,
        t2_us=cal.t2_us,
        duration_us=cal.cycle_duration_us,
        residual_error=cal.residual_error,
    )

    gate_us = cal.gate_duration_ns * 1e-3
    gx, gy, gz = pauli_twirl_single_qubit(
        t1_us=cal.t1_us,
        t2_us=cal.t2_us,
        duration_us=gate_us,
        residual_error=cal.residual_error,
    )
    p_2q = float(np.clip(2.0 * (gx + gy + gz), 0.0, 0.75))

    p_iz, p_zi, p_zz = pauli_twirl_zz_crosstalk(
        zz_coupling_mhz=cal.zz_coupling_mhz,
        duration_us=cal.cycle_duration_us,
    )

    p_meas = float(np.clip(cal.readout_error, 0.0, 0.5))
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
    """Load ibm_fez calibration metadata from Phase 2 SRB JSON if present."""
    hw_path = Path(hardware_dir)
    t1_us = 56.77
    t2_us = 49.63
    sx_ns = 24.0
    readout_err = 0.005
    source = "ibm_fez_phase1_phase2_summary"

    if hw_path.exists():
        srb_files = sorted(hw_path.glob("srb_2q_ibm_quantum_ibm_fez_*.json"), reverse=True)
        if srb_files:
            with srb_files[0].open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            cal_a = data.get("calibration_a", {})
            cal_b = data.get("calibration_b", {})
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
