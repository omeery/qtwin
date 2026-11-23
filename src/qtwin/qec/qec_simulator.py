"""Stim rotated surface code memory benchmark with PyMatching MWPM decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pymatching
import stim

from qtwin.qec.noise_bridge import (
    CircuitNoiseParams,
    HardwareCalibrationInput,
    lindblad_to_circuit_noise,
)


@dataclass(frozen=True)
class QECBenchmarkPoint:
    regime: str
    distance: int
    rounds: int
    zz_coupling_mhz: float
    shots: int
    num_logical_errors: int
    ler: float
    ler_std: float


def _extract_data_qubits_and_neighbors(
    noiseless_circuit: stim.Circuit,
) -> tuple[list[int], list[tuple[int, int]]]:
    coords = noiseless_circuit.get_final_qubit_coordinates()
    # Data qubits sit on odd-odd coordinates in Stim's rotated_memory_x layout
    data_qubits = sorted(
        q for q, xy in coords.items() if int(round(xy[0])) % 2 == 1 and int(round(xy[1])) % 2 == 1
    )

    pairs: list[tuple[int, int]] = []
    for idx_i, qi in enumerate(data_qubits):
        xi, yi = coords[qi][0], coords[qi][1]
        for qj in data_qubits[idx_i + 1 :]:
            xj, yj = coords[qj][0], coords[qj][1]
            if abs(xi - xj) + abs(yi - yj) == 2.0:
                pairs.append((qi, qj))
    return data_qubits, pairs


def _inject_noise_into_block(
    block: stim.Circuit,
    noise: CircuitNoiseParams,
    data_qubits: list[int],
    *,
    is_repeat_body: bool = False,
) -> stim.Circuit:
    out = stim.Circuit()
    zz_probs = [0.0] * 15
    # Stim PAULI_CHANNEL_2 order: IX,IY,IZ, XI,XX,XY,XZ, YI,YX,YY,YZ, ZI,ZX,ZY,ZZ
    zz_probs[2] = noise.p_iz
    zz_probs[11] = noise.p_zi
    zz_probs[14] = noise.p_zz
    has_zz = noise.total_zz_error > 0.0
    has_1q = noise.total_1q_error > 0.0

    if is_repeat_body and has_1q and data_qubits:
        out.append("PAULI_CHANNEL_1", data_qubits, [noise.p_x, noise.p_y, noise.p_z])

    for inst in block:
        if isinstance(inst, stim.CircuitRepeatBlock):
            noisy_body = _inject_noise_into_block(
                inst.body_copy(),
                noise,
                data_qubits,
                is_repeat_body=True,
            )
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, noisy_body))
            continue

        name = inst.name
        targets = inst.targets_copy()
        args = inst.gate_args_copy()

        if name in ("M", "MR") and noise.p_meas > 0.0:
            qubits = [t.value for t in targets if t.is_qubit_target]
            if qubits:
                out.append("X_ERROR", qubits, noise.p_meas)
        elif name in ("MX", "MRX") and noise.p_meas > 0.0:
            qubits = [t.value for t in targets if t.is_qubit_target]
            if qubits:
                out.append("Z_ERROR", qubits, noise.p_meas)

        out.append(name, targets, args)

        if name in ("CX", "CY", "CZ", "CNOT"):
            qubits = [t.value for t in targets if t.is_qubit_target]
            if qubits and noise.p_2q_depol > 0.0:
                out.append("DEPOLARIZE2", qubits, noise.p_2q_depol)
            if qubits and has_zz:
                out.append("PAULI_CHANNEL_2", qubits, zz_probs)

        if name in ("R", "MR") and noise.p_reset > 0.0:
            qubits = [t.value for t in targets if t.is_qubit_target]
            if qubits:
                out.append("X_ERROR", qubits, noise.p_reset)
        elif name in ("RX", "MRX") and noise.p_reset > 0.0:
            qubits = [t.value for t in targets if t.is_qubit_target]
            if qubits:
                out.append("Z_ERROR", qubits, noise.p_reset)

    return out


def build_surface_code_circuit(
    distance: int,
    noise: CircuitNoiseParams,
    *,
    rounds: int | None = None,
) -> stim.Circuit:
    """Build a rotated surface code X-memory circuit with calibrated Lindblad + ZZ noise."""
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"Surface code distance must be an odd integer >= 3, got {distance}.")
    n_rounds = distance if rounds is None else rounds

    base = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=n_rounds,
    )
    data_qubits, _ = _extract_data_qubits_and_neighbors(base)

    noisy = _inject_noise_into_block(base, noise, data_qubits, is_repeat_body=False)

    # Round 1 sits outside the REPEAT block in Stim's generated circuit
    final = stim.Circuit()
    injected_first_round = False
    for inst in noisy:
        if not injected_first_round and not isinstance(inst, stim.CircuitRepeatBlock) and inst.name == "TICK":
            final.append(inst)
            if noise.total_1q_error > 0.0 and data_qubits:
                final.append("PAULI_CHANNEL_1", data_qubits, [noise.p_x, noise.p_y, noise.p_z])
            injected_first_round = True
            continue
        final.append(inst)

    return final


def evaluate_surface_code_memory(
    sample_circuit: stim.Circuit,
    *,
    decoder_circuit: stim.Circuit | None = None,
    shots: int = 50_000,
    seed: int = 42,
    regime: str = "custom",
    distance: int = 3,
    rounds: int = 3,
    zz_coupling_mhz: float = 0.0,
) -> QECBenchmarkPoint:
    """Sample syndromes from sample_circuit and decode with PyMatching MWPM."""
    dec_circ = sample_circuit if decoder_circuit is None else decoder_circuit
    dem = dec_circ.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    matcher = pymatching.Matching.from_detector_error_model(dem)

    sampler = sample_circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    preds = matcher.decode_batch(dets)

    errors = int(np.sum(np.any(preds != obs, axis=1)))
    ler = float(errors / shots)
    ler_std = float(math.sqrt(max(ler * (1.0 - ler), 0.0) / shots))

    return QECBenchmarkPoint(
        regime=regime,
        distance=distance,
        rounds=rounds,
        zz_coupling_mhz=zz_coupling_mhz,
        shots=shots,
        num_logical_errors=errors,
        ler=ler,
        ler_std=ler_std,
    )


def compute_suppression_factor(ler_d1: float, ler_d2: float) -> float | None:
    if ler_d2 <= 0.0 or ler_d1 <= 0.0:
        return None
    return float(ler_d1 / ler_d2)


def run_four_regime_benchmark(
    cal: HardwareCalibrationInput,
    *,
    distances: tuple[int, ...] = (3, 5, 7),
    shots: int = 50_000,
    seed: int = 42,
) -> list[QECBenchmarkPoint]:
    """Compare IBM-reported, uncorrelated, naive-decoded, and noise-aware QEC regimes."""
    results: list[QECBenchmarkPoint] = []

    cal_ibm_baseline = HardwareCalibrationInput(
        t1_us=cal.t1_us,
        t2_us=cal.t2_us,
        gate_duration_ns=cal.gate_duration_ns,
        cycle_duration_us=cal.cycle_duration_us,
        residual_error=0.0,
        zz_coupling_mhz=0.0,
        readout_error=cal.readout_error,
        source=cal.source,
    )
    cal_uncorr_twin = HardwareCalibrationInput(
        t1_us=cal.t1_us,
        t2_us=cal.t2_us,
        gate_duration_ns=cal.gate_duration_ns,
        cycle_duration_us=cal.cycle_duration_us,
        residual_error=cal.residual_error,
        zz_coupling_mhz=0.0,
        readout_error=cal.readout_error,
        source=cal.source,
    )

    noise_baseline = lindblad_to_circuit_noise(cal_ibm_baseline)
    noise_uncorr = lindblad_to_circuit_noise(cal_uncorr_twin)
    noise_corr = lindblad_to_circuit_noise(cal)

    for d in distances:
        circ_baseline = build_surface_code_circuit(d, noise_baseline, rounds=d)
        circ_uncorr = build_surface_code_circuit(d, noise_uncorr, rounds=d)
        circ_corr = build_surface_code_circuit(d, noise_corr, rounds=d)

        results.append(
            evaluate_surface_code_memory(
                circ_baseline,
                decoder_circuit=circ_baseline,
                shots=shots,
                seed=seed + d,
                regime="ibm_reported_baseline",
                distance=d,
                rounds=d,
                zz_coupling_mhz=0.0,
            )
        )
        results.append(
            evaluate_surface_code_memory(
                circ_uncorr,
                decoder_circuit=circ_uncorr,
                shots=shots,
                seed=seed + 100 + d,
                regime="uncorrelated_twin",
                distance=d,
                rounds=d,
                zz_coupling_mhz=0.0,
            )
        )
        # Same seed so naive and noise-aware decoders see identical syndrome samples
        results.append(
            evaluate_surface_code_memory(
                circ_corr,
                decoder_circuit=circ_uncorr,
                shots=shots,
                seed=seed + 200 + d,
                regime="correlated_naive_decoder",
                distance=d,
                rounds=d,
                zz_coupling_mhz=cal.zz_coupling_mhz,
            )
        )
        results.append(
            evaluate_surface_code_memory(
                circ_corr,
                decoder_circuit=circ_corr,
                shots=shots,
                seed=seed + 200 + d,
                regime="correlated_noise_aware_decoder",
                distance=d,
                rounds=d,
                zz_coupling_mhz=cal.zz_coupling_mhz,
            )
        )

    return results


def run_zz_crosstalk_sweep(
    cal: HardwareCalibrationInput,
    *,
    zz_values_mhz: tuple[float, ...] = (0.0, 0.1, 0.2, 0.2864, 0.5, 0.8),
    distances: tuple[int, ...] = (3, 5),
    shots: int = 50_000,
    seed: int = 1000,
) -> list[QECBenchmarkPoint]:
    """Sweep f_ZZ comparing naive vs. noise-aware PyMatching decoders."""
    results: list[QECBenchmarkPoint] = []

    cal_uncorr = HardwareCalibrationInput(
        t1_us=cal.t1_us,
        t2_us=cal.t2_us,
        gate_duration_ns=cal.gate_duration_ns,
        cycle_duration_us=cal.cycle_duration_us,
        residual_error=cal.residual_error,
        zz_coupling_mhz=0.0,
        readout_error=cal.readout_error,
        source=cal.source,
    )
    noise_uncorr = lindblad_to_circuit_noise(cal_uncorr)

    for d in distances:
        circ_uncorr = build_surface_code_circuit(d, noise_uncorr, rounds=d)
        for idx, f_zz in enumerate(zz_values_mhz):
            cal_zz = HardwareCalibrationInput(
                t1_us=cal.t1_us,
                t2_us=cal.t2_us,
                gate_duration_ns=cal.gate_duration_ns,
                cycle_duration_us=cal.cycle_duration_us,
                residual_error=cal.residual_error,
                zz_coupling_mhz=f_zz,
                readout_error=cal.readout_error,
                source=cal.source,
            )
            noise_zz = lindblad_to_circuit_noise(cal_zz)
            circ_zz = build_surface_code_circuit(d, noise_zz, rounds=d)
            point_seed = seed + 50 * d + idx

            results.append(
                evaluate_surface_code_memory(
                    circ_zz,
                    decoder_circuit=circ_uncorr,
                    shots=shots,
                    seed=point_seed,
                    regime="naive_decoder",
                    distance=d,
                    rounds=d,
                    zz_coupling_mhz=f_zz,
                )
            )
            results.append(
                evaluate_surface_code_memory(
                    circ_zz,
                    decoder_circuit=circ_zz,
                    shots=shots,
                    seed=point_seed,
                    regime="noise_aware_decoder",
                    distance=d,
                    rounds=d,
                    zz_coupling_mhz=f_zz,
                )
            )

    return results
