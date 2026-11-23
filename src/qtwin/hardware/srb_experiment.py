"""2-qubit Isolated + Simultaneous RB runner for IBM Quantum and local Aer ZZ dry-run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import random_clifford
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from scipy.optimize import curve_fit

from qtwin.hardware.rb_experiment import (
    PHYSICAL_1Q_GATES,
    QubitCalibrationMetadata,
    connect_ibm_backend,
    extract_qubit_calibration,
)


@dataclass(frozen=True)
class SRBConfig:
    """Configuration for 2-qubit Isolated + Simultaneous RB."""

    qubit_pair: tuple[int, int] | None = None  # None -> auto-select connected pair
    lengths: tuple[int, ...] = (1, 20, 50, 100, 200, 350, 500)
    num_samples: int = 5
    shots: int = 1024
    seed: int = 42
    dry_run: bool = False
    backend_name: str | None = None
    timeout_s: float = 300.0
    # Synthetic Aer dry-run ground-truth noise parameters
    dry_run_t1_a_us: float = 80.0
    dry_run_t2_a_us: float = 60.0
    dry_run_t1_b_us: float = 75.0
    dry_run_t2_b_us: float = 55.0
    dry_run_gate_time_ns: float = 35.0
    dry_run_zz_mhz: float = 0.25
    output_dir: Path = Path("outputs/hardware")


@dataclass(frozen=True)
class DecaySummary:
    """Standard exponential RB fit P_0(m) = A * alpha^m + B."""

    alpha: float
    alpha_std: float | None
    epc: float
    epc_std: float | None
    amplitude: float
    offset: float


@dataclass(frozen=True)
class SRBResult:
    """Complete Isolated + Simultaneous RB output and pair calibration metadata."""

    mode: str
    backend_name: str
    qubit_pair: list[int]
    lengths: list[int]
    num_samples: int
    shots: int
    # Raw survival curves (means and 1-sigma SEMs)
    p0_iso_a: list[float]
    p0_iso_a_std: list[float]
    p0_iso_b: list[float]
    p0_iso_b_std: list[float]
    p0_sim_a: list[float]
    p0_sim_a_std: list[float]
    p0_sim_b: list[float]
    p0_sim_b_std: list[float]
    p00_sim: list[float]
    p00_sim_std: list[float]
    # Fitted RB summaries
    iso_a: DecaySummary
    iso_b: DecaySummary
    sim_a: DecaySummary
    sim_b: DecaySummary
    # Crosstalk deltas (EPC_sim - EPC_iso)
    delta_epc_a: float
    delta_epc_b: float
    avg_physical_gates_per_clifford: float
    calibration_a: QubitCalibrationMetadata
    calibration_b: QubitCalibrationMetadata
    synthetic_zz_mhz: float | None
    job_ids: list[str]
    timestamp_utc: str
    output_file: str


def select_connected_qubit_pair(
    backend: Any,
    requested_pair: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Select a physically connected qubit pair with valid calibration metadata."""
    edges = [tuple(e) for e in backend.coupling_map.get_edges()]
    edge_set = set(edges)

    if requested_pair is not None:
        qa, qb = requested_pair
        if (qa, qb) in edge_set or (qb, qa) in edge_set:
            return (qa, qb)
        raise ValueError(f"Requested pair {requested_pair} is not connected on {backend.name}.")

    # Prefer (0, 1) if connected and calibrated
    candidate_order: list[tuple[int, int]] = []
    if (0, 1) in edge_set or (1, 0) in edge_set:
        candidate_order.append((0, 1))
    for u, v in edges:
        pair = (min(u, v), max(u, v))
        if pair not in candidate_order:
            candidate_order.append(pair)

    props = backend.properties()
    for qa, qb in candidate_order:
        if props is None:
            return (qa, qb)
        try:
            t1a, t2a = props.t1(qa), props.t2(qa)
            t1b, t2b = props.t1(qb), props.t2(qb)
            if t1a and t2a and t1b and t2b:
                return (qa, qb)
        except Exception:
            continue

    raise RuntimeError(f"No calibrated connected qubit pair found on {backend.name}.")


def build_two_qubit_aer_simulator(config: SRBConfig) -> AerSimulator:
    """Build a local 2-qubit AerSimulator with independent T1/T2 thermal relaxation."""
    gate_time_s = config.dry_run_gate_time_ns * 1e-9
    err_a = thermal_relaxation_error(
        config.dry_run_t1_a_us * 1e-6, config.dry_run_t2_a_us * 1e-6, gate_time_s
    )
    err_b = thermal_relaxation_error(
        config.dry_run_t1_b_us * 1e-6, config.dry_run_t2_b_us * 1e-6, gate_time_s
    )

    noise_model = NoiseModel()
    noisy_gates = ["sx", "x", "h", "u1", "u2", "u3", "id"]
    noise_model.add_quantum_error(err_a, noisy_gates, [0])
    noise_model.add_quantum_error(err_b, noisy_gates, [1])

    # Include 'cp' in basis_gates without adding thermal error to it (it is the coherent ZZ channel)
    return AerSimulator(
        noise_model=noise_model,
        basis_gates=["id", "rz", "sx", "x", "cp"],
    )


def _clifford_to_native(clifford: Any) -> QuantumCircuit:
    """Convert a 1-qubit Clifford to {rz, sx, x} native gates."""
    return transpile(clifford.to_circuit(), basis_gates=["rz", "sx", "x"], optimization_level=1)


def _build_srb_circuits(
    lengths: list[int],
    num_samples: int,
    seed: int,
    *,
    zz_phase_per_clifford_rad: float = 0.0,
) -> tuple[list[QuantumCircuit], list[dict[str, Any]], float]:
    """Generate Isolated-A, Isolated-B, and Simultaneous-(A,B) 1-qubit RB circuits.

    Each circuit uses a 2-qubit virtual register ``(0, 1)`` (later mapped to physical
    ``(q_a, q_b)`` via ``initial_layout``) so that isolated and simultaneous circuits
    have identical structure and execute in a single SamplerV2 batch.

    If ``zz_phase_per_clifford_rad != 0``, appends ``cp(-zz_phase_per_clifford_rad, 0, 1)``
    after each Clifford layer to model exact static ZZ crosstalk ``H_ZZ = zeta_ZZ |11><11|``
    in local Aer dry-run mode (note: acts as identity when spectator qubit is in |0>).
    """
    rng = np.random.default_rng(seed)
    circuits: list[QuantumCircuit] = []
    meta_list: list[dict[str, Any]] = []

    total_phys_gates = 0
    total_cliffords = 0

    for length in lengths:
        for sample_idx in range(num_samples):
            # Pre-generate independent random 1-qubit Clifford sequences for A and B
            seq_a = [random_clifford(1, seed=int(rng.integers(0, 2**31 - 1))) for _ in range(length)]
            seq_b = [random_clifford(1, seed=int(rng.integers(0, 2**31 - 1))) for _ in range(length)]

            acc_a = seq_a[0]
            for c in seq_a[1:]:
                acc_a = acc_a.compose(c)
            inv_a = acc_a.adjoint()

            acc_b = seq_b[0]
            for c in seq_b[1:]:
                acc_b = acc_b.compose(c)
            inv_b = acc_b.adjoint()

            gates_a = [_clifford_to_native(c) for c in seq_a + [inv_a]]
            gates_b = [_clifford_to_native(c) for c in seq_b + [inv_b]]

            for g_circ in gates_a:
                ops = g_circ.count_ops()
                total_phys_gates += sum(int(ops.get(g, 0)) for g in PHYSICAL_1Q_GATES)
                total_cliffords += 1

            # 1) Isolated RB on qubit A (qubit B stays idle in |0>)
            qr_a = QuantumRegister(2, "q")
            cr_a = ClassicalRegister(1, "c")
            qc_iso_a = QuantumCircuit(qr_a, cr_a, name=f"isoA_m{length}_s{sample_idx}")
            for g_circ in gates_a:
                qc_iso_a.compose(g_circ, qubits=[0], inplace=True)
                if zz_phase_per_clifford_rad != 0.0:
                    qc_iso_a.cp(-zz_phase_per_clifford_rad, 0, 1)
                qc_iso_a.barrier(0, 1)
            qc_iso_a.measure(0, 0)
            circuits.append(qc_iso_a)
            meta_list.append({"protocol": "iso_a", "length": length, "sample": sample_idx})

            # 2) Isolated RB on qubit B (qubit A stays idle in |0>)
            qr_b = QuantumRegister(2, "q")
            cr_b = ClassicalRegister(1, "c")
            qc_iso_b = QuantumCircuit(qr_b, cr_b, name=f"isoB_m{length}_s{sample_idx}")
            for g_circ in gates_b:
                qc_iso_b.compose(g_circ, qubits=[1], inplace=True)
                if zz_phase_per_clifford_rad != 0.0:
                    qc_iso_b.cp(-zz_phase_per_clifford_rad, 0, 1)
                qc_iso_b.barrier(0, 1)
            qc_iso_b.measure(1, 0)
            circuits.append(qc_iso_b)
            meta_list.append({"protocol": "iso_b", "length": length, "sample": sample_idx})

            # 3) Simultaneous RB on (A, B)
            qr_s = QuantumRegister(2, "q")
            cr_s = ClassicalRegister(2, "c")
            qc_sim = QuantumCircuit(qr_s, cr_s, name=f"simAB_m{length}_s{sample_idx}")
            for g_a, g_b in zip(gates_a, gates_b):
                qc_sim.compose(g_a, qubits=[0], inplace=True)
                qc_sim.compose(g_b, qubits=[1], inplace=True)
                if zz_phase_per_clifford_rad != 0.0:
                    qc_sim.cp(-zz_phase_per_clifford_rad, 0, 1)
                qc_sim.barrier(0, 1)
            qc_sim.measure(0, 0)
            qc_sim.measure(1, 1)
            circuits.append(qc_sim)
            meta_list.append({"protocol": "sim_ab", "length": length, "sample": sample_idx})

    avg_phys_gates = float(total_phys_gates / max(total_cliffords, 1))
    return circuits, meta_list, avg_phys_gates


def _fit_exponential_rb(
    lengths: list[int],
    p0_mean: list[float],
    p0_std: list[float],
) -> DecaySummary:
    """Fit standard RB decay P_0(m) = A * alpha^m + B via weighted curve_fit."""
    m = np.asarray(lengths, dtype=float)
    y = np.asarray(p0_mean, dtype=float)
    sigma = np.maximum(np.asarray(p0_std, dtype=float), 1e-6)

    def _model(x: np.ndarray, amp: float, alpha: float, offset: float) -> np.ndarray:
        return amp * np.power(alpha, x) + offset

    p0_guess = [float(np.clip(y[0] - 0.5, 0.1, 0.55)), 0.998, 0.5]
    popt, pcov = curve_fit(
        _model,
        m,
        y,
        p0=p0_guess,
        sigma=sigma,
        absolute_sigma=True,
        bounds=([0.05, 0.5, 0.2], [0.6, 1.0, 0.8]),
        maxfev=10000,
    )
    amp, alpha, offset = float(popt[0]), float(popt[1]), float(popt[2])
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0)) if np.all(np.isfinite(pcov)) else None
    alpha_std = float(perr[1]) if perr is not None else None
    epc = 0.5 * (1.0 - alpha)
    epc_std = 0.5 * alpha_std if alpha_std is not None else None
    return DecaySummary(
        alpha=alpha,
        alpha_std=alpha_std,
        epc=epc,
        epc_std=epc_std,
        amplitude=amp,
        offset=offset,
    )


def _aggregate_curve(
    samples_by_length: dict[int, list[float]],
    lengths: list[int],
    shots: int,
) -> tuple[list[float], list[float]]:
    """Compute mean and SEM (floored by binomial shot noise) across RB seeds."""
    means: list[float] = []
    stds: list[float] = []
    for length in lengths:
        arr = np.asarray(samples_by_length[length], dtype=float)
        mean_val = float(np.mean(arr))
        n = len(arr)
        sem_emp = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        sem_shot = float(np.sqrt(max(mean_val * (1.0 - mean_val), 1e-6) / (shots * n)))
        means.append(mean_val)
        stds.append(max(sem_emp, sem_shot))
    return means, stds


def run_simultaneous_rb(config: SRBConfig) -> SRBResult:
    """Execute 2-qubit Isolated + Simultaneous RB on Aer dry-run or live IBM QPU."""
    lengths = list(config.lengths)

    if config.dry_run:
        backend = build_two_qubit_aer_simulator(config)
        qubit_pair = (0, 1)
        mode = "dry_run_aer"
        backend_label = (
            f"aer_zz({config.dry_run_zz_mhz:.2f}MHz,"
            f"T1={config.dry_run_t1_a_us:.0f}/{config.dry_run_t1_b_us:.0f}us)"
        )
        cal_a = QubitCalibrationMetadata(
            t1_us=config.dry_run_t1_a_us,
            t2_us=config.dry_run_t2_a_us,
            sx_duration_ns=config.dry_run_gate_time_ns,
            readout_error=0.0,
        )
        cal_b = QubitCalibrationMetadata(
            t1_us=config.dry_run_t1_b_us,
            t2_us=config.dry_run_t2_b_us,
            sx_duration_ns=config.dry_run_gate_time_ns,
            readout_error=0.0,
        )
        # First pass to measure exact average physical gates per Clifford
        _, _, avg_phys_est = _build_srb_circuits(lengths, config.num_samples, config.seed)
        t_cliff_us = avg_phys_est * (config.dry_run_gate_time_ns * 1e-3)
        zz_phase_rad = float(2.0 * np.pi * config.dry_run_zz_mhz * t_cliff_us)
        synthetic_zz = config.dry_run_zz_mhz
    else:
        backend = connect_ibm_backend(config.backend_name)
        qubit_pair = select_connected_qubit_pair(backend, config.qubit_pair)
        mode = "ibm_quantum"
        backend_label = getattr(backend, "name", str(backend))
        cal_a = extract_qubit_calibration(backend, qubit_pair[0])
        cal_b = extract_qubit_calibration(backend, qubit_pair[1])
        zz_phase_rad = 0.0
        synthetic_zz = None

    circuits, meta_list, avg_phys_gates = _build_srb_circuits(
        lengths,
        config.num_samples,
        config.seed,
        zz_phase_per_clifford_rad=zz_phase_rad,
    )

    transpiled = transpile(
        circuits,
        backend=backend,
        initial_layout=list(qubit_pair),
        optimization_level=0,  # Preserve barriers & Clifford layer structure
    )

    counts_list: list[dict[str, int]] = []
    job_ids: list[str] = []

    if config.dry_run:
        job = backend.run(transpiled, shots=config.shots, seed_simulator=config.seed)
        job_ids.append(str(job.job_id()))
        sim_res = job.result()
        for idx in range(len(transpiled)):
            counts_list.append(dict(sim_res.get_counts(idx)))
    else:
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=backend)
        job = sampler.run(transpiled, shots=config.shots)
        job_id = str(job.job_id())
        job_ids.append(job_id)

        try:
            pub_results = job.result(timeout=config.timeout_s)
        except Exception as exc:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            config.output_dir.mkdir(parents=True, exist_ok=True)
            pending_file = (
                config.output_dir
                / f"pending_srb_{backend_label}_q{qubit_pair[0]}_{qubit_pair[1]}_{timestamp}.json"
            )
            with pending_file.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "backend_name": backend_label,
                        "qubit_pair": list(qubit_pair),
                        "lengths": lengths,
                        "num_samples": config.num_samples,
                        "shots": config.shots,
                        "seed": config.seed,
                        "avg_physical_gates_per_clifford": avg_phys_gates,
                        "calibration_a": asdict(cal_a),
                        "calibration_b": asdict(cal_b),
                        "job_ids": job_ids,
                        "meta_list": meta_list,
                        "timestamp_utc": timestamp,
                        "error": str(exc),
                    },
                    fh,
                    indent=2,
                )
            raise RuntimeError(
                f"SRB job {job_id} did not complete within {config.timeout_s}s. "
                f"Saved pending metadata to {pending_file}."
            ) from exc

        for pub_res in pub_results:
            counts_list.append(dict(pub_res.data.c.get_counts()))

    # Parse survival probabilities from counts
    iso_a_samples: dict[int, list[float]] = {m: [] for m in lengths}
    iso_b_samples: dict[int, list[float]] = {m: [] for m in lengths}
    sim_a_samples: dict[int, list[float]] = {m: [] for m in lengths}
    sim_b_samples: dict[int, list[float]] = {m: [] for m in lengths}
    sim_00_samples: dict[int, list[float]] = {m: [] for m in lengths}

    for meta, counts in zip(meta_list, counts_list):
        total = float(sum(counts.values()))
        length = int(meta["length"])
        proto = meta["protocol"]
        if proto == "iso_a":
            iso_a_samples[length].append(counts.get("0", 0) / total)
        elif proto == "iso_b":
            iso_b_samples[length].append(counts.get("0", 0) / total)
        elif proto == "sim_ab":
            # Qiskit little-endian bitstring: c[1]c[0] -> s[-1] is q_a, s[-2] is q_b
            p_a0 = sum(v for k, v in counts.items() if k[-1] == "0") / total
            p_b0 = sum(v for k, v in counts.items() if k[-2] == "0") / total
            p_00 = counts.get("00", 0) / total
            sim_a_samples[length].append(p_a0)
            sim_b_samples[length].append(p_b0)
            sim_00_samples[length].append(p_00)

    p0_iso_a, p0_iso_a_std = _aggregate_curve(iso_a_samples, lengths, config.shots)
    p0_iso_b, p0_iso_b_std = _aggregate_curve(iso_b_samples, lengths, config.shots)
    p0_sim_a, p0_sim_a_std = _aggregate_curve(sim_a_samples, lengths, config.shots)
    p0_sim_b, p0_sim_b_std = _aggregate_curve(sim_b_samples, lengths, config.shots)
    p00_sim, p00_sim_std = _aggregate_curve(sim_00_samples, lengths, config.shots)

    fit_iso_a = _fit_exponential_rb(lengths, p0_iso_a, p0_iso_a_std)
    fit_iso_b = _fit_exponential_rb(lengths, p0_iso_b, p0_iso_b_std)
    fit_sim_a = _fit_exponential_rb(lengths, p0_sim_a, p0_sim_a_std)
    fit_sim_b = _fit_exponential_rb(lengths, p0_sim_b, p0_sim_b_std)

    delta_epc_a = fit_sim_a.epc - fit_iso_a.epc
    delta_epc_b = fit_sim_b.epc - fit_iso_b.epc

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    safe_backend = backend_label.split("(")[0].replace("/", "_")
    out_file = (
        config.output_dir
        / f"srb_2q_{mode}_{safe_backend}_q{qubit_pair[0]}_{qubit_pair[1]}_{timestamp}.json"
    )

    result = SRBResult(
        mode=mode,
        backend_name=backend_label,
        qubit_pair=list(qubit_pair),
        lengths=lengths,
        num_samples=config.num_samples,
        shots=config.shots,
        p0_iso_a=p0_iso_a,
        p0_iso_a_std=p0_iso_a_std,
        p0_iso_b=p0_iso_b,
        p0_iso_b_std=p0_iso_b_std,
        p0_sim_a=p0_sim_a,
        p0_sim_a_std=p0_sim_a_std,
        p0_sim_b=p0_sim_b,
        p0_sim_b_std=p0_sim_b_std,
        p00_sim=p00_sim,
        p00_sim_std=p00_sim_std,
        iso_a=fit_iso_a,
        iso_b=fit_iso_b,
        sim_a=fit_sim_a,
        sim_b=fit_sim_b,
        delta_epc_a=delta_epc_a,
        delta_epc_b=delta_epc_b,
        avg_physical_gates_per_clifford=avg_phys_gates,
        calibration_a=cal_a,
        calibration_b=cal_b,
        synthetic_zz_mhz=synthetic_zz,
        job_ids=job_ids,
        timestamp_utc=timestamp,
        output_file=str(out_file),
    )

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2)

    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for 2-qubit SRB runner."""
    parser = argparse.ArgumentParser(
        description="Run 2-qubit Isolated + Simultaneous RB on IBM Quantum or local Aer dry-run."
    )
    parser.add_argument(
        "--pair",
        type=int,
        nargs=2,
        default=None,
        help="Optional physical qubit pair (e.g. --pair 0 1). Defaults to auto-selection.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1, 20, 50, 100, 200, 350, 500],
        help="Clifford sequence lengths.",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Random seeds per length.")
    parser.add_argument("--shots", type=int, default=1024, help="Measurement shots.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--timeout", type=float, default=300.0, help="IBM job timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Run locally on Qiskit Aer with ZZ crosstalk.")
    parser.add_argument("--zz-mhz", type=float, default=0.25, help="Synthetic ZZ coupling (MHz) for --dry-run.")
    parser.add_argument("--backend", type=str, default=None, help="Optional IBM Quantum backend name.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hardware"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = SRBConfig(
        qubit_pair=(args.pair[0], args.pair[1]) if args.pair is not None else None,
        lengths=tuple(args.lengths),
        num_samples=args.num_samples,
        shots=args.shots,
        seed=args.seed,
        timeout_s=args.timeout,
        dry_run=args.dry_run,
        dry_run_zz_mhz=args.zz_mhz,
        backend_name=args.backend,
        output_dir=args.output_dir,
    )
    res = run_simultaneous_rb(config)
    qa, qb = res.qubit_pair
    print(f"SRB completed ({res.mode} | backend={res.backend_name} | pair=({qa}, {qb}))")
    print(f"  Q{qa}: EPC_iso={res.iso_a.epc:.3e}, EPC_sim={res.sim_a.epc:.3e}, Delta_EPC={res.delta_epc_a:+.3e}")
    print(f"  Q{qb}: EPC_iso={res.iso_b.epc:.3e}, EPC_sim={res.sim_b.epc:.3e}, Delta_EPC={res.delta_epc_b:+.3e}")
    print(f"  Phys gates / Clifford = {res.avg_physical_gates_per_clifford:.3f}")
    print(f"  Saved = {res.output_file}")


if __name__ == "__main__":
    main()
