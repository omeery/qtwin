"""Standalone 1-qubit Randomized Benchmarking (RB) characterization script.

Supports three execution modes:
1. Live IBM Quantum cloud execution (free tier):
   Reads ``IBM_QUANTUM_TOKEN`` (and optional ``IBM_QUANTUM_INSTANCE``,
   ``IBM_QUANTUM_CHANNEL``, ``IBM_QUANTUM_BACKEND``) from environment variables
   or a local ``.env`` file. Saves raw RB survival probabilities, EPC/alpha
   analysis results, transpiled physical gate density, and the backend's
   reported qubit calibration metadata (T1, T2, gate duration, gate/readout errors).
2. Local dry-run mode (``--dry-run``):
   Runs the exact same ``qiskit-experiments`` ``StandardRB`` workflow on a local
   ``qiskit-aer`` ``AerSimulator`` configured with a realistic superconducting
   thermal-relaxation (T1/T2) noise model—no API token required.
3. Pending-job retrieval mode (``--retrieve-job <job_id>``):
   If a live IBM Quantum job timed out waiting in queue, retrieves its results
   later without losing progress.

Usage:
    # Local dry-run with noisy Aer simulator:
    python -m qtwin.hardware.rb_experiment --dry-run

    # Live IBM Quantum execution (requires IBM_QUANTUM_TOKEN in .env or env):
    python -m qtwin.hardware.rb_experiment --qubit 0 --shots 1024
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_experiments.library import StandardRB

# Non-virtual single-qubit physical gates (rz is virtual / zero-duration on IBM hardware)
PHYSICAL_1Q_GATES = {"sx", "x", "h", "u", "u1", "u2", "u3", "id"}


@dataclass(frozen=True)
class QubitCalibrationMetadata:
    """Backend-reported calibration properties for the characterized qubit."""

    t1_us: float | None = None
    t2_us: float | None = None
    sx_duration_ns: float | None = None
    sx_error: float | None = None
    x_error: float | None = None
    readout_error: float | None = None
    frequency_ghz: float | None = None


@dataclass(frozen=True)
class RBConfig:
    """Configuration for a 1-qubit Randomized Benchmarking experiment."""

    qubit: int = 0
    lengths: tuple[int, ...] = (1, 50, 150, 300, 500, 800)
    num_samples: int = 5
    shots: int = 1024
    seed: int = 42
    dry_run: bool = False
    backend_name: str | None = None
    timeout_s: float = 300.0
    # Dry-run Aer noise parameters (realistic transmon defaults)
    dry_run_t1_us: float = 80.0
    dry_run_t2_us: float = 60.0
    dry_run_gate_time_ns: float = 35.0
    output_dir: Path = Path("outputs/hardware")


@dataclass(frozen=True)
class RBResult:
    """Full RB experiment output including raw survival curve and backend calibration."""

    mode: str
    backend_name: str
    qubit: int
    lengths: list[int]
    num_samples: int
    shots: int
    epc: float
    epc_std: float | None
    alpha: float
    alpha_std: float | None
    p0_mean: list[float]
    p0_std: list[float]
    samples_by_length: dict[str, list[float]]
    avg_physical_gates_per_clifford: float
    calibration: QubitCalibrationMetadata
    job_ids: list[str]
    timestamp_utc: str
    output_file: str


def build_local_noisy_simulator(
    t1_us: float = 80.0,
    t2_us: float = 60.0,
    gate_time_ns: float = 35.0,
) -> AerSimulator:
    """Create a local Qiskit AerSimulator with single-qubit T1/T2 thermal relaxation noise."""
    t1_s = t1_us * 1e-6
    t2_s = t2_us * 1e-6
    gate_time_s = gate_time_ns * 1e-9

    error_1q = thermal_relaxation_error(t1_s, t2_s, gate_time_s)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1q, ["sx", "x", "h", "u1", "u2", "u3", "id"])

    return AerSimulator(
        noise_model=noise_model,
        basis_gates=["id", "rz", "sx", "x"],
    )


def connect_ibm_service() -> Any:
    """Authenticate with IBM Quantum using environment variables or .env."""
    load_dotenv()

    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token or token == "your_ibm_quantum_token_here":
        raise RuntimeError(
            "IBM_QUANTUM_TOKEN is not set. Set it in your environment or .env file, "
            "or run with --dry-run to use the local Qiskit Aer noisy simulator."
        )

    channel = os.getenv("IBM_QUANTUM_CHANNEL") or None
    instance = os.getenv("IBM_QUANTUM_INSTANCE") or "auto"

    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs: dict[str, Any] = {"token": token, "instance": instance}
    if channel:
        kwargs["channel"] = channel
    return QiskitRuntimeService(**kwargs)


def connect_ibm_backend(backend_name: str | None = None) -> Any:
    """Return the requested IBM Quantum backend or the least-busy real QPU."""
    service = connect_ibm_service()
    target_backend = backend_name or os.getenv("IBM_QUANTUM_BACKEND") or None
    if target_backend:
        return service.backend(target_backend)
    return service.least_busy(operational=True, simulator=False)


def extract_qubit_calibration(
    backend: Any,
    qubit: int,
    *,
    dry_run: bool = False,
    config: RBConfig | None = None,
) -> QubitCalibrationMetadata:
    """Fetch reported qubit calibration metadata from an IBM backend or dry-run config."""
    if dry_run and config is not None:
        return QubitCalibrationMetadata(
            t1_us=config.dry_run_t1_us,
            t2_us=config.dry_run_t2_us,
            sx_duration_ns=config.dry_run_gate_time_ns,
            sx_error=None,
            x_error=None,
            readout_error=0.0,
            frequency_ghz=5.0,
        )

    props = backend.properties()
    if props is None:
        return QubitCalibrationMetadata()

    def _safe(fn: Any, *args: Any, scale: float = 1.0) -> float | None:
        try:
            val = fn(*args)
            return float(val) * scale if val is not None else None
        except Exception:
            return None

    return QubitCalibrationMetadata(
        t1_us=_safe(props.t1, qubit, scale=1e6),
        t2_us=_safe(props.t2, qubit, scale=1e6),
        sx_duration_ns=_safe(props.gate_length, "sx", qubit, scale=1e9),
        sx_error=_safe(props.gate_error, "sx", qubit),
        x_error=_safe(props.gate_error, "x", qubit),
        readout_error=_safe(props.readout_error, qubit),
        frequency_ghz=_safe(props.frequency, qubit, scale=1e-9),
    )


def estimate_physical_gates_per_clifford(rb_exp: StandardRB, backend: Any) -> float:
    """Estimate average number of non-virtual 1q physical gates per Clifford on ``backend``."""
    circuits = rb_exp.circuits()
    transpiled = transpile(circuits, backend=backend, optimization_level=1)

    total_phys_gates = 0
    total_cliffords = 0
    for circ in transpiled:
        xval = circ.metadata.get("xval")
        if xval is None:
            continue
        ops = circ.count_ops()
        phys = sum(int(ops.get(g, 0)) for g in PHYSICAL_1Q_GATES)
        total_phys_gates += phys
        # Each RB circuit of length m contains m random Cliffords + 1 inverting Clifford
        total_cliffords += int(xval) + 1

    if total_cliffords == 0:
        return 1.0
    return float(total_phys_gates / total_cliffords)


def _extract_metric(exp_data: Any, name: str) -> tuple[float, float | None]:
    """Safely extract nominal value and standard deviation from ExperimentData."""
    df = exp_data.analysis_results(name, dataframe=True)
    val = df.iloc[0]["value"]
    nominal = getattr(val, "nominal_value", None)
    std_dev = getattr(val, "std_dev", None)
    if nominal is not None:
        std_clean = None if (std_dev is None or np.isnan(std_dev)) else float(std_dev)
        return float(nominal), std_clean
    return float(val), None


def _extract_survival_curve(
    exp_data: Any,
    lengths: list[int],
    shots: int,
) -> tuple[list[float], list[float], dict[str, list[float]]]:
    """Extract mean and standard error of ground-state survival probability P_0(m)."""
    samples: dict[int, list[float]] = defaultdict(list)
    for datum in exp_data.data():
        meta = datum.get("metadata", {})
        xval = meta.get("xval")
        counts = datum.get("counts", {})
        if xval is None or not counts:
            continue
        total = sum(counts.values())
        if total <= 0:
            continue
        # Ground state bitstring for 1-qubit RB is "0" (or "0x0" in hex)
        c0 = counts.get("0", counts.get("0x0", 0))
        samples[int(xval)].append(float(c0) / float(total))

    p0_mean: list[float] = []
    p0_std: list[float] = []
    samples_str: dict[str, list[float]] = {}

    for length in lengths:
        vals = samples.get(int(length), [])
        samples_str[str(length)] = vals
        if not vals:
            p0_mean.append(float("nan"))
            p0_std.append(float("nan"))
            continue
        arr = np.asarray(vals, dtype=float)
        mean_val = float(np.mean(arr))
        # Standard error across random Clifford seeds, floored by binomial shot noise
        n_samp = len(arr)
        sem_empirical = float(np.std(arr, ddof=1) / np.sqrt(n_samp)) if n_samp > 1 else 0.0
        sem_shot = float(np.sqrt(max(mean_val * (1.0 - mean_val), 1e-6) / (shots * n_samp)))
        p0_mean.append(mean_val)
        p0_std.append(max(sem_empirical, sem_shot))

    return p0_mean, p0_std, samples_str


def run_single_qubit_rb(config: RBConfig) -> RBResult:
    """Execute 1-qubit StandardRB on either a local noisy simulator or IBM Quantum QPU."""
    if config.dry_run:
        backend = build_local_noisy_simulator(
            t1_us=config.dry_run_t1_us,
            t2_us=config.dry_run_t2_us,
            gate_time_ns=config.dry_run_gate_time_ns,
        )
        mode = "dry_run_aer"
        backend_label = (
            f"aer_thermal(T1={config.dry_run_t1_us}us, T2={config.dry_run_t2_us}us)"
        )
    else:
        backend = connect_ibm_backend(config.backend_name)
        mode = "ibm_quantum"
        backend_label = getattr(backend, "name", str(backend))

    calibration = extract_qubit_calibration(
        backend, config.qubit, dry_run=config.dry_run, config=config
    )

    rb_exp = StandardRB(
        physical_qubits=(config.qubit,),
        lengths=list(config.lengths),
        num_samples=config.num_samples,
        seed=config.seed,
    )
    avg_phys_gates = estimate_physical_gates_per_clifford(rb_exp, backend)

    exp_data = rb_exp.run(backend, shots=config.shots)
    job_ids = [str(jid) for jid in getattr(exp_data, "job_ids", [])]

    try:
        exp_data.block_for_results(timeout=config.timeout_s)
    except Exception as exc:
        # Save pending job metadata so results can be retrieved later without losing progress
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        pending_file = config.output_dir / f"pending_rb_{backend_label}_q{config.qubit}_{timestamp}.json"
        with pending_file.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "backend_name": backend_label,
                    "qubit": config.qubit,
                    "lengths": list(config.lengths),
                    "num_samples": config.num_samples,
                    "shots": config.shots,
                    "seed": config.seed,
                    "avg_physical_gates_per_clifford": avg_phys_gates,
                    "calibration": asdict(calibration),
                    "job_ids": job_ids,
                    "timestamp_utc": timestamp,
                    "error": str(exc),
                },
                fh,
                indent=2,
            )
        raise RuntimeError(
            f"RB job did not complete within {config.timeout_s}s. "
            f"Saved pending job_ids={job_ids} to {pending_file}."
        ) from exc

    epc, epc_std = _extract_metric(exp_data, "EPC")
    alpha, alpha_std = _extract_metric(exp_data, "alpha")
    p0_mean, p0_std, samples_by_length = _extract_survival_curve(
        exp_data, list(config.lengths), config.shots
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    safe_backend = backend_label.split("(")[0].replace("/", "_")
    out_file = config.output_dir / f"rb_1q_{mode}_{safe_backend}_q{config.qubit}_{timestamp}.json"

    result = RBResult(
        mode=mode,
        backend_name=backend_label,
        qubit=config.qubit,
        lengths=list(config.lengths),
        num_samples=config.num_samples,
        shots=config.shots,
        epc=epc,
        epc_std=epc_std,
        alpha=alpha,
        alpha_std=alpha_std,
        p0_mean=p0_mean,
        p0_std=p0_std,
        samples_by_length=samples_by_length,
        avg_physical_gates_per_clifford=avg_phys_gates,
        calibration=calibration,
        job_ids=job_ids,
        timestamp_utc=timestamp,
        output_file=str(out_file),
    )

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2)

    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the 1-qubit RB runner."""
    parser = argparse.ArgumentParser(
        description="Run 1-qubit Randomized Benchmarking on IBM Quantum or local Aer dry-run."
    )
    parser.add_argument("--qubit", type=int, default=0, help="Physical qubit index (default: 0).")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1, 50, 150, 300, 500, 800],
        help="Clifford sequence lengths.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Random Clifford seeds per length (default: 5)."
    )
    parser.add_argument("--shots", type=int, default=1024, help="Measurement shots (default: 1024).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Max seconds to wait for IBM Quantum job completion (default: 300).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run locally on a noisy Qiskit Aer simulator without an IBM Quantum token.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional IBM Quantum backend name (ignored in --dry-run mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hardware"),
        help="Directory for saving JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = RBConfig(
        qubit=args.qubit,
        lengths=tuple(args.lengths),
        num_samples=args.num_samples,
        shots=args.shots,
        seed=args.seed,
        timeout_s=args.timeout,
        dry_run=args.dry_run,
        backend_name=args.backend,
        output_dir=args.output_dir,
    )
    result = run_single_qubit_rb(config)
    print(f"RB completed ({result.mode} | backend={result.backend_name} | qubit={result.qubit})")
    print(f"  alpha = {result.alpha:.6f} +/- {result.alpha_std}")
    print(f"  EPC   = {result.epc:.6e} +/- {result.epc_std}")
    print(f"  Phys gates / Clifford = {result.avg_physical_gates_per_clifford:.3f}")
    if result.calibration.t1_us is not None:
        print(
            f"  Calibration: T1={result.calibration.t1_us:.2f} us, "
            f"T2={result.calibration.t2_us:.2f} us, "
            f"sx_len={result.calibration.sx_duration_ns} ns, "
            f"sx_err={result.calibration.sx_error}, "
            f"readout_err={result.calibration.readout_error}"
        )
    print(f"  Saved = {result.output_file}")


if __name__ == "__main__":
    main()
