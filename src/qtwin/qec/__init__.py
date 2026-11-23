"""Phase 3 QEC bridge: Lindblad-to-Pauli twirling and Stim + PyMatching surface code evaluation."""

from qtwin.qec.noise_bridge import (
    CircuitNoiseParams,
    HardwareCalibrationInput,
    lindblad_to_circuit_noise,
    load_ibm_fez_calibration,
    pauli_twirl_single_qubit,
    pauli_twirl_zz_crosstalk,
)
from qtwin.qec.qec_simulator import (
    QECBenchmarkPoint,
    build_surface_code_circuit,
    compute_suppression_factor,
    evaluate_surface_code_memory,
    run_four_regime_benchmark,
    run_zz_crosstalk_sweep,
)

__all__ = [
    "CircuitNoiseParams",
    "HardwareCalibrationInput",
    "pauli_twirl_single_qubit",
    "pauli_twirl_zz_crosstalk",
    "lindblad_to_circuit_noise",
    "load_ibm_fez_calibration",
    "QECBenchmarkPoint",
    "build_surface_code_circuit",
    "evaluate_surface_code_memory",
    "compute_suppression_factor",
    "run_four_regime_benchmark",
    "run_zz_crosstalk_sweep",
]
