"""Unit tests for the Phase 3 Lindblad-to-Pauli bridge and Stim + PyMatching QEC pipeline."""

from __future__ import annotations

import pytest

from qtwin.qec import (
    HardwareCalibrationInput,
    build_surface_code_circuit,
    evaluate_surface_code_memory,
    lindblad_to_circuit_noise,
    pauli_twirl_single_qubit,
    pauli_twirl_zz_crosstalk,
)


def test_pauli_twirl_normalization_and_monotonicity() -> None:
    """Verify valid probabilities and monotonic scaling with T1, T2, and f_ZZ."""
    # Higher decoherence (shorter T1/T2) -> strictly higher single-qubit Pauli error
    px_good, py_good, pz_good = pauli_twirl_single_qubit(t1_us=100.0, t2_us=80.0, duration_us=0.2)
    px_bad, py_bad, pz_bad = pauli_twirl_single_qubit(t1_us=30.0, t2_us=20.0, duration_us=0.2)

    assert px_good == py_good
    assert 0.0 < px_good + py_good + pz_good < 1.0
    assert (px_bad + py_bad + pz_bad) > (px_good + py_good + pz_good)
    assert pz_bad > pz_good

    # Stronger ZZ crosstalk -> strictly higher p_ZZ (for theta < pi)
    piz_0, pzi_0, pzz_0 = pauli_twirl_zz_crosstalk(zz_coupling_mhz=0.0, duration_us=0.2)
    piz_1, pzi_1, pzz_1 = pauli_twirl_zz_crosstalk(zz_coupling_mhz=0.2, duration_us=0.2)
    piz_2, pzi_2, pzz_2 = pauli_twirl_zz_crosstalk(zz_coupling_mhz=0.6, duration_us=0.2)

    assert piz_0 == pzi_0 == pzz_0 == 0.0
    assert 0.0 < pzz_1 < pzz_2 <= 0.25
    assert piz_2 + pzi_2 + pzz_2 <= 0.75


def test_pauli_twirl_rejects_unphysical_params() -> None:
    """Reject T2 > 2*T1 or negative ZZ coupling."""
    with pytest.raises(ValueError, match="Physical bound violated"):
        pauli_twirl_single_qubit(t1_us=50.0, t2_us=110.0, duration_us=0.2)
    with pytest.raises(ValueError, match="non-negative"):
        pauli_twirl_zz_crosstalk(zz_coupling_mhz=-0.1, duration_us=0.2)


def test_stim_dem_and_pymatching_below_threshold_suppression() -> None:
    """In low-noise regime, DEM decomposes cleanly and LER strictly decreases with d (d=5 < d=3)."""
    cal = HardwareCalibrationInput(
        t1_us=150.0,
        t2_us=120.0,
        gate_duration_ns=24.0,
        cycle_duration_us=0.20,
        residual_error=5.0e-4,
        zz_coupling_mhz=0.0,
        readout_error=0.003,
    )
    noise = lindblad_to_circuit_noise(cal)

    c3 = build_surface_code_circuit(3, noise, rounds=3)
    c5 = build_surface_code_circuit(5, noise, rounds=5)

    # Ensure DEM decomposes without exceptions
    dem3 = c3.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    dem5 = c5.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    assert dem3.num_detectors > 0 and dem5.num_detectors > dem3.num_detectors

    r3 = evaluate_surface_code_memory(c3, shots=20_000, seed=11, distance=3, rounds=3)
    r5 = evaluate_surface_code_memory(c5, shots=20_000, seed=11, distance=5, rounds=5)

    assert r3.ler > 0.0
    assert r5.ler < r3.ler, f"Expected LER(d=5)={r5.ler} < LER(d=3)={r3.ler} below threshold"


def test_noise_aware_decoder_beats_naive_under_strong_zz_crosstalk() -> None:
    """Under strong correlated ZZ crosstalk, Noise-Aware decoder achieves <= LER vs Naive decoder."""
    cal_uncorr = HardwareCalibrationInput(
        t1_us=120.0,
        t2_us=100.0,
        gate_duration_ns=24.0,
        cycle_duration_us=0.25,
        residual_error=4.0e-4,
        zz_coupling_mhz=0.0,
        readout_error=0.002,
    )
    cal_corr = HardwareCalibrationInput(
        t1_us=120.0,
        t2_us=100.0,
        gate_duration_ns=24.0,
        cycle_duration_us=0.25,
        residual_error=4.0e-4,
        zz_coupling_mhz=0.65,  # strong correlated ZZ crosstalk
        readout_error=0.002,
    )

    circ_uncorr = build_surface_code_circuit(5, lindblad_to_circuit_noise(cal_uncorr), rounds=5)
    circ_corr = build_surface_code_circuit(5, lindblad_to_circuit_noise(cal_corr), rounds=5)

    naive = evaluate_surface_code_memory(
        circ_corr, decoder_circuit=circ_uncorr, shots=20_000, seed=99, distance=5, rounds=5
    )
    aware = evaluate_surface_code_memory(
        circ_corr, decoder_circuit=circ_corr, shots=20_000, seed=99, distance=5, rounds=5
    )

    assert aware.ler <= naive.ler, (
        f"Noise-aware decoder LER ({aware.ler:.4e}) should not exceed naive ({naive.ler:.4e})"
    )
