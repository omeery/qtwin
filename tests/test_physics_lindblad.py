"""Analytical verification tests for the 1-qubit and 2-qubit Lindblad solver."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from qtwin.physics import (
    SingleQubitParams,
    TwoQubitParams,
    excited_state,
    plus_state,
    simulate_ramsey_decay,
    simulate_t1_relaxation,
    simulate_two_qubit,
    simulate_zz_phase_accumulation,
)

STRICT_TOL = 1e-4
DENSITY_MATRIX_TOL = 1e-9


def _assert_valid_density_matrices(states: list[qt.Qobj]) -> None:
    """Assert Tr(rho) == 1, Hermiticity, and positive semi-definiteness at all steps."""
    assert len(states) > 0
    for idx, rho in enumerate(states):
        mat = rho.full()
        # Trace preservation
        tr = np.trace(mat)
        assert abs(tr - 1.0) < DENSITY_MATRIX_TOL, (
            f"Trace violation at step {idx}: Tr(rho) = {tr}"
        )
        # Hermiticity
        herm_err = float(np.max(np.abs(mat - mat.conj().T)))
        assert herm_err < DENSITY_MATRIX_TOL, (
            f"Hermiticity violation at step {idx}: max |rho - rho^dag| = {herm_err}"
        )
        # Positivity (all eigenvalues >= 0 within numerical tolerance)
        eigvals = np.linalg.eigvalsh(0.5 * (mat + mat.conj().T))
        min_eig = float(np.min(eigvals))
        assert min_eig >= -DENSITY_MATRIX_TOL, (
            f"Positivity violation at step {idx}: min eigenvalue = {min_eig}"
        )


@pytest.mark.parametrize(
    ("t1_us", "t2_us", "freq_mhz"),
    [
        (50.0, None, 0.0),
        (75.0, 40.0, 0.25),  # Pure dephasing & detuning must not affect T1 population decay
        (20.0, 40.0, 0.10),  # T2 = 2 * T1 boundary
    ],
)
def test_t1_relaxation_matches_analytical(
    t1_us: float,
    t2_us: float | None,
    freq_mhz: float,
) -> None:
    """Verify numerical P_1(t) matches analytical exp(-t / T1) within 1e-4."""
    params = SingleQubitParams(freq_mhz=freq_mhz, t1_us=t1_us, t2_us=t2_us)
    times_us = np.linspace(0.0, 4.0 * t1_us, 200)

    p1_num, states = simulate_t1_relaxation(params, times_us)
    p1_exact = np.exp(-times_us / t1_us)

    max_err = float(np.max(np.abs(p1_num - p1_exact)))
    assert max_err < STRICT_TOL, f"T1 max error {max_err:.3e} >= {STRICT_TOL}"
    _assert_valid_density_matrices(states)


@pytest.mark.parametrize(
    ("t1_us", "t2_us", "detuning_mhz"),
    [
        (80.0, 45.0, 0.08),   # General case: both T1 and pure dephasing
        (60.0, 120.0, 0.12),  # T1-limited coherence: T2 = 2 * T1
        (None, 35.0, 0.15),   # Pure dephasing only (no T1)
        (100.0, 50.0, 0.0),   # On-resonance Ramsey decay
    ],
)
def test_t2_ramsey_decay_matches_analytical(
    t1_us: float | None,
    t2_us: float,
    detuning_mhz: float,
) -> None:
    """Verify numerical Ramsey decay matches analytical exp(-t / T2) cos(Delta t) within 1e-4."""
    params = SingleQubitParams(freq_mhz=detuning_mhz, t1_us=t1_us, t2_us=t2_us)
    times_us = np.linspace(0.0, 3.0 * t2_us, 300)

    sx_num, p_plus_num, states = simulate_ramsey_decay(params, times_us)

    omega = 2.0 * np.pi * detuning_mhz
    sx_exact = np.exp(-times_us / t2_us) * np.cos(omega * times_us)
    p_plus_exact = 0.5 * (1.0 + sx_exact)

    sx_err = float(np.max(np.abs(sx_num - sx_exact)))
    p_plus_err = float(np.max(np.abs(p_plus_num - p_plus_exact)))

    assert sx_err < STRICT_TOL, f"Ramsey <sigma_x> max error {sx_err:.3e} >= {STRICT_TOL}"
    assert p_plus_err < STRICT_TOL, f"Ramsey P_+ max error {p_plus_err:.3e} >= {STRICT_TOL}"
    _assert_valid_density_matrices(states)


def test_two_qubit_zz_phase_accumulation_matches_analytical() -> None:
    """Verify differential ZZ phase accumulation matches analytical zeta_ZZ * t within 1e-4."""
    zz_mhz = 0.15
    params = TwoQubitParams(
        qubit0=SingleQubitParams(freq_mhz=0.05, t1_us=100.0, t2_us=60.0),
        qubit1=SingleQubitParams(freq_mhz=0.0, t1_us=None, t2_us=None),
        zz_coupling_mhz=zz_mhz,
    )
    times_us = np.linspace(0.0, 15.0, 300)

    result = simulate_zz_phase_accumulation(params, times_us)
    exact_zz_phase = 2.0 * np.pi * zz_mhz * times_us

    max_phase_err = float(np.max(np.abs(result["zz_phase_diff_rad"] - exact_zz_phase)))
    assert max_phase_err < STRICT_TOL, (
        f"ZZ phase accumulation max error {max_phase_err:.3e} >= {STRICT_TOL}"
    )

    _assert_valid_density_matrices(result["states_q1_0"])
    _assert_valid_density_matrices(result["states_q1_1"])


def test_two_qubit_noisy_evolution_trace_and_positivity() -> None:
    """Verify trace preservation and positivity for fully dissipative two-qubit evolution."""
    params = TwoQubitParams(
        qubit0=SingleQubitParams(freq_mhz=0.10, t1_us=45.0, t2_us=30.0),
        qubit1=SingleQubitParams(freq_mhz=0.18, t1_us=55.0, t2_us=35.0),
        zz_coupling_mhz=0.20,
    )
    times_us = np.linspace(0.0, 60.0, 150)
    psi0 = qt.tensor(plus_state(), excited_state())

    res = simulate_two_qubit(params, psi0, times_us, store_states=True)
    _assert_valid_density_matrices(list(res.states))


def test_invalid_decoherence_parameters_raise() -> None:
    """Ensure unphysical T1 / T2 combinations are rejected."""
    with pytest.raises(ValueError, match="cannot exceed 2 \\* T1"):
        SingleQubitParams(t1_us=50.0, t2_us=105.0)

    with pytest.raises(ValueError, match="t1_us must be positive"):
        SingleQubitParams(t1_us=-10.0)

    with pytest.raises(ValueError, match="t2_us must be positive"):
        SingleQubitParams(t2_us=0.0)
