"""Single-qubit and two-qubit Lindblad master equation simulator using QuTiP.

Conventions (hbar = 1):
- |0> = basis(2, 0) is the ground state; |1> = basis(2, 1) is the excited state.
- Number operator: n = |1><1| = destroy(2).dag() * destroy(2).
- Amplitude damping collapse operator: L_1 = sqrt(gamma_1) * destroy(2),
  which maps |1> -> |0> at rate gamma_1 = 1 / T1.
- Pure dephasing collapse operator: L_phi = sqrt(gamma_phi / 2) * sigmaz(),
  which damps off-diagonal coherence rho_01 at rate gamma_phi = 1/T2 - 1/(2*T1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import qutip as qt

from qtwin.physics.params import SingleQubitParams, TwoQubitParams


def ground_state() -> qt.Qobj:
    """Return single-qubit ground state |0>."""
    return qt.basis(2, 0)


def excited_state() -> qt.Qobj:
    """Return single-qubit excited state |1>."""
    return qt.basis(2, 1)


def plus_state() -> qt.Qobj:
    """Return single-qubit superposition state |+> = (|0> + |1>) / sqrt(2)."""
    return (qt.basis(2, 0) + qt.basis(2, 1)).unit()


def build_single_qubit_hamiltonian(params: SingleQubitParams) -> qt.Qobj:
    """Construct the single-qubit Hamiltonian H = omega_q |1><1| (rad/us)."""
    return params.omega_rad_per_us * qt.num(2)


def build_single_qubit_collapse_ops(params: SingleQubitParams) -> list[qt.Qobj]:
    """Construct Lindblad collapse operators for T1 relaxation and pure dephasing."""
    c_ops: list[qt.Qobj] = []
    if params.gamma1_per_us > 0.0:
        c_ops.append(np.sqrt(params.gamma1_per_us) * qt.destroy(2))
    if params.gamma_phi_per_us > 0.0:
        c_ops.append(np.sqrt(0.5 * params.gamma_phi_per_us) * qt.sigmaz())
    return c_ops


def build_two_qubit_hamiltonian(params: TwoQubitParams) -> qt.Qobj:
    """Construct two-qubit Hamiltonian with static ZZ crosstalk.

    H = omega_0 (n x I) + omega_1 (I x n) + zeta_zz (n x n).
    """
    n_op = qt.num(2)
    eye = qt.qeye(2)

    h0 = params.qubit0.omega_rad_per_us * qt.tensor(n_op, eye)
    h1 = params.qubit1.omega_rad_per_us * qt.tensor(eye, n_op)
    h_zz = params.zeta_zz_rad_per_us * qt.tensor(n_op, n_op)
    return h0 + h1 + h_zz


def build_two_qubit_collapse_ops(params: TwoQubitParams) -> list[qt.Qobj]:
    """Construct local Lindblad collapse operators for a two-qubit system."""
    eye = qt.qeye(2)
    c_ops: list[qt.Qobj] = []

    for op in build_single_qubit_collapse_ops(params.qubit0):
        c_ops.append(qt.tensor(op, eye))
    for op in build_single_qubit_collapse_ops(params.qubit1):
        c_ops.append(qt.tensor(eye, op))
    return c_ops


def _as_density_matrix(state: qt.Qobj) -> qt.Qobj:
    """Convert a ket to a density matrix, or return an existing density matrix."""
    return qt.ket2dm(state) if state.isket else state


def simulate_single_qubit(
    params: SingleQubitParams,
    initial_state: qt.Qobj,
    times_us: np.ndarray,
    *,
    e_ops: list[qt.Qobj] | dict[str, qt.Qobj] | None = None,
    store_states: bool = True,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> qt.solver.Result:
    """Evolve a single-qubit open quantum system via QuTiP's ``mesolve``.

    Args:
        params: Single-qubit physical parameters.
        initial_state: Initial ket or density matrix (2D Hilbert space).
        times_us: 1D array of evaluation times in microseconds.
        e_ops: Optional expectation-value operators.
        store_states: Whether to store full density matrices at each time step.
        atol: Absolute ODE solver tolerance.
        rtol: Relative ODE solver tolerance.

    Returns:
        QuTiP ``Result`` object containing ``states`` and/or ``expect``.
    """
    hamiltonian = build_single_qubit_hamiltonian(params)
    c_ops = build_single_qubit_collapse_ops(params)
    rho0 = _as_density_matrix(initial_state)

    return qt.mesolve(
        hamiltonian,
        rho0,
        times_us,
        c_ops=c_ops,
        e_ops=e_ops,
        options={"store_states": store_states, "atol": atol, "rtol": rtol},
    )


def simulate_two_qubit(
    params: TwoQubitParams,
    initial_state: qt.Qobj,
    times_us: np.ndarray,
    *,
    e_ops: list[qt.Qobj] | dict[str, qt.Qobj] | None = None,
    store_states: bool = True,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> qt.solver.Result:
    """Evolve a two-qubit open quantum system with static ZZ crosstalk via ``mesolve``.

    Args:
        params: Two-qubit physical parameters.
        initial_state: Initial ket or density matrix in the 4D two-qubit Hilbert space.
        times_us: 1D array of evaluation times in microseconds.
        e_ops: Optional expectation-value operators.
        store_states: Whether to store full density matrices at each time step.
        atol: Absolute ODE solver tolerance.
        rtol: Relative ODE solver tolerance.

    Returns:
        QuTiP ``Result`` object containing ``states`` and/or ``expect``.
    """
    hamiltonian = build_two_qubit_hamiltonian(params)
    c_ops = build_two_qubit_collapse_ops(params)
    rho0 = _as_density_matrix(initial_state)

    return qt.mesolve(
        hamiltonian,
        rho0,
        times_us,
        c_ops=c_ops,
        e_ops=e_ops,
        options={"store_states": store_states, "atol": atol, "rtol": rtol},
    )


def simulate_t1_relaxation(
    params: SingleQubitParams,
    times_us: np.ndarray,
) -> tuple[np.ndarray, list[qt.Qobj]]:
    """Run a T1 energy relaxation experiment starting from |1>.

    Returns:
        Tuple of ``(p1_array, states)``, where ``p1_array`` is the excited-state
        population P_1(t) = <1|rho(t)|1>.
    """
    result = simulate_single_qubit(
        params=params,
        initial_state=excited_state(),
        times_us=times_us,
        e_ops=[qt.num(2)],
        store_states=True,
    )
    p1 = np.asarray(result.expect[0], dtype=float)
    return p1, list(result.states)


def simulate_ramsey_decay(
    params: SingleQubitParams,
    times_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[qt.Qobj]]:
    """Run a T2 Ramsey free-induction decay experiment starting from |+>.

    Returns:
        Tuple of ``(sigma_x, p_plus, states)``, where:
        - ``sigma_x`` is <sigma_x(t)> = exp(-t / T2) * cos(omega * t)
        - ``p_plus`` is P_+(t) = <+|rho(t)|+> = 0.5 * (1 + <sigma_x(t)>)
    """
    result = simulate_single_qubit(
        params=params,
        initial_state=plus_state(),
        times_us=times_us,
        e_ops=[qt.sigmax()],
        store_states=True,
    )
    sigma_x = np.asarray(result.expect[0], dtype=float)
    p_plus = 0.5 * (1.0 + sigma_x)
    return sigma_x, p_plus, list(result.states)


def simulate_zz_phase_accumulation(
    params: TwoQubitParams,
    times_us: np.ndarray,
) -> dict[str, Any]:
    """Run a conditional Ramsey experiment on qubit 0 with spectator qubit 1 in |0> vs |1>.

    When qubit 0 is initialized in |+>, its phase evolves at rate omega_0 if
    qubit 1 is in |0>, and at rate (omega_0 + zeta_zz) if qubit 1 is in |1>.
    Because <sigma_x> = r * cos(phi) and <sigma_y> = -r * sin(phi) under
    H = omega |1><1|, the unwrapped phase is extracted via
    ``angle(<sigma_x> - 1j * <sigma_y>)``.

    Returns:
        Dictionary containing expectation values, unwrapped phases, differential
        ZZ phase ``zz_phase_diff_rad``, and density matrix trajectories.
    """
    eye = qt.qeye(2)
    sx0 = qt.tensor(qt.sigmax(), eye)
    sy0 = qt.tensor(qt.sigmay(), eye)

    psi_q1_0 = qt.tensor(plus_state(), ground_state())
    psi_q1_1 = qt.tensor(plus_state(), excited_state())

    res_0 = simulate_two_qubit(
        params=params,
        initial_state=psi_q1_0,
        times_us=times_us,
        e_ops=[sx0, sy0],
        store_states=True,
    )
    res_1 = simulate_two_qubit(
        params=params,
        initial_state=psi_q1_1,
        times_us=times_us,
        e_ops=[sx0, sy0],
        store_states=True,
    )

    sx_0 = np.asarray(res_0.expect[0], dtype=float)
    sy_0 = np.asarray(res_0.expect[1], dtype=float)
    sx_1 = np.asarray(res_1.expect[0], dtype=float)
    sy_1 = np.asarray(res_1.expect[1], dtype=float)

    phase_0 = np.unwrap(np.angle(sx_0 - 1j * sy_0))
    phase_1 = np.unwrap(np.angle(sx_1 - 1j * sy_1))
    zz_phase_diff = phase_1 - phase_0

    return {
        "times_us": times_us,
        "sigma_x_q1_0": sx_0,
        "sigma_y_q1_0": sy_0,
        "sigma_x_q1_1": sx_1,
        "sigma_y_q1_1": sy_1,
        "phase_q1_0_rad": phase_0,
        "phase_q1_1_rad": phase_1,
        "zz_phase_diff_rad": zz_phase_diff,
        "states_q1_0": list(res_0.states),
        "states_q1_1": list(res_1.states),
    }
