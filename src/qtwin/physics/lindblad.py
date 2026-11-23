"""Single-qubit and two-qubit Lindblad master equation solver (QuTiP mesolve, hbar = 1).

Basis: |0> = basis(2, 0) ground, |1> = basis(2, 1) excited, n = |1><1|.
Collapse ops: L_1 = sqrt(1/T1) * destroy(2), L_phi = sqrt(gamma_phi / 2) * sigmaz().
"""

from __future__ import annotations

from typing import Any

import numpy as np
import qutip as qt

from qtwin.physics.params import SingleQubitParams, TwoQubitParams


def ground_state() -> qt.Qobj:
    return qt.basis(2, 0)


def excited_state() -> qt.Qobj:
    return qt.basis(2, 1)


def plus_state() -> qt.Qobj:
    return (qt.basis(2, 0) + qt.basis(2, 1)).unit()


def build_single_qubit_hamiltonian(params: SingleQubitParams) -> qt.Qobj:
    return params.omega_rad_per_us * qt.num(2)


def build_single_qubit_collapse_ops(params: SingleQubitParams) -> list[qt.Qobj]:
    c_ops: list[qt.Qobj] = []
    if params.gamma1_per_us > 0.0:
        c_ops.append(np.sqrt(params.gamma1_per_us) * qt.destroy(2))
    if params.gamma_phi_per_us > 0.0:
        c_ops.append(np.sqrt(0.5 * params.gamma_phi_per_us) * qt.sigmaz())
    return c_ops


def build_two_qubit_hamiltonian(params: TwoQubitParams) -> qt.Qobj:
    n_op = qt.num(2)
    eye = qt.qeye(2)
    h0 = params.qubit0.omega_rad_per_us * qt.tensor(n_op, eye)
    h1 = params.qubit1.omega_rad_per_us * qt.tensor(eye, n_op)
    h_zz = params.zeta_zz_rad_per_us * qt.tensor(n_op, n_op)
    return h0 + h1 + h_zz


def build_two_qubit_collapse_ops(params: TwoQubitParams) -> list[qt.Qobj]:
    eye = qt.qeye(2)
    c_ops: list[qt.Qobj] = []
    for op in build_single_qubit_collapse_ops(params.qubit0):
        c_ops.append(qt.tensor(op, eye))
    for op in build_single_qubit_collapse_ops(params.qubit1):
        c_ops.append(qt.tensor(eye, op))
    return c_ops


def _as_density_matrix(state: qt.Qobj) -> qt.Qobj:
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
    """Evolve a single-qubit open system via QuTiP mesolve."""
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
    """Evolve a two-qubit open system with static ZZ coupling via QuTiP mesolve."""
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
    """T1 relaxation from |1>; returns (P_1(t), states)."""
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
    """Ramsey decay from |+>; returns (<sigma_x(t)>, P_+(t), states)."""
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
    """Conditional Ramsey on qubit 0 with spectator qubit 1 in |0> vs |1>."""
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

    # Under H = w |1><1|, <sigma_x> - i<sigma_y> = r * exp(+i * phi)
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
