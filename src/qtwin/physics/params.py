"""Physical parameter containers and decoherence rate conversions.

Units convention throughout the physics engine (with hbar = 1):
- Time (T1, T2, evolution times): microseconds (us)
- Frequencies (qubit frequency / detuning, ZZ coupling): megahertz (MHz)
- Angular frequencies (omega, zeta_zz): rad / us  (since 1 MHz * 1 us = 1)
- Decoherence rates (gamma1, gamma_phi): 1 / us
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingleQubitParams:
    """Parameters for a single two-level superconducting qubit.

    Attributes:
        freq_mhz: Qubit transition frequency (or rotating-frame detuning) in MHz.
        t1_us: Energy relaxation time T1 in microseconds. ``None`` or ``inf``
            disables amplitude damping.
        t2_us: Coherence (Ramsey dephasing) time T2 in microseconds. ``None``
            or ``inf`` disables pure dephasing (leaving only the T1-limited
            dephasing rate 1 / (2 * T1)).
    """

    freq_mhz: float = 0.0
    t1_us: float | None = None
    t2_us: float | None = None

    def __post_init__(self) -> None:
        if self.t1_us is not None and self.t1_us <= 0.0:
            raise ValueError(f"t1_us must be positive, got {self.t1_us}.")
        if self.t2_us is not None and self.t2_us <= 0.0:
            raise ValueError(f"t2_us must be positive, got {self.t2_us}.")

        if (
            self.t1_us is not None
            and self.t2_us is not None
            and math.isfinite(self.t1_us)
            and math.isfinite(self.t2_us)
        ):
            # Physical bound: T2 <= 2 * T1 (small tolerance for float rounding)
            if self.t2_us > 2.0 * self.t1_us * (1.0 + 1e-12):
                raise ValueError(
                    f"Physical decoherence bound violated: T2 ({self.t2_us} us) "
                    f"cannot exceed 2 * T1 ({2.0 * self.t1_us} us)."
                )

    @property
    def omega_rad_per_us(self) -> float:
        """Angular frequency omega = 2 * pi * freq_mhz in rad/us."""
        return float(2.0 * np.pi * self.freq_mhz)

    @property
    def gamma1_per_us(self) -> float:
        """Amplitude damping rate gamma_1 = 1 / T1 in 1/us."""
        if self.t1_us is None or math.isinf(self.t1_us):
            return 0.0
        return 1.0 / self.t1_us

    @property
    def gamma_phi_per_us(self) -> float:
        """Pure dephasing rate gamma_phi = 1 / T2 - 1 / (2 * T1) in 1/us."""
        if self.t2_us is None or math.isinf(self.t2_us):
            return 0.0
        rate = (1.0 / self.t2_us) - 0.5 * self.gamma1_per_us
        return max(0.0, rate)


@dataclass(frozen=True)
class TwoQubitParams:
    """Parameters for a pair of superconducting qubits with static ZZ crosstalk.

    The two-qubit Hamiltonian is:
        H = omega_0 (n_0 x I) + omega_1 (I x n_1) + zeta_zz (n_0 x n_1)
    where n = |1><1| is the excited-state projector and zeta_zz = 2 * pi * zz_coupling_mhz.

    Attributes:
        qubit0: Parameters for qubit 0.
        qubit1: Parameters for qubit 1.
        zz_coupling_mhz: Static ZZ frequency shift (f_ZZ) in MHz.
    """

    qubit0: SingleQubitParams
    qubit1: SingleQubitParams
    zz_coupling_mhz: float = 0.0

    @property
    def zeta_zz_rad_per_us(self) -> float:
        """Angular ZZ coupling strength zeta_zz = 2 * pi * zz_coupling_mhz in rad/us."""
        return float(2.0 * np.pi * self.zz_coupling_mhz)
