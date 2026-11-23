"""Physical parameter containers and decoherence rate conversions (hbar = 1, us / MHz)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingleQubitParams:
    """Two-level qubit parameters (freq in MHz, T1/T2 in us)."""

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
            if self.t2_us > 2.0 * self.t1_us * (1.0 + 1e-12):
                raise ValueError(
                    f"Physical decoherence bound violated: T2 ({self.t2_us} us) "
                    f"cannot exceed 2 * T1 ({2.0 * self.t1_us} us)."
                )

    @property
    def omega_rad_per_us(self) -> float:
        return float(2.0 * np.pi * self.freq_mhz)

    @property
    def gamma1_per_us(self) -> float:
        if self.t1_us is None or math.isinf(self.t1_us):
            return 0.0
        return 1.0 / self.t1_us

    @property
    def gamma_phi_per_us(self) -> float:
        if self.t2_us is None or math.isinf(self.t2_us):
            return 0.0
        rate = (1.0 / self.t2_us) - 0.5 * self.gamma1_per_us
        return max(0.0, rate)


@dataclass(frozen=True)
class TwoQubitParams:
    """Two-qubit system with static ZZ shift: H = w0 n0 + w1 n1 + zeta_zz (n0 x n1)."""

    qubit0: SingleQubitParams
    qubit1: SingleQubitParams
    zz_coupling_mhz: float = 0.0

    @property
    def zeta_zz_rad_per_us(self) -> float:
        return float(2.0 * np.pi * self.zz_coupling_mhz)
