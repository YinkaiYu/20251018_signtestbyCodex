r"""Measurement routines for the sign/phase observable ``S(X)``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class MeasurementAccumulator:
    """Running statistics for S(X)."""

    samples: int = 0
    sum_real: float = 0.0
    sum_imag: float = 0.0
    sum_abs: float = 0.0
    sum_real_sq: float = 0.0
    sum_imag_sq: float = 0.0
    sum_abs_sq: float = 0.0

    def push(self, sign_phase: complex) -> None:
        r"""Record a new sample of the complex sign/phase.

        ``note.md`` defines :math:`S(X) = \mathrm{sgn}(P_\uparrow)\mathrm{sgn}(P_\downarrow)\exp[i\Phi(X)]`.
        This method accumulates the real, imaginary, and magnitude components for
        later averaging and optional variance estimates.
        """

        if not np.isfinite(sign_phase.real) or not np.isfinite(sign_phase.imag):
            raise ValueError("sign_phase must be finite complex value.")

        magnitude = abs(sign_phase)
        self.samples += 1
        self.sum_real += sign_phase.real
        self.sum_imag += sign_phase.imag
        self.sum_abs += magnitude
        self.sum_real_sq += sign_phase.real**2
        self.sum_imag_sq += sign_phase.imag**2
        self.sum_abs_sq += magnitude**2

    def averages(self) -> Dict[str, float]:
        """Return mean values of the real, imaginary, and magnitude components."""

        if self.samples == 0:
            return {"re": 0.0, "im": 0.0, "abs": 0.0}
        return {
            "re": self.sum_real / self.samples,
            "im": self.sum_imag / self.samples,
            "abs": self.sum_abs / self.samples,
        }

    def variances(self) -> Dict[str, float]:
        """Unbiased variance estimates for diagnostics (no autocorrelation corr.)."""

        if self.samples < 2:
            return {"re": 0.0, "im": 0.0, "abs": 0.0}
        n = self.samples
        mean_re = self.sum_real / n
        mean_im = self.sum_imag / n
        mean_abs = self.sum_abs / n
        var_re = max((self.sum_real_sq - n * mean_re**2) / (n - 1), 0.0)
        var_im = max((self.sum_imag_sq - n * mean_im**2) / (n - 1), 0.0)
        var_abs = max((self.sum_abs_sq - n * mean_abs**2) / (n - 1), 0.0)
        return {"re": var_re, "im": var_im, "abs": var_abs}
