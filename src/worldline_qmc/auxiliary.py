"""
Auxiliary field generation and FFT precomputation.

Stage 1 will implement sampling of s_il and FFT-based evaluation of
W_{l, sigma}(q).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


Spin = str  # Placeholder alias for spin labels ("up" / "down").


@dataclass
class AuxiliaryFieldSlice:
    """Holds data for a single imaginary-time slice."""

    slice_index: int
    spatial_field: np.ndarray
    w_fourier: Dict[Spin, np.ndarray]


@dataclass
class AuxiliaryField:
    """Container for all auxiliary field slices."""

    slices: Tuple[AuxiliaryFieldSlice, ...]

    def magnitude(self, slice_index: int, spin: Spin) -> np.ndarray:
        """Return |W_{l, sigma}(q)| for the requested slice and spin."""

        raise NotImplementedError("Stage 1 implements FFT magnitude access.")

    def phase(self, slice_index: int, spin: Spin) -> np.ndarray:
        """Return arg W_{l, sigma}(q) for the requested slice and spin."""

        raise NotImplementedError("Stage 1 implements FFT phase access.")


def generate_auxiliary_field(
    lattice_size: int,
    time_slices: int,
    interaction: float,
    delta_tau: float,
    seed: int | None = None,
) -> AuxiliaryField:
    """
    Sample {s_il} and cache Fourier transforms W_{l, sigma}(q).

    Stage 1 fills in the concrete implementation.
    """

    raise NotImplementedError("Stage 1 implements auxiliary field generation.")

