"""
Momentum lattice helpers.

Routines in this module will be implemented in Stage 1-2 to construct
momentum grids and evaluate the free-fermion dispersion.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def momentum_grid(lattice_size: int) -> np.ndarray:
    """
    Return an array of momentum vectors compatible with periodic boundaries.

    The grid is ordered in the standard FFT layout corresponding to
    `np.fft.fftfreq`, yielding components in the range [-π, π).
    """

    _validate_lattice_size(lattice_size)

    k_components = 2.0 * np.pi * np.fft.fftfreq(lattice_size)
    kx, ky = np.meshgrid(k_components, k_components, indexing="ij", copy=False)
    return np.stack((kx, ky), axis=-1).reshape(-1, 2)


def dispersion(momentum: np.ndarray, hopping: float) -> np.ndarray:
    """
    Compute epsilon_k = -2t (cos kx + cos ky) for the provided momentum grid.
    """

    momentum = np.asarray(momentum, dtype=float)
    if momentum.ndim != 2 or momentum.shape[1] != 2:
        raise ValueError("Momentum array must have shape (N, 2).")

    kx = momentum[:, 0]
    ky = momentum[:, 1]
    return -2.0 * hopping * (np.cos(kx) + np.cos(ky))


def half_filling_particle_count(lattice_size: int) -> Tuple[int, int]:
    """
    Return (N_up, N_down) at half filling for a square lattice of size L x L.
    """

    particles = lattice_size * lattice_size // 2
    return particles, particles


def _validate_lattice_size(lattice_size: int) -> None:
    if lattice_size <= 0:
        raise ValueError("lattice_size must be positive.")
