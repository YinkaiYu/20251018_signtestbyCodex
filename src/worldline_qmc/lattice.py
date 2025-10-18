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

    The actual grid construction is deferred to Stage 1.
    """

    raise NotImplementedError("Stage 1 provides the momentum grid generator.")


def dispersion(momentum: np.ndarray, hopping: float) -> np.ndarray:
    """
    Compute epsilon_k = -2t (cos kx + cos ky) for the provided momentum grid.

    Stage 1 will replace this placeholder implementation.
    """

    raise NotImplementedError("Stage 1 provides the dispersion calculation.")


def half_filling_particle_count(lattice_size: int) -> Tuple[int, int]:
    """
    Return (N_up, N_down) at half filling for a square lattice of size L x L.
    """

    particles = lattice_size * lattice_size // 2
    return particles, particles

