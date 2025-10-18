"""
Worldline and permutation data structures.

Stage 2 will implement the logic for maintaining Pauli-safe configurations
and evaluating transition matrix elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

Spin = str


@dataclass
class PermutationState:
    """Stores the permutation indices for one spin species."""

    values: np.ndarray

    def parity(self) -> int:
        """Return +1 for even permutations, -1 for odd permutations."""

        raise NotImplementedError("Stage 2 implements permutation parity.")


@dataclass
class Worldline:
    """Stores occupied momenta for one spin species across time slices."""

    trajectories: np.ndarray  # Shape: (L_tau, N_sigma)

    def occupancy(self, time_slice: int) -> np.ndarray:
        """Return the set of occupied momenta at the specified time slice."""

        raise NotImplementedError("Stage 2 implements occupancy queries.")

    def update_momentum(self, time_slice: int, particle: int, new_k: int) -> None:
        """Apply a momentum update with Pauli checking (to be implemented)."""

        raise NotImplementedError("Stage 2 implements momentum updates.")


@dataclass
class WorldlineConfiguration:
    """Aggregate worldlines and permutations for both spin species."""

    worldlines: Dict[Spin, Worldline]
    permutations: Dict[Spin, PermutationState]

