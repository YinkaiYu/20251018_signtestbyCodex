"""Worldline and permutation data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

Spin = str


def flatten_momentum_index(ix: int, iy: int, lattice_size: int) -> int:
    """Return the flattened index for a momentum grid point."""

    _ensure_non_negative(ix, "ix")
    _ensure_non_negative(iy, "iy")
    if lattice_size <= 0:
        raise ValueError("lattice_size must be positive.")
    if ix >= lattice_size or iy >= lattice_size:
        raise ValueError("Momentum component out of range for lattice size.")
    return ix * lattice_size + iy


def unflatten_momentum_index(index: int, lattice_size: int) -> Tuple[int, int]:
    """Return `(ix, iy)` for a flattened momentum index."""

    if lattice_size <= 0:
        raise ValueError("lattice_size must be positive.")
    if index < 0 or index >= lattice_size * lattice_size:
        raise ValueError("Momentum index out of bounds for lattice size.")
    ix, iy = divmod(int(index), lattice_size)
    return ix, iy


@dataclass
class PermutationState:
    """Stores the permutation indices for one spin species."""

    values: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.int64)
        if self.values.ndim != 1:
            raise ValueError("Permutation array must be 1-dimensional.")
        self._validate_is_permutation()

    @property
    def size(self) -> int:
        return self.values.size

    def parity(self) -> int:
        """Return +1 for even permutations, -1 for odd permutations."""

        visited = np.zeros(self.size, dtype=bool)
        sign = 1
        for start in range(self.size):
            if visited[start]:
                continue
            cycle_length = 0
            idx = start
            while not visited[idx]:
                visited[idx] = True
                idx = self.values[idx]
                cycle_length += 1
            if cycle_length > 0 and cycle_length % 2 == 0:
                sign *= -1
        return sign

    @classmethod
    def identity(cls, size: int) -> "PermutationState":
        """Return the identity permutation of the requested size."""

        if size < 0:
            raise ValueError("size must be non-negative.")
        return cls(np.arange(size, dtype=np.int64))

    def copy(self) -> "PermutationState":
        return PermutationState(self.values.copy())

    def inverse(self) -> np.ndarray:
        """Return the inverse permutation indices.

        This is used when applying the boundary link updates described in
        ``note.md`` for momentum moves at ``l = 0``.
        """

        inverse = np.empty_like(self.values)
        inverse[self.values] = np.arange(self.values.size, dtype=np.int64)
        return inverse

    def swap(self, a: int, b: int) -> None:
        """Swap the images of particle labels ``a`` and ``b`` in place."""

        if a == b:
            return
        if not (0 <= a < self.size and 0 <= b < self.size):
            raise IndexError("Permutation swap indices out of range.")
        self.values[a], self.values[b] = self.values[b], self.values[a]

    def _validate_is_permutation(self) -> None:
        expected = np.arange(self.values.size, dtype=np.int64)
        if not np.array_equal(np.sort(self.values), expected):
            msg = "values must be a permutation of [0, N)."
            raise ValueError(msg)


@dataclass
class Worldline:
    """Stores occupied momenta for one spin species across time slices."""

    trajectories: np.ndarray  # Shape: (L_tau, N_sigma)

    def __post_init__(self) -> None:
        self.trajectories = np.asarray(self.trajectories, dtype=np.int64)
        if self.trajectories.ndim != 2:
            raise ValueError("Worldline trajectories must be 2-dimensional.")
        self._validate_pauli_all_slices()

    @property
    def time_slices(self) -> int:
        return self.trajectories.shape[0]

    @property
    def particles(self) -> int:
        return self.trajectories.shape[1]

    def occupancy(self, time_slice: int) -> np.ndarray:
        """Return the set of occupied momenta at the specified time slice."""

        self._ensure_time_slice(time_slice)
        return self.trajectories[time_slice].copy()

    def update_momentum(
        self,
        time_slice: int,
        particle: int,
        new_k: int,
        *,
        enforce_pauli: bool = True,
    ) -> None:
        """Apply a momentum update while optionally enforcing Pauli exclusion."""

        self._ensure_time_slice(time_slice)
        self._ensure_particle_index(particle)
        slice_data = self.trajectories[time_slice]
        if (
            enforce_pauli
            and new_k != slice_data[particle]
            and np.any(slice_data == new_k)
        ):
            raise ValueError("Proposed momentum already occupied at this slice.")
        slice_data[particle] = int(new_k)
        if enforce_pauli:
            self._validate_slice(slice_data)

    def _validate_pauli_all_slices(self) -> None:
        for slice_data in self.trajectories:
            self._validate_slice(slice_data)

    def _validate_slice(self, slice_data: Iterable[int]) -> None:
        slice_array = np.asarray(slice_data, dtype=np.int64)
        if slice_array.size != np.unique(slice_array).size:
            raise ValueError("Worldline slice violates Pauli exclusion (duplicate momenta).")

    def _ensure_time_slice(self, time_slice: int) -> None:
        if time_slice < 0 or time_slice >= self.time_slices:
            raise IndexError("time_slice out of range.")

    def _ensure_particle_index(self, particle: int) -> None:
        if particle < 0 or particle >= self.particles:
            raise IndexError("particle index out of range.")


@dataclass
class WorldlineConfiguration:
    """Aggregate worldlines and permutations for both spin species."""

    worldlines: Dict[Spin, Worldline]
    permutations: Dict[Spin, PermutationState]


def _ensure_non_negative(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
