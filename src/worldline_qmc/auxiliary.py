"""
Auxiliary field generation and FFT precomputation.

Stage 1 implements sampling of s_il and FFT-based evaluation of
W_{l, sigma}(q).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal

import numpy as np

from .config import SimulationParameters
from .rng import make_generator

Spin = Literal["up", "down"]
SPINS: Tuple[Spin, Spin] = ("up", "down")


@dataclass
class AuxiliaryFieldSlice:
    """Holds data for a single imaginary-time slice."""

    slice_index: int
    spatial_field: np.ndarray  # shape: (L, L) with entries ±1
    w_fourier: Dict[Spin, np.ndarray]  # shape: (L, L)

    def magnitude(self, spin: Spin) -> np.ndarray:
        """Return |W_{l, sigma}(q)| for the requested slice and spin."""

        return np.abs(self.w_fourier[spin])

    def phase(self, spin: Spin) -> np.ndarray:
        """Return arg W_{l, sigma}(q) for the requested slice and spin."""

        return np.angle(self.w_fourier[spin])


@dataclass
class AuxiliaryField:
    """Container for all auxiliary field slices."""

    slices: Tuple[AuxiliaryFieldSlice, ...]
    lattice_size: int
    auxiliary_coupling: float
    fft_mode: str

    def magnitude(self, slice_index: int, spin: Spin) -> np.ndarray:
        """Return |W_{l, sigma}(q)| for the requested slice and spin."""

        return self.slices[slice_index].magnitude(spin)

    def phase(self, slice_index: int, spin: Spin) -> np.ndarray:
        """Return arg W_{l, sigma}(q) for the requested slice and spin."""

        return self.slices[slice_index].phase(spin)

    def w(self, slice_index: int, spin: Spin) -> np.ndarray:
        """Return the complex Fourier array W_{l, sigma}(q)."""

        return self.slices[slice_index].w_fourier[spin]

    @property
    def time_slices(self) -> int:
        return len(self.slices)


def generate_auxiliary_field(
    params: SimulationParameters,
    *,
    seed: int | None = None,
) -> AuxiliaryField:
    """
    Sample {s_il} and cache Fourier transforms W_{l, sigma}(q).

    Parameters
    ----------
    params:
        Validated simulation parameters containing lattice and interaction
        information.
    seed:
        Optional override for the RNG seed. Defaults to `params.seed` when not
        provided.
    """

    lattice_size = params.lattice_size
    time_slices = params.time_slices
    coupling = params.auxiliary_coupling
    fft_mode = params.fft_mode

    rng_seed = params.seed if seed is None else seed
    rng = make_generator(rng_seed)

    slices = []
    for l in range(time_slices):
        spatial_field = rng.choice(
            (-1, 1), size=(lattice_size, lattice_size)
        ).astype(np.int8)

        spatial_float = spatial_field.astype(float)
        exp_up = np.exp(coupling * spatial_float)
        exp_down = np.exp(-coupling * spatial_float)

        w_up = _fourier_sum(exp_up, mode=fft_mode)
        w_down = _fourier_sum(exp_down, mode=fft_mode)

        slice_cache = AuxiliaryFieldSlice(
            slice_index=l,
            spatial_field=spatial_field,
            w_fourier={"up": w_up, "down": w_down},
        )
        slices.append(slice_cache)

    return AuxiliaryField(
        slices=tuple(slices),
        lattice_size=lattice_size,
        auxiliary_coupling=coupling,
        fft_mode=fft_mode,
    )


def _fourier_sum(field: np.ndarray, mode: str) -> np.ndarray:
    """
    Compute Σ_i exp(i q · r_i) field_i via FFT conventions.

    NumPy's FFT implements the negative exponential convention; we take the
    complex conjugate to match the positive phase definition in `note.md`.
    """
    fft = np.fft.fftn(field, norm=None).conj()
    if mode == "complex":
        return fft
    if mode == "real":
        return np.real(fft)
    msg = f"Unsupported fft_mode: {mode}"
    raise ValueError(msg)
