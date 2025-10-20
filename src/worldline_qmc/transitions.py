r"""Transition matrix elements :math:`M_{l,\sigma}(k' \leftarrow k)`."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from .auxiliary import AuxiliaryField, Spin
from .config import SimulationParameters
from . import lattice
from .worldline import unflatten_momentum_index


def transition_amplitude(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    spin: Spin,
    time_slice: int,
    k_from: int,
    k_to: int,
) -> complex:
    r"""
    Compute :math:`M_{l,\sigma}(k' \leftarrow k)` for the specified data.

    Implementation follows Eq. (M) in ``note.md``:

    .. math::
        M_{l,\sigma}(k' \leftarrow k) =
        e^{-\frac{\Delta\tau}{2}(\varepsilon_{k'} + \varepsilon_k)}
        \frac{W_{l,\sigma}(k - k')}{V}.
    """

    log_mag, phase = transition_log_components(
        params, auxiliary, spin, time_slice, k_from, k_to
    )
    if np.isneginf(log_mag):
        return 0.0 + 0.0j
    return np.exp(log_mag + 1j * phase)


def transition_log_components(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    spin: Spin,
    time_slice: int,
    k_from: int,
    k_to: int,
) -> Tuple[float, float]:
    """Return (log |M|, arg M) without constructing the complex amplitude."""

    _validate_indices(params, auxiliary, time_slice, k_from, k_to)

    lattice_size = params.lattice_size
    eps = _energies(lattice_size, params.hopping)
    eps_from = eps[k_from]
    eps_to = eps[k_to]

    w_array = auxiliary.w(time_slice, spin)
    qx, qy = _momentum_difference(k_from, k_to, lattice_size)
    w_value = w_array[qx, qy]
    magnitude = abs(w_value)
    if magnitude <= 0.0:
        return float("-inf"), 0.0

    exponent = -0.5 * params.delta_tau * (eps_from + eps_to)
    log_volume = float(np.log(params.volume))
    log_mag = exponent + float(np.log(magnitude)) - log_volume
    phase = float(np.angle(w_value))
    return log_mag, phase


def transition_ratio(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    spin: Spin,
    time_slice: int,
    k_from: int,
    k_to: int,
    k_from_new: int,
    k_to_new: int,
) -> complex:
    """
    Ratio of new to old transition amplitudes (used in Metropolis steps).

    Stage 3 will provide the concrete implementation.
    """

    raise NotImplementedError("Stage 3 computes transition ratios.")


def _validate_indices(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    time_slice: int,
    k_from: int,
    k_to: int,
) -> None:
    if time_slice < 0 or time_slice >= auxiliary.time_slices:
        raise IndexError("time_slice out of range for auxiliary field.")
    volume = params.volume
    if k_from < 0 or k_from >= volume:
        raise IndexError("k_from index out of range for lattice volume.")
    if k_to < 0 or k_to >= volume:
        raise IndexError("k_to index out of range for lattice volume.")
    if auxiliary.lattice_size != params.lattice_size:
        raise ValueError("Auxiliary field lattice size mismatch with parameters.")


def _momentum_difference(k_from: int, k_to: int, lattice_size: int) -> Tuple[int, int]:
    ix_from, iy_from = unflatten_momentum_index(k_from, lattice_size)
    ix_to, iy_to = unflatten_momentum_index(k_to, lattice_size)
    qx = (ix_from - ix_to) % lattice_size
    qy = (iy_from - iy_to) % lattice_size
    return qx, qy


@lru_cache(maxsize=None)
def _energies(lattice_size: int, hopping: float) -> np.ndarray:
    grid = lattice.momentum_grid(lattice_size)
    return lattice.dispersion(grid, hopping)
