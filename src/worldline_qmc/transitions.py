"""
Transition matrix elements M_{l, sigma}(k' <- k).

Implementations arrive in Stage 2 once auxiliary field data structures are in
place. This module defines the function signatures and shared constants.
"""

from __future__ import annotations

import numpy as np

from .auxiliary import AuxiliaryField, Spin
from .config import SimulationParameters


def transition_amplitude(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    spin: Spin,
    time_slice: int,
    k_from: int,
    k_to: int,
) -> complex:
    """
    Compute M_{l, sigma}(k_to <- k_from).

    Stage 2 implements the evaluation using cached Fourier data.
    """

    raise NotImplementedError("Stage 2 computes transition amplitudes.")


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

