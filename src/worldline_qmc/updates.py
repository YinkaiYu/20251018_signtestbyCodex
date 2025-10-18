"""
Monte Carlo update kernels.

Stage 3 will flesh out worldline momentum updates and permutation swaps while
keeping track of phase increments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .auxiliary import AuxiliaryField, Spin
from .config import SimulationParameters
from .worldline import WorldlineConfiguration


Proposal = Callable[[WorldlineConfiguration], float]


@dataclass
class UpdateSchedule:
    """Describes how many times to attempt each move per sweep."""

    worldline_moves: int
    permutation_moves: int


def metropolis_sweep(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    config: WorldlineConfiguration,
    schedule: UpdateSchedule,
) -> Dict[str, float]:
    """
    Execute a full sweep of Monte Carlo updates.

    Returns diagnostic data such as acceptance ratios. Placeholder raises
    `NotImplementedError` until Stage 3.
    """

    raise NotImplementedError("Stage 3 implements the Metropolis sweep.")

