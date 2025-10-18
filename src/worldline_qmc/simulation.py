"""
High-level simulation orchestration.

Stage 3-4 will wire configuration, auxiliary field handling, updates, and
measurements into a full Monte Carlo loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .auxiliary import AuxiliaryField
from .config import SimulationParameters
from .measurement import MeasurementAccumulator
from .updates import UpdateSchedule
from .worldline import WorldlineConfiguration


@dataclass
class SimulationResult:
    """Container for measurement outputs and diagnostics."""

    measurements: Dict[str, float]
    diagnostics: Dict[str, float]


def run_simulation(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    initial_state: Optional[WorldlineConfiguration] = None,
    schedule: Optional[UpdateSchedule] = None,
) -> SimulationResult:
    """
    Execute the Monte Carlo simulation and return aggregated results.

    Implementation will be provided in Stage 3-4.
    """

    raise NotImplementedError("Simulation loop arrives in later stages.")

