"""
Configuration utilities for the worldline QMC simulation.

Stage 1 will implement parsing and validation routines using the
`SimulationParameters` dataclass defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


@dataclass
class SimulationParameters:
    """Holds physical and algorithmic parameters for a simulation run."""

    lattice_size: int
    beta: float
    delta_tau: float
    hopping: float
    interaction: float
    seed: Optional[int] = None
    sweeps: int = 0
    thermalization_sweeps: int = 0
    worldline_moves_per_slice: int = 0
    permutation_moves_per_slice: int = 0
    output_path: Optional[Path] = None
    extra: MutableMapping[str, Any] = field(default_factory=dict)

    @property
    def time_slices(self) -> int:
        """Return L_tau = beta / delta_tau when available."""
        if self.delta_tau == 0:
            msg = "delta_tau must be non-zero to compute time slices."
            raise ValueError(msg)
        return int(round(self.beta / self.delta_tau))


def load_parameters(source: Path | Mapping[str, Any]) -> SimulationParameters:
    """
    Load and validate simulation parameters from disk or an in-memory mapping.

    Stage 1 will provide the concrete implementation. The placeholder raises
    `NotImplementedError` to signal the missing functionality.
    """

    raise NotImplementedError("Stage 1 implements configuration loading.")

