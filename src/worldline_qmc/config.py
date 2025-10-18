"""
Configuration utilities for the worldline QMC simulation.

Stage 1 provides concrete parsing, validation, and derived quantities for the
simulation parameters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np

_VALID_JSON_SUFFIXES = {".json"}
_TIME_SLICE_TOL = 1e-9


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
        """Return L_tau = beta / delta_tau after validation."""

        slices = self.beta / self.delta_tau
        return int(round(slices))

    @property
    def volume(self) -> int:
        """Total number of lattice sites L^2."""

        return self.lattice_size * self.lattice_size

    @property
    def particles_per_spin(self) -> int:
        """Number of particles per spin at half filling."""

        return self.volume // 2

    @property
    def auxiliary_coupling(self) -> float:
        r"""
        Return λ = arccosh(exp(Δτ U / 2)).

        The quantity is real-valued for the repulsive Hubbard interaction
        (U ≥ 0) assumed in `note.md`.
        """

        exponent = math.exp(self.delta_tau * self.interaction / 2.0)
        if exponent < 1.0:
            msg = (
                "Interaction exponent exp(Δτ U / 2) must be ≥ 1. "
                "The current parameters correspond to attractive interaction, "
                "which is outside the supported regime."
            )
            raise ValueError(msg)
        return float(np.arccosh(exponent))


def load_parameters(source: Path | Mapping[str, Any]) -> SimulationParameters:
    """
    Load and validate simulation parameters from disk or an in-memory mapping.

    Parameters may be supplied as:
    - A mapping (e.g., `dict`) with the required keys.
    - A path to a JSON file containing the mapping representation.
    """

    if isinstance(source, Path):
        mapping = _load_from_path(source)
    else:
        mapping = dict(source)

    params = _build_parameters(mapping)
    _validate_parameters(params)
    return params


def _load_from_path(path: Path) -> Mapping[str, Any]:
    """Load a configuration mapping from the supported on-disk formats."""

    if path.suffix.lower() in _VALID_JSON_SUFFIXES:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            msg = f"Configuration file {path} does not contain a mapping."
            raise ValueError(msg)
        return data

    msg = f"Unsupported configuration format: {path.suffix}"
    raise ValueError(msg)


def _build_parameters(mapping: Mapping[str, Any]) -> SimulationParameters:
    """Populate the SimulationParameters dataclass from a mapping."""

    dataclass_fields = {f.name for f in fields(SimulationParameters)}
    required = {"lattice_size", "beta", "delta_tau", "hopping", "interaction"}

    missing = required - mapping.keys()
    if missing:
        missing_csv = ", ".join(sorted(missing))
        msg = f"Missing required configuration fields: {missing_csv}"
        raise ValueError(msg)

    kwargs: dict[str, Any] = {}
    extras: dict[str, Any] = {}

    for key, value in mapping.items():
        if key in {"extra"}:
            if not isinstance(value, Mapping):
                msg = "The 'extra' field must be a mapping if provided."
                raise ValueError(msg)
            extras.update(value)
            continue
        if key in dataclass_fields:
            kwargs[key] = value
        else:
            extras[key] = value

    kwargs.setdefault("extra", {})
    kwargs["extra"].update(extras)

    _coerce_integer_fields(kwargs)
    _normalize_output_path(kwargs)
    return SimulationParameters(**kwargs)


def _coerce_integer_fields(kwargs: dict[str, Any]) -> None:
    """Convert integer-like values and reject non-integer inputs."""

    integer_fields = {
        "lattice_size",
        "seed",
        "sweeps",
        "thermalization_sweeps",
        "worldline_moves_per_slice",
        "permutation_moves_per_slice",
    }

    for field_name in integer_fields:
        if field_name not in kwargs or kwargs[field_name] is None:
            continue
        value = kwargs[field_name]
        coerced = _coerce_int(value, field_name)
        kwargs[field_name] = coerced


def _coerce_int(value: Any, field_name: str) -> Optional[int]:
    """Return an integer if `value` is integer-like, otherwise raise."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)

    msg = f"Field '{field_name}' expects an integer value, received {value!r}."
    raise ValueError(msg)


def _normalize_output_path(kwargs: dict[str, Any]) -> None:
    """Convert output_path strings into Path objects."""

    if "output_path" not in kwargs or kwargs["output_path"] is None:
        return
    output_value = kwargs["output_path"]
    if isinstance(output_value, Path):
        return
    if isinstance(output_value, str):
        kwargs["output_path"] = Path(output_value)
        return

    msg = "output_path must be a string or pathlib.Path if provided."
    raise ValueError(msg)


def _validate_parameters(params: SimulationParameters) -> None:
    """Run cross-field validations on simulation parameters."""

    if params.lattice_size <= 0:
        raise ValueError("lattice_size must be positive.")
    if params.beta <= 0:
        raise ValueError("beta must be positive.")
    if params.delta_tau <= 0:
        raise ValueError("delta_tau must be positive.")
    if params.sweeps < 0:
        raise ValueError("sweeps must be non-negative.")
    if params.thermalization_sweeps < 0:
        raise ValueError("thermalization_sweeps must be non-negative.")
    if params.worldline_moves_per_slice < 0:
        raise ValueError("worldline_moves_per_slice must be non-negative.")
    if params.permutation_moves_per_slice < 0:
        raise ValueError("permutation_moves_per_slice must be non-negative.")

    slices = params.beta / params.delta_tau
    nearest = round(slices)
    if not math.isclose(slices, nearest, rel_tol=0.0, abs_tol=_TIME_SLICE_TOL):
        msg = (
            "beta must be an integer multiple of delta_tau. "
            f"Received beta/delta_tau = {slices}."
        )
        raise ValueError(msg)

    if params.volume % 2 != 0:
        msg = (
            "Half filling requires an even number of lattice sites. "
            f"Lattice size {params.lattice_size} is incompatible."
        )
        raise ValueError(msg)
