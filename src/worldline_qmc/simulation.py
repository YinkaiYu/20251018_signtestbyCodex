"""High-level simulation orchestration and result handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .auxiliary import AuxiliaryField
from .config import SimulationParameters
from .measurement import MeasurementAccumulator
from .rng import make_generator
from .transitions import transition_amplitude
from .updates import MonteCarloState, UpdateSchedule, metropolis_sweep
from .worldline import PermutationState, Worldline, WorldlineConfiguration

Spin = str
_SPINS: tuple[Spin, Spin] = ("up", "down")


@dataclass
class SimulationResult:
    """Container for measurement outputs, diagnostics, and metadata."""

    measurements: Dict[str, float]
    variances: Dict[str, float]
    diagnostics: Dict[str, float]
    samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the result."""

        return {
            "measurements": self.measurements,
            "variances": self.variances,
            "diagnostics": self.diagnostics,
            "samples": self.samples,
        }


def run_simulation(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    initial_state: Optional[WorldlineConfiguration] = None,
    schedule: Optional[UpdateSchedule] = None,
) -> SimulationResult:
    r"""Execute the Monte Carlo simulation and return aggregated results.

    The sampling follows the scheme in ``note.md``: each sweep applies momentum
    and permutation moves governed by the ``\mathcal{R}_k``/``\mathcal{R}_p``
    acceptance ratios, and the phase observable ``S(X)`` is accumulated via the
    same incremental updates used in Stage 3.
    """

    rng = make_generator(params.seed)
    configuration = initial_state or _initialize_configuration(params, rng)
    resolved_schedule = _resolve_schedule(params, schedule, configuration)

    mc_state = _initialize_mc_state(params, auxiliary, configuration, rng)
    accumulator = MeasurementAccumulator()

    total_momentum_attempts = 0
    total_momentum_accepts = 0
    total_permutation_attempts = 0
    total_permutation_accepts = 0

    total_sweeps = params.thermalization_sweeps + params.sweeps
    for sweep_index in range(total_sweeps):
        diagnostics = metropolis_sweep(params, auxiliary, mc_state, resolved_schedule)
        total_momentum_attempts += diagnostics.get("momentum_attempts", 0)
        total_momentum_accepts += diagnostics.get("momentum_accepts", 0)
        total_permutation_attempts += diagnostics.get("permutation_attempts", 0)
        total_permutation_accepts += diagnostics.get("permutation_accepts", 0)

        if sweep_index >= params.thermalization_sweeps:
            accumulator.push(mc_state.phase)

    measurement_count = accumulator.samples
    momentum_acceptance = (
        total_momentum_accepts / total_momentum_attempts
        if total_momentum_attempts
        else 0.0
    )
    permutation_acceptance = (
        total_permutation_accepts / total_permutation_attempts
        if total_permutation_attempts
        else 0.0
    )

    diagnostics_result = {
        "momentum_attempts": float(total_momentum_attempts),
        "momentum_accepts": float(total_momentum_accepts),
        "permutation_attempts": float(total_permutation_attempts),
        "permutation_accepts": float(total_permutation_accepts),
        "momentum_acceptance": momentum_acceptance,
        "permutation_acceptance": permutation_acceptance,
        "total_sweeps": float(total_sweeps),
        "measurement_sweeps": float(params.sweeps),
    }

    return SimulationResult(
        measurements=accumulator.averages(),
        variances=accumulator.variances(),
        diagnostics=diagnostics_result,
        samples=measurement_count,
    )


def _initialize_configuration(
    params: SimulationParameters, rng: np.random.Generator
) -> WorldlineConfiguration:
    time_slices = params.time_slices
    particles = params.particles_per_spin
    volume = params.volume

    worldlines: Dict[Spin, Worldline] = {}
    permutations: Dict[Spin, PermutationState] = {}
    for spin in _SPINS:
        base_slice = rng.choice(volume, size=particles, replace=False)
        trajectories = np.tile(base_slice, (time_slices, 1))
        worldlines[spin] = Worldline(trajectories)
        permutations[spin] = PermutationState.identity(particles)
    return WorldlineConfiguration(worldlines=worldlines, permutations=permutations)


def _initialize_mc_state(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    configuration: WorldlineConfiguration,
    rng: np.random.Generator,
) -> MonteCarloState:
    log_weight, phase = _compute_weight_and_phase(params, auxiliary, configuration)
    return MonteCarloState(
        configuration=configuration,
        phase=phase,
        log_weight=log_weight,
        rng=rng,
    )


def _compute_weight_and_phase(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    configuration: WorldlineConfiguration,
) -> tuple[float, complex]:
    time_slices = params.time_slices
    if time_slices <= 0:
        raise ValueError("time_slices must be positive.")

    total_log_mag = 0.0
    phase_angle = 0.0
    sign_factor = 1

    for spin, worldline in configuration.worldlines.items():
        perm = configuration.permutations[spin]
        sign_factor *= perm.parity()

        for n in range(worldline.particles):
            # note.md: weight w(X) includes product over slices of M_{l,σ}(k' ← k)
            for l in range(max(time_slices - 1, 0)):
                k_from = int(worldline.trajectories[l, n])
                k_to = int(worldline.trajectories[(l + 1) % time_slices, n])
                amplitude = transition_amplitude(params, auxiliary, spin, l, k_from, k_to)
                total_log_mag, phase_angle = _accumulate_link(
                    amplitude, total_log_mag, phase_angle
                )

            # note.md: boundary link couples slice Lτ-1 to slice 0 via P_σ
            l_boundary = time_slices - 1
            k_last = int(worldline.trajectories[l_boundary, n])
            target = int(perm.values[n])
            k_first = int(worldline.trajectories[0, target])
            amplitude = transition_amplitude(
                params, auxiliary, spin, l_boundary, k_last, k_first
            )
            total_log_mag, phase_angle = _accumulate_link(
                amplitude, total_log_mag, phase_angle
            )

    phase = sign_factor * np.exp(1j * phase_angle)
    return total_log_mag, phase


def _accumulate_link(
    amplitude: complex, current_log: float, current_phase_angle: float
) -> tuple[float, float]:
    magnitude = abs(amplitude)
    if magnitude <= 0.0:
        raise ValueError(
            "Encountered zero-magnitude transition amplitude during initialization."
        )
    return (
        current_log + float(np.log(magnitude)),
        current_phase_angle + float(np.angle(amplitude)),
    )


def _resolve_schedule(
    params: SimulationParameters,
    schedule: Optional[UpdateSchedule],
    configuration: WorldlineConfiguration,
) -> UpdateSchedule:
    if schedule is not None:
        return schedule

    time_slices = params.time_slices
    num_spins = len(configuration.worldlines)
    particles = next(iter(configuration.worldlines.values())).particles

    worldline_moves_per_slice = (
        params.worldline_moves_per_slice if params.worldline_moves_per_slice > 0 else particles
    )
    permutation_moves_per_slice = (
        params.permutation_moves_per_slice
        if params.permutation_moves_per_slice > 0
        else max(particles - 1, 1)
    )

    worldline_moves = worldline_moves_per_slice * time_slices * num_spins
    permutation_moves = permutation_moves_per_slice * num_spins

    return UpdateSchedule(
        worldline_moves=worldline_moves,
        permutation_moves=permutation_moves,
    )
