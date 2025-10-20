"""Monte Carlo update kernels for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np

from .auxiliary import AuxiliaryField, Spin
from .config import SimulationParameters
from .transitions import transition_log_components
from .worldline import PermutationState, Worldline, WorldlineConfiguration


@dataclass
class UpdateSchedule:
    """Number of attempts for each move type within one sweep."""

    worldline_moves: int
    permutation_moves: int


@dataclass
class MonteCarloState:
    """Holds mutable Monte Carlo state for incremental updates."""

    configuration: WorldlineConfiguration
    phase: complex
    log_weight: float
    rng: np.random.Generator
    occupancy_masks: Dict[Spin, np.ndarray]


def metropolis_sweep(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    mc_state: MonteCarloState,
    schedule: UpdateSchedule,
    *,
    measurement_interval: int,
) -> Tuple[Dict[str, float], List[complex]]:
    r"""Execute one Metropolis sweep of worldline and permutation updates.

    The acceptance ratios and phase increments follow the formulas collected in
    ``note.md`` (see the ``\mathcal{R}_k`` and ``\mathcal{R}_p`` expressions, as
    well as the corresponding ``\Delta\Phi`` updates).
    """

    diagnostics = {
        "momentum_attempts": 0,
        "momentum_accepts": 0,
        "permutation_attempts": 0,
        "permutation_accepts": 0,
    }

    config = mc_state.configuration
    rng = mc_state.rng
    spins = tuple(sorted(config.worldlines.keys()))
    time_slices = params.time_slices
    volume = params.volume
    phase_samples: List[complex] = []
    interval = measurement_interval if measurement_interval > 0 else 0
    attempt_counter = 0

    def maybe_record_phase() -> None:
        nonlocal attempt_counter
        if interval > 0 and attempt_counter % interval == 0:
            phase_samples.append(mc_state.phase)

    for _ in range(schedule.worldline_moves):
        diagnostics["momentum_attempts"] += 1
        spin = spins[int(rng.integers(len(spins)))]
        wl = config.worldlines[spin]
        num_particles = wl.particles
        l = int(rng.integers(time_slices))
        n = int(rng.integers(num_particles))
        new_k = int(rng.integers(volume))
        if _attempt_momentum_update(
            params,
            auxiliary,
            mc_state,
            spin,
            wl,
            config.permutations[spin],
            l,
            n,
            new_k,
        ):
            diagnostics["momentum_accepts"] += 1
        attempt_counter += 1
        maybe_record_phase()

    for _ in range(schedule.permutation_moves):
        diagnostics["permutation_attempts"] += 1
        spin = spins[int(rng.integers(len(spins)))]
        wl = config.worldlines[spin]
        perm = config.permutations[spin]
        move = _propose_permutation_move(perm, rng)
        if move is None:
            continue
        indices, new_targets = move
        if _attempt_permutation_move(
            params,
            auxiliary,
            mc_state,
            spin,
            wl,
            perm,
            indices,
            new_targets,
        ):
            diagnostics["permutation_accepts"] += 1
        attempt_counter += 1
        maybe_record_phase()

    diagnostics["momentum_acceptance"] = _ratio(
        diagnostics["momentum_accepts"], diagnostics["momentum_attempts"]
    )
    diagnostics["permutation_acceptance"] = _ratio(
        diagnostics["permutation_accepts"], diagnostics["permutation_attempts"]
    )
    return diagnostics, phase_samples


def _attempt_momentum_update(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    mc_state: MonteCarloState,
    spin: Spin,
    wl: Worldline,
    perm: PermutationState,
    l: int,
    n: int,
    new_k: int,
) -> bool:
    volume = params.volume
    if new_k < 0 or new_k >= volume:
        return False

    old_k = int(wl.trajectories[l, n])
    if new_k == old_k:
        return False

    occupancy_mask = mc_state.occupancy_masks[spin]
    if occupancy_mask[l, new_k]:
        return False

    forward_ts, forward_target = _forward_link_info(wl, perm, l, n, params.time_slices)
    backward_ts, backward_source = _backward_link_info(
        wl, perm, l, n, params.time_slices
    )

    old_forward_log, old_forward_phase = transition_log_components(
        params, auxiliary, spin, forward_ts, old_k, forward_target
    )
    old_backward_log, old_backward_phase = transition_log_components(
        params, auxiliary, spin, backward_ts, backward_source, old_k
    )
    new_forward_log, new_forward_phase = transition_log_components(
        params, auxiliary, spin, forward_ts, new_k, forward_target
    )
    new_backward_log, new_backward_phase = transition_log_components(
        params, auxiliary, spin, backward_ts, backward_source, new_k
    )

    if np.isneginf(new_forward_log) or np.isneginf(new_backward_log):
        return False

    total_log_ratio = (new_forward_log - old_forward_log) + (
        new_backward_log - old_backward_log
    )
    if total_log_ratio < 0.0 and np.log(mc_state.rng.random()) >= total_log_ratio:
        return False

    delta_phi = (new_forward_phase - old_forward_phase) + (
        new_backward_phase - old_backward_phase
    )

    occupancy_mask[l, old_k] = False
    occupancy_mask[l, new_k] = True
    wl.update_momentum(l, n, new_k, enforce_pauli=False)

    mc_state.phase *= np.exp(1j * delta_phi)
    mc_state.log_weight += total_log_ratio
    return True


def _attempt_permutation_move(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    mc_state: MonteCarloState,
    spin: Spin,
    wl: Worldline,
    perm: PermutationState,
    indices: np.ndarray,
    new_targets: np.ndarray,
) -> bool:
    if indices.size == 0:
        return False

    l_boundary = params.time_slices - 1
    old_targets = perm.values[indices].copy()
    if np.array_equal(old_targets, new_targets):
        return False

    log_ratio = 0.0
    delta_phi = 0.0
    for idx, old_target, new_target in zip(indices, old_targets, new_targets):
        k_last = int(wl.trajectories[l_boundary, idx])
        k0_old = int(wl.trajectories[0, old_target])
        k0_new = int(wl.trajectories[0, new_target])

        old_log, old_phase = transition_log_components(
            params, auxiliary, spin, l_boundary, k_last, k0_old
        )
        new_log, new_phase = transition_log_components(
            params, auxiliary, spin, l_boundary, k_last, k0_new
        )

        if np.isneginf(new_log):
            return False

        log_ratio += new_log - old_log
        delta_phi += new_phase - old_phase

    if log_ratio < 0.0 and np.log(mc_state.rng.random()) >= log_ratio:
        return False

    parity_change = _permutation_parity_change(old_targets, new_targets)

    perm.values[indices] = new_targets
    mc_state.phase *= parity_change * np.exp(1j * delta_phi)
    mc_state.log_weight += log_ratio
    return True


def _propose_permutation_move(
    perm: PermutationState, rng: np.random.Generator
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    size = perm.size
    if size < 2:
        return None
    if size < 3:
        return _swap_move(perm, rng)

    selector = rng.random()
    if selector < 0.5:
        return _swap_move(perm, rng)
    if selector < 0.8:
        return _cycle_move(perm, rng)
    return _shuffle_move(perm, rng)


def _swap_move(
    perm: PermutationState, rng: np.random.Generator
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    size = perm.size
    a = int(rng.integers(size))
    b = int(rng.integers(size))
    retries = 0
    while a == b and retries < 8:
        b = int(rng.integers(size))
        retries += 1
    if a == b:
        return None
    indices = np.array([a, b], dtype=np.int64)
    old_targets = perm.values[indices]
    new_targets = old_targets[::-1].copy()
    return indices, new_targets


def _cycle_move(
    perm: PermutationState, rng: np.random.Generator
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    size = perm.size
    max_len = min(size, 5)
    cycle_len = int(rng.integers(3, max_len + 1))
    indices = _sample_unique_indices(size, cycle_len, rng)
    if indices is None:
        return None
    old_targets = perm.values[indices]
    shift = int(rng.integers(cycle_len))
    new_targets = np.roll(old_targets, -((shift % cycle_len) + 1))
    return indices, new_targets


def _shuffle_move(
    perm: PermutationState, rng: np.random.Generator
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    size = perm.size
    max_len = min(size, 6)
    subset_len = int(rng.integers(3, max_len + 1))
    indices = _sample_unique_indices(size, subset_len, rng)
    if indices is None:
        return None
    old_targets = perm.values[indices]
    new_targets = _random_permutation(old_targets, rng)
    if np.array_equal(new_targets, old_targets):
        new_targets = np.roll(old_targets, -1)
        if np.array_equal(new_targets, old_targets):
            return None
    return indices, new_targets


def _sample_unique_indices(
    size: int, count: int, rng: np.random.Generator
) -> Optional[np.ndarray]:
    if count > size:
        return None
    selected: List[int] = []
    attempts = 0
    max_attempts = max(10, count * 5)
    while len(selected) < count and attempts < max_attempts:
        candidate = int(rng.integers(size))
        if candidate not in selected:
            selected.append(candidate)
        attempts += 1
    if len(selected) < count:
        return None
    return np.array(selected, dtype=np.int64)


def _random_permutation(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = values.copy()
    for i in range(permuted.size - 1, 0, -1):
        j = int(rng.integers(i + 1))
        permuted[i], permuted[j] = permuted[j], permuted[i]
    return permuted


def _permutation_parity_change(
    old_targets: np.ndarray, new_targets: np.ndarray
) -> int:
    mapping = {int(value): idx for idx, value in enumerate(old_targets)}
    sigma = [mapping[int(value)] for value in new_targets]
    visited = [False] * len(sigma)
    sign = 1
    for start in range(len(sigma)):
        if visited[start]:
            continue
        cycle_len = 0
        idx = start
        while not visited[idx]:
            visited[idx] = True
            idx = sigma[idx]
            cycle_len += 1
        if cycle_len > 0 and cycle_len % 2 == 0:
            sign *= -1
    return sign


def _forward_link_info(
    wl: Worldline, perm: PermutationState, l: int, n: int, time_slices: int
) -> Tuple[int, int]:
    if time_slices == 0:
        raise ValueError("time_slices must be positive.")
    if l < time_slices - 1:
        return l, int(wl.trajectories[l + 1, n])
    target = int(perm.values[n])
    return time_slices - 1, int(wl.trajectories[0, target])


def _backward_link_info(
    wl: Worldline, perm: PermutationState, l: int, n: int, time_slices: int
) -> Tuple[int, int]:
    if time_slices == 0:
        raise ValueError("time_slices must be positive.")
    if l > 0:
        return l - 1, int(wl.trajectories[l - 1, n])
    # For l = 0 we must locate which particle at the previous slice maps to n.
    # The permutation maps slice L_τ-1 → 0, so we consult P^{-1}(n).
    inverse = perm.inverse()
    prev_particle = int(inverse[n])
    return time_slices - 1, int(wl.trajectories[time_slices - 1, prev_particle])


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
