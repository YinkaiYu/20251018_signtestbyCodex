"""Monte Carlo update kernels for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .auxiliary import AuxiliaryField, Spin
from .config import SimulationParameters
from .transitions import transition_amplitude
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


def metropolis_sweep(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    mc_state: MonteCarloState,
    schedule: UpdateSchedule,
) -> Dict[str, float]:
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

    for _ in range(schedule.permutation_moves):
        diagnostics["permutation_attempts"] += 1
        spin = spins[int(rng.integers(len(spins)))]
        wl = config.worldlines[spin]
        perm = config.permutations[spin]
        num_particles = wl.particles
        if num_particles < 2:
            continue
        a = int(rng.integers(num_particles))
        b = int(rng.integers(num_particles))
        retries = 0
        while a == b and retries < 5:
            b = int(rng.integers(num_particles))
            retries += 1
        if a == b:
            continue
        if _attempt_permutation_swap(
            params,
            auxiliary,
            mc_state,
            spin,
            wl,
            perm,
            a,
            b,
        ):
            diagnostics["permutation_accepts"] += 1

    diagnostics["momentum_acceptance"] = _ratio(
        diagnostics["momentum_accepts"], diagnostics["momentum_attempts"]
    )
    diagnostics["permutation_acceptance"] = _ratio(
        diagnostics["permutation_accepts"], diagnostics["permutation_attempts"]
    )
    return diagnostics


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

    slice_data = wl.trajectories[l]
    if new_k in slice_data:
        return False

    forward_ts, forward_target = _forward_link_info(wl, perm, l, n, params.time_slices)
    backward_ts, backward_source = _backward_link_info(
        wl, perm, l, n, params.time_slices
    )

    old_forward = transition_amplitude(
        params, auxiliary, spin, forward_ts, old_k, forward_target
    )
    old_backward = transition_amplitude(
        params, auxiliary, spin, backward_ts, backward_source, old_k
    )
    new_forward = transition_amplitude(
        params, auxiliary, spin, forward_ts, new_k, forward_target
    )
    new_backward = transition_amplitude(
        params, auxiliary, spin, backward_ts, backward_source, new_k
    )

    # note.md: 使用 \mathcal{R}_k 接受率，依赖相邻两条传输矩阵元的模长比值。
    mag_ratio = _safe_mag_ratio(new_forward, old_forward) * _safe_mag_ratio(
        new_backward, old_backward
    )
    if mag_ratio <= 0.0:
        return False

    accept_prob = min(1.0, mag_ratio)
    if mc_state.rng.random() >= accept_prob:
        return False

    # note.md: ΔΦ_k 增量由更新前后矩阵元相位之差组成。
    delta_phi = _phase_difference(new_forward, old_forward) + _phase_difference(
        new_backward, old_backward
    )

    wl.update_momentum(l, n, new_k)

    mc_state.phase *= np.exp(1j * delta_phi)
    mc_state.log_weight += _log_ratio(new_forward, old_forward)
    mc_state.log_weight += _log_ratio(new_backward, old_backward)
    return True


def _attempt_permutation_swap(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    mc_state: MonteCarloState,
    spin: Spin,
    wl: Worldline,
    perm: PermutationState,
    a: int,
    b: int,
) -> bool:
    l_boundary = params.time_slices - 1
    k_last_a = int(wl.trajectories[l_boundary, a])
    k_last_b = int(wl.trajectories[l_boundary, b])

    target_a = int(perm.values[a])
    target_b = int(perm.values[b])
    k0_target_a = int(wl.trajectories[0, target_a])
    k0_target_b = int(wl.trajectories[0, target_b])

    old_a = transition_amplitude(
        params, auxiliary, spin, l_boundary, k_last_a, k0_target_a
    )
    old_b = transition_amplitude(
        params, auxiliary, spin, l_boundary, k_last_b, k0_target_b
    )
    new_a = transition_amplitude(
        params, auxiliary, spin, l_boundary, k_last_a, k0_target_b
    )
    new_b = transition_amplitude(
        params, auxiliary, spin, l_boundary, k_last_b, k0_target_a
    )

    mag_ratio = _safe_mag_ratio(new_a, old_a) * _safe_mag_ratio(new_b, old_b)
    if mag_ratio <= 0.0:
        return False

    accept_prob = min(1.0, mag_ratio)
    if mc_state.rng.random() >= accept_prob:
        return False

    # note.md: ΔΦ_perm 相位增量及 permutation 奇偶变化导致额外的 -1 因子。
    delta_phi = _phase_difference(new_a, old_a) + _phase_difference(new_b, old_b)

    perm.swap(a, b)
    mc_state.phase *= -np.exp(1j * delta_phi)
    mc_state.log_weight += _log_ratio(new_a, old_a)
    mc_state.log_weight += _log_ratio(new_b, old_b)
    return True


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
    inverse = perm.inverse()
    prev_particle = int(inverse[n])
    return time_slices - 1, int(wl.trajectories[time_slices - 1, prev_particle])


def _safe_mag_ratio(new: complex, old: complex) -> float:
    new_mag = abs(new)
    old_mag = abs(old)
    if new_mag <= 0.0 or old_mag <= 0.0:
        return 0.0
    return new_mag / old_mag


def _phase_difference(new: complex, old: complex) -> float:
    if abs(new) == 0.0 or abs(old) == 0.0:
        return 0.0
    return float(np.angle(new / old))


def _log_ratio(new: complex, old: complex) -> float:
    new_mag = abs(new)
    old_mag = abs(old)
    if new_mag <= 0.0 or old_mag <= 0.0:
        return 0.0
    return float(np.log(new_mag) - np.log(old_mag))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
