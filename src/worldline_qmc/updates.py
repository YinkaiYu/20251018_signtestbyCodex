"""Monte Carlo update kernels for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np

from .auxiliary import AuxiliaryField, Spin, SPINS
from .config import SimulationParameters
from .transitions import transition_log_components
from .worldline import (
    PermutationState,
    Worldline,
    WorldlineConfiguration,
    flatten_momentum_index,
    unflatten_momentum_index,
)


@dataclass
class UpdateSchedule:
    """Number of attempts for each move type within one sweep."""

    worldline_moves: int
    permutation_moves: int
    auxiliary_moves: int


@dataclass
class MomentumProposalTable:
    """Precomputed proposal distribution for |W_{l,σ}(q)| sampling."""

    indices: np.ndarray
    cdf: np.ndarray
    log_probabilities: np.ndarray
    weighted: bool

    @classmethod
    def from_magnitude(cls, magnitude: np.ndarray) -> "MomentumProposalTable":
        flat = np.asarray(magnitude, dtype=float).ravel()
        if flat.size == 0:
            raise ValueError("Magnitude array must be non-empty.")

        positive = flat > 0.0
        if np.any(positive):
            weights = flat[positive]
            norm = float(weights.sum())
            if norm <= 0.0:
                raise ValueError("Magnitude weights must sum to a positive value.")
            probabilities = weights / norm
            cdf = np.cumsum(probabilities, dtype=float)
            cdf[-1] = 1.0  # guard against floating-point drift
            log_probs = np.full(flat.shape, float("-inf"), dtype=float)
            log_probs[positive] = np.log(weights) - np.log(norm)
            indices = np.flatnonzero(positive).astype(np.int64, copy=False)
            return cls(indices=indices, cdf=cdf, log_probabilities=log_probs, weighted=True)

        # Fallback to uniform sampling when |W| vanishes everywhere.
        volume = flat.size
        probabilities = np.full(volume, 1.0 / volume, dtype=float)
        cdf = np.cumsum(probabilities, dtype=float)
        cdf[-1] = 1.0
        log_probs = np.full(volume, -np.log(volume), dtype=float)
        indices = np.arange(volume, dtype=np.int64)
        return cls(indices=indices, cdf=cdf, log_probabilities=log_probs, weighted=False)

    def sample(self, rng: np.random.Generator) -> Tuple[int, float]:
        """Draw a momentum difference index and return (index, log_prob)."""

        draw = float(rng.random())
        idx = int(np.searchsorted(self.cdf, draw, side="right"))
        if idx >= self.indices.size:
            idx = self.indices.size - 1
        flat_index = int(self.indices[idx])
        log_p = float(self.log_probabilities[flat_index])
        return flat_index, log_p

    def log_probability(self, flat_index: int) -> float:
        """Return log probability of the specified flattened q index."""

        value = float(self.log_probabilities[int(flat_index)])
        return value


def build_momentum_tables(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    *,
    spins: Optional[Iterable[Spin]] = None,
) -> Dict[Spin, Tuple[MomentumProposalTable, ...]]:
    """Precompute |W|-weighted proposal tables for every slice and spin."""

    if params.time_slices != auxiliary.time_slices:
        raise ValueError("Parameter time_slices does not match auxiliary field.")

    if spins is None:
        spins = SPINS

    tables: Dict[Spin, Tuple[MomentumProposalTable, ...]] = {}
    for spin in spins:
        per_spin: List[MomentumProposalTable] = []
        for l in range(params.time_slices):
            magnitude = auxiliary.magnitude(l, spin)
            table = MomentumProposalTable.from_magnitude(magnitude)
            per_spin.append(table)
        tables[spin] = tuple(per_spin)
    return tables


@dataclass
class MonteCarloState:
    """Holds mutable Monte Carlo state for incremental updates."""

    configuration: WorldlineConfiguration
    phase: complex
    log_weight: float
    rng: np.random.Generator
    occupancy_masks: Dict[Spin, np.ndarray]
    momentum_tables: Optional[Dict[Spin, Tuple[MomentumProposalTable, ...]]] = None


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
        "auxiliary_attempts": 0,
        "auxiliary_accepts": 0,
    }

    config = mc_state.configuration
    rng = mc_state.rng
    spins = tuple(sorted(config.worldlines.keys()))
    time_slices = params.time_slices
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
        if _attempt_momentum_update(
            params,
            auxiliary,
            mc_state,
            spin,
            wl,
            config.permutations[spin],
            l,
            n,
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

    for _ in range(schedule.auxiliary_moves):
        diagnostics["auxiliary_attempts"] += 1
        slice_index = int(rng.integers(time_slices))
        site_index = int(rng.integers(params.volume))
        if _attempt_auxiliary_flip(
            params,
            auxiliary,
            mc_state,
            slice_index,
            site_index,
        ):
            diagnostics["auxiliary_accepts"] += 1
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
) -> bool:
    rng = mc_state.rng
    volume = params.volume
    lattice_size = params.lattice_size

    old_k = int(wl.trajectories[l, n])
    occupancy_mask = mc_state.occupancy_masks[spin]

    forward_ts, forward_target = _forward_link_info(wl, perm, l, n, params.time_slices)
    backward_ts, backward_source = _backward_link_info(
        wl, perm, l, n, params.time_slices
    )

    proposal_log_forward_new = 0.0
    proposal_log_forward_old = 0.0
    new_k = old_k

    tables = mc_state.momentum_tables
    table: Optional[MomentumProposalTable] = None
    if tables is not None and spin in tables:
        table = tables[spin][forward_ts]
        old_q_index = _momentum_difference_index(
            old_k, forward_target, lattice_size
        )
        proposal_log_forward_old = table.log_probability(old_q_index)
        if not np.isfinite(proposal_log_forward_old):
            table = None
            proposal_log_forward_old = 0.0
        else:
            new_q_index, proposal_log_forward_new = table.sample(rng)
            new_k = _apply_momentum_difference(
                forward_target, new_q_index, lattice_size
            )
    if table is None:
        new_k = int(rng.integers(volume))
        proposal_log_forward_new = 0.0
        proposal_log_forward_old = 0.0

    if new_k == old_k:
        return False
    if new_k < 0 or new_k >= volume:
        return False
    if occupancy_mask[l, new_k]:
        return False

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

    log_weight_ratio = (new_forward_log - old_forward_log) + (
        new_backward_log - old_backward_log
    )
    total_log_ratio = (
        log_weight_ratio + proposal_log_forward_old - proposal_log_forward_new
    )
    if total_log_ratio < 0.0 and np.log(rng.random()) >= total_log_ratio:
        return False

    delta_phi = (new_forward_phase - old_forward_phase) + (
        new_backward_phase - old_backward_phase
    )

    occupancy_mask[l, old_k] = False
    occupancy_mask[l, new_k] = True
    wl.update_momentum(l, n, new_k, enforce_pauli=False)

    mc_state.phase *= np.exp(1j * delta_phi)
    mc_state.log_weight += log_weight_ratio
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


def _attempt_auxiliary_flip(
    params: SimulationParameters,
    auxiliary: AuxiliaryField,
    mc_state: MonteCarloState,
    slice_index: int,
    site_index: int,
) -> bool:
    coupling = auxiliary.auxiliary_coupling
    old_value = auxiliary.site_value(slice_index, site_index)
    new_value = -old_value
    if new_value == old_value:
        return False

    phase_vector = np.asarray(auxiliary.site_phase_vector(site_index))
    if auxiliary.fft_mode == "real":
        phase_vector = phase_vector.real

    exp_up_old = np.exp(coupling * old_value)
    exp_up_new = np.exp(coupling * new_value)
    exp_down_old = np.exp(-coupling * old_value)
    exp_down_new = np.exp(-coupling * new_value)
    delta_up = exp_up_new - exp_up_old
    delta_down = exp_down_new - exp_down_old

    if np.isclose(delta_up, 0.0) and np.isclose(delta_down, 0.0):
        return False

    lattice_size = params.lattice_size
    time_slices = params.time_slices
    log_ratio = 0.0
    phase_delta = 0.0

    for spin in mc_state.configuration.worldlines:
        wl = mc_state.configuration.worldlines[spin]
        perm = mc_state.configuration.permutations[spin]
        w_array = auxiliary.w(slice_index, spin)
        delta_factor = delta_up if spin == "up" else delta_down
        if np.isclose(delta_factor, 0.0):
            continue
        for n in range(wl.particles):
            k_from = int(wl.trajectories[slice_index, n])
            if slice_index < time_slices - 1:
                k_to = int(wl.trajectories[slice_index + 1, n])
            else:
                target = int(perm.values[n])
                k_to = int(wl.trajectories[0, target])
            q_flat = _momentum_difference_index(k_from, k_to, lattice_size)
            old_w = w_array.flat[q_flat]
            magnitude_old = abs(old_w)
            if magnitude_old <= 0.0:
                return False
            delta = delta_factor * phase_vector[q_flat]
            new_w = old_w + delta
            magnitude_new = abs(new_w)
            if magnitude_new <= 0.0:
                return False
            log_ratio += float(np.log(magnitude_new) - np.log(magnitude_old))
            phase_delta += float(np.angle(new_w) - np.angle(old_w))

    rng = mc_state.rng
    if log_ratio < 0.0 and np.log(rng.random()) >= log_ratio:
        return False

    auxiliary.apply_site_update(
        slice_index,
        site_index,
        new_value,
        phase_vector,
        float(delta_up),
        float(delta_down),
    )

    if mc_state.momentum_tables is not None:
        for spin in list(mc_state.momentum_tables.keys()):
            current_tables = list(mc_state.momentum_tables[spin])
            magnitude = auxiliary.magnitude(slice_index, spin)
            current_tables[slice_index] = MomentumProposalTable.from_magnitude(magnitude)
            mc_state.momentum_tables[spin] = tuple(current_tables)

    mc_state.log_weight += log_ratio
    mc_state.phase *= np.exp(1j * phase_delta)
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


def _momentum_difference_index(
    k_from: int, k_to: int, lattice_size: int
) -> int:
    from_x, from_y = unflatten_momentum_index(k_from, lattice_size)
    to_x, to_y = unflatten_momentum_index(k_to, lattice_size)
    diff_x = (from_x - to_x) % lattice_size
    diff_y = (from_y - to_y) % lattice_size
    return diff_x * lattice_size + diff_y


def _apply_momentum_difference(
    target: int, diff_index: int, lattice_size: int
) -> int:
    target_x, target_y = unflatten_momentum_index(target, lattice_size)
    diff_x, diff_y = divmod(int(diff_index), lattice_size)
    new_x = (target_x + diff_x) % lattice_size
    new_y = (target_y + diff_y) % lattice_size
    return flatten_momentum_index(new_x, new_y, lattice_size)
