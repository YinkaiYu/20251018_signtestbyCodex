import numpy as np

from worldline_qmc import config, updates, worldline


class DummyRNG:
    def __init__(self, integers_seq, random_seq):
        self._integers = list(integers_seq)
        self._randoms = list(random_seq)
        self._int_index = 0
        self._rand_index = 0

    def integers(self, *args, **kwargs):
        if len(args) == 1:
            high = args[0]
            low = 0
        elif len(args) == 2:
            low, high = args
        else:
            raise TypeError("DummyRNG.integers expects 1 or 2 positional args")
        if high <= low:
            raise ValueError("high must be greater than low")
        if self._int_index >= len(self._integers):
            value = 0
        else:
            value = self._integers[self._int_index]
        self._int_index += 1
        return int(low + (value % (high - low)))

    def random(self, *args, **kwargs):
        if self._rand_index >= len(self._randoms):
            return 1.0
        value = self._randoms[self._rand_index]
        self._rand_index += 1
        return float(value)


def _base_configuration(lattice_size: int = 2, time_slices: int = 3):
    params = config.load_parameters(
        {
            "lattice_size": lattice_size,
            "beta": float(time_slices),
            "delta_tau": 1.0,
            "hopping": 1.0,
            "interaction": 0.0,
        }
    )

    num_particles = params.volume // 2
    traj = np.tile(np.arange(num_particles, dtype=np.int64), (params.time_slices, 1))
    perm = worldline.PermutationState.identity(traj.shape[1])
    cfg = worldline.WorldlineConfiguration(
        worldlines={"up": worldline.Worldline(traj.copy())},
        permutations={"up": perm},
    )
    return params, cfg


def _build_occupancy_masks(params, cfg):
    masks = {}
    volume = params.volume
    for spin, wl in cfg.worldlines.items():
        mask = np.zeros((params.time_slices, volume), dtype=bool)
        for l in range(params.time_slices):
            mask[l, wl.trajectories[l]] = True
        masks[spin] = mask
    return masks


def test_momentum_update_accepts_and_updates(monkeypatch) -> None:
    params, cfg = _base_configuration(time_slices=3)
    wl = cfg.worldlines["up"]
    wl.trajectories[:, 0] = np.array([0, 0, 0])

    amplitude_table = {
        ("up", 1, 0, 0): 1.0,
        ("up", 1, 2, 0): 0.5 * np.exp(0.3j),
        ("up", 0, 0, 0): 1.0,
        ("up", 0, 0, 2): 0.6 * np.exp(0.2j),
    }

    def log_components_stub(_params, _aux, spin, time_slice, k_from, k_to):
        amplitude = amplitude_table.get((spin, time_slice, k_from, k_to), 1.0)
        magnitude = abs(amplitude)
        if magnitude == 0.0:
            return float("-inf"), 0.0
        return float(np.log(magnitude)), float(np.angle(amplitude))

    monkeypatch.setattr(updates, "transition_log_components", log_components_stub)

    rng = DummyRNG(integers_seq=[0, 1, 0, 2], random_seq=[0.1])
    state = updates.MonteCarloState(
        configuration=cfg,
        phase=1.0 + 0.0j,
        log_weight=0.0,
        rng=rng,
        occupancy_masks=_build_occupancy_masks(params, cfg),
    )

    diagnostics, samples = updates.metropolis_sweep(
        params,
        auxiliary=None,
        mc_state=state,
        schedule=updates.UpdateSchedule(worldline_moves=1, permutation_moves=0),
        measurement_interval=1,
    )

    assert wl.trajectories[1, 0] == 2
    expected_phase = np.exp(1j * 0.5)
    assert np.isclose(state.phase, expected_phase)
    assert np.isclose(state.log_weight, np.log(0.5) + np.log(0.6))
    assert diagnostics["momentum_accepts"] == 1
    assert diagnostics["momentum_attempts"] == 1
    assert len(samples) == 1


def test_permutation_swap_updates_phase(monkeypatch) -> None:
    params, cfg = _base_configuration(time_slices=2)
    wl = cfg.worldlines["up"]
    wl.trajectories[:, 0] = np.array([0, 0])
    wl.trajectories[:, 1] = np.array([1, 1])

    amplitude_table = {
        ("up", params.time_slices - 1, 0, 0): 1.0,
        ("up", params.time_slices - 1, 1, 1): 1.0,
        ("up", params.time_slices - 1, 0, 1): 0.8 * np.exp(0.1j),
        ("up", params.time_slices - 1, 1, 0): 0.9 * np.exp(0.2j),
    }

    def log_components_stub(_params, _aux, spin, time_slice, k_from, k_to):
        amplitude = amplitude_table.get((spin, time_slice, k_from, k_to), 1.0)
        magnitude = abs(amplitude)
        if magnitude == 0.0:
            return float("-inf"), 0.0
        return float(np.log(magnitude)), float(np.angle(amplitude))

    monkeypatch.setattr(updates, "transition_log_components", log_components_stub)

    rng = DummyRNG(integers_seq=[0, 0, 1], random_seq=[0.1, 0.1])
    state = updates.MonteCarloState(
        configuration=cfg,
        phase=1.0 + 0.0j,
        log_weight=0.0,
        rng=rng,
        occupancy_masks=_build_occupancy_masks(params, cfg),
    )

    diagnostics, samples = updates.metropolis_sweep(
        params,
        auxiliary=None,
        mc_state=state,
        schedule=updates.UpdateSchedule(worldline_moves=0, permutation_moves=1),
        measurement_interval=1,
    )

    assert np.array_equal(cfg.permutations["up"].values, np.array([1, 0]))
    expected_phase = -np.exp(1j * 0.3)
    assert np.isclose(state.phase, expected_phase)
    assert np.isclose(state.log_weight, np.log(0.8) + np.log(0.9))
    assert diagnostics["permutation_accepts"] == 1
    assert diagnostics["permutation_attempts"] == 1
    assert len(samples) == 1


def test_momentum_update_rejects_pauli(monkeypatch) -> None:
    params, cfg = _base_configuration(time_slices=2)
    wl = cfg.worldlines["up"]
    wl.trajectories[:, 0] = np.array([0, 0])
    wl.trajectories[:, 1] = np.array([1, 1])

    monkeypatch.setattr(
        updates,
        "transition_log_components",
        lambda *args, **kwargs: (0.0, 0.0),
    )

    rng = DummyRNG(integers_seq=[0, 0, 0, 1], random_seq=[0.0])
    state = updates.MonteCarloState(
        configuration=cfg,
        phase=1.0 + 0.0j,
        log_weight=0.0,
        rng=rng,
        occupancy_masks=_build_occupancy_masks(params, cfg),
    )

    diagnostics, samples = updates.metropolis_sweep(
        params,
        auxiliary=None,
        mc_state=state,
        schedule=updates.UpdateSchedule(worldline_moves=1, permutation_moves=0),
        measurement_interval=1,
    )

    assert wl.trajectories[0, 0] == 0
    assert diagnostics["momentum_accepts"] == 0
    assert diagnostics["momentum_attempts"] == 1
    assert np.isclose(state.phase, 1.0 + 0.0j)
    assert np.isclose(state.log_weight, 0.0)
    assert len(samples) == 1
