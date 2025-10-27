import numpy as np

from worldline_qmc import auxiliary, config, lattice, simulation, updates, worldline


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


class ArrayRNG:
    def __init__(self, arrays, *, scalar=0.0):
        self._arrays = [np.array(arr, dtype=np.float64) for arr in arrays]
        self._scalar = float(scalar)

    def random(self, size=None):
        if size is None:
            if self._arrays:
                value = self._arrays.pop(0)
                if value.shape == ():
                    return float(value)
                if value.size == 1:
                    return float(value.reshape(-1)[0])
                raise ValueError("Scalar draw requested but preset array has more than one element.")
            return self._scalar
        if not self._arrays:
            return np.full(size, self._scalar, dtype=np.float64)
        value = self._arrays.pop(0)
        if tuple(size) != value.shape:
            raise ValueError("Requested shape does not match preset array.")
        return value.copy()

    def integers(self, *args, **kwargs):
        return 0


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
        half_step_factor=np.ones(params.volume, dtype=np.float64),
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
        half_step_factor=np.ones(params.volume, dtype=np.float64),
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
        half_step_factor=np.ones(params.volume, dtype=np.float64),
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


def test_momentum_update_weighted_proposal(monkeypatch) -> None:
    params, cfg = _base_configuration(time_slices=2)
    wl = cfg.worldlines["up"]
    wl.trajectories[:, 0] = np.array([0, 1])
    wl.trajectories[:, 1] = np.array([2, 3])

    amplitude_table = {
        ("up", 0, 0, 1): 2.0,
        ("up", 0, 3, 1): 4.0,
        ("up", 1, 1, 0): 3.0,
        ("up", 1, 1, 3): 6.0,
    }

    def log_components_stub(_params, _aux, spin, time_slice, k_from, k_to):
        amplitude = amplitude_table.get((spin, time_slice, k_from, k_to), 1.0)
        magnitude = abs(amplitude)
        if magnitude == 0.0:
            return float("-inf"), 0.0
        return float(np.log(magnitude)), float(np.angle(amplitude))

    monkeypatch.setattr(updates, "transition_log_components", log_components_stub)

    rng = DummyRNG(integers_seq=[0, 0, 0], random_seq=[0.9])
    magnitude = np.array([[0.0, 1.0], [2.0, 0.0]])
    table = updates.MomentumProposalTable.from_magnitude(magnitude)
    momentum_tables = {"up": [table, table]}

    state = updates.MonteCarloState(
        configuration=cfg,
        phase=1.0 + 0.0j,
        log_weight=0.0,
        rng=rng,
        occupancy_masks=_build_occupancy_masks(params, cfg),
        half_step_factor=np.ones(params.volume, dtype=np.float64),
        momentum_tables=momentum_tables,
    )

    diagnostics, _ = updates.metropolis_sweep(
        params,
        auxiliary=None,
        mc_state=state,
        schedule=updates.UpdateSchedule(worldline_moves=1, permutation_moves=0),
        measurement_interval=1,
    )

    assert wl.trajectories[0, 0] == 3
    assert np.isclose(state.log_weight, np.log(4.0))
    assert np.isclose(state.phase, 1.0 + 0.0j)
    assert diagnostics["momentum_accepts"] == 1


def test_magnetization_balanced_spins_yields_zero() -> None:
    params = config.load_parameters(
        {
            "lattice_size": 2,
            "beta": 2.0,
            "delta_tau": 1.0,
            "hopping": 1.0,
            "interaction": 4.0,
        }
    )
    time_slices = params.time_slices
    particles = params.volume // 2
    base_slice = np.arange(particles, dtype=np.int64)
    trajectories = np.tile(base_slice, (time_slices, 1))

    worldlines_dict = {
        "up": worldline.Worldline(trajectories.copy()),
        "down": worldline.Worldline(trajectories.copy()),
    }
    permutations = {
        spin: worldline.PermutationState.identity(particles) for spin in worldlines_dict
    }
    cfg = worldline.WorldlineConfiguration(worldlines=worldlines_dict, permutations=permutations)
    occupancy_masks = _build_occupancy_masks(params, cfg)

    grid = lattice.momentum_grid(params.lattice_size)
    energies = lattice.dispersion(grid, params.hopping)
    half_step_factor = np.exp(-0.5 * params.delta_tau * energies).ravel()

    state = updates.MonteCarloState(
        configuration=cfg,
        phase=1.0 + 0.0j,
        log_weight=0.0,
        rng=np.random.default_rng(0),
        occupancy_masks=occupancy_masks,
        half_step_factor=half_step_factor,
    )

    magnetization = updates._compute_local_magnetization(params, state)
    assert np.allclose(magnetization, 0.0, atol=1e-12)


def test_gibbs_update_resamples_auxiliary(monkeypatch) -> None:
    params = config.load_parameters(
        {
            "lattice_size": 2,
            "beta": 2.0,
            "delta_tau": 1.0,
            "hopping": 1.0,
            "interaction": 4.0,
        }
    )
    aux_field = auxiliary.generate_auxiliary_field(params, seed=1)
    lattice_shape = (params.lattice_size, params.lattice_size)
    for slice_index in range(params.time_slices):
        aux_field.refresh_slice(slice_index, -np.ones(lattice_shape, dtype=np.int8))

    particles = params.volume // 2
    trajectories = np.tile(np.arange(particles, dtype=np.int64), (params.time_slices, 1))
    worldlines_dict = {"up": worldline.Worldline(trajectories)}
    permutations = {"up": worldline.PermutationState.identity(particles)}
    cfg = worldline.WorldlineConfiguration(worldlines=worldlines_dict, permutations=permutations)
    occupancy_masks = _build_occupancy_masks(params, cfg)

    grid = lattice.momentum_grid(params.lattice_size)
    energies = lattice.dispersion(grid, params.hopping)
    half_step_factor = np.exp(-0.5 * params.delta_tau * energies).ravel()

    draws = [
        np.array([[0.1, 0.6], [0.4, 0.8]], dtype=np.float64),
        np.array([[0.2, 0.9], [0.3, 0.7]], dtype=np.float64),
    ]
    rng = ArrayRNG(draws)

    state = updates.MonteCarloState(
        configuration=cfg,
        phase=1.0 + 0.0j,
        log_weight=0.0,
        rng=rng,
        occupancy_masks=occupancy_masks,
        half_step_factor=half_step_factor,
    )

    const_magnetization = np.full(
        (params.time_slices, params.lattice_size, params.lattice_size),
        0.5,
        dtype=np.float64,
    )
    monkeypatch.setattr(
        updates,
        "_compute_local_magnetization",
        lambda *_args, **_kwargs: const_magnetization,
    )

    def fake_refresh(_params, auxiliary_field, _cfg, slice_index, old_field, new_field):
        auxiliary_field.refresh_slice(slice_index, new_field)
        return 0.0, 0.0, True

    monkeypatch.setattr(updates, "_apply_auxiliary_slice_refresh", fake_refresh)

    diagnostics = updates._gibbs_update_auxiliary(params, aux_field, state)

    coupling = params.auxiliary_coupling
    prob = 0.5 * (1.0 + np.tanh(coupling * 0.5))
    expected_fields = [
        np.where(draws[0] < prob, 1, -1).astype(np.int8),
        np.where(draws[1] < prob, 1, -1).astype(np.int8),
    ]

    total_flips = 0
    changed_slices = 0
    for slice_index, expected_field in enumerate(expected_fields):
        actual = aux_field.slices[slice_index].spatial_field
        assert np.array_equal(actual, expected_field)
        flips = int(np.count_nonzero(expected_field != -1))
        total_flips += flips
        if flips > 0:
            changed_slices += 1

    assert diagnostics["auxiliary_site_flips"] == float(total_flips)
    assert diagnostics["auxiliary_slice_changes"] == float(changed_slices)
    assert diagnostics["auxiliary_log_delta"] == 0.0
    assert diagnostics["auxiliary_phase_delta"] == 0.0
    assert np.isclose(state.phase, 1.0 + 0.0j)
    assert np.isclose(state.log_weight, 0.0)


def test_gibbs_update_zero_magnetization_half_probability(monkeypatch) -> None:
    params = config.load_parameters(
        {
            "lattice_size": 2,
            "beta": 2.0,
            "delta_tau": 1.0,
            "hopping": 1.0,
            "interaction": 4.0,
        }
    )
    aux_field = auxiliary.generate_auxiliary_field(params, seed=2)
    lattice_shape = (params.lattice_size, params.lattice_size)
    for slice_index in range(params.time_slices):
        aux_field.refresh_slice(slice_index, -np.ones(lattice_shape, dtype=np.int8))

    particles = params.volume // 2
    trajectories = np.tile(np.arange(particles, dtype=np.int64), (params.time_slices, 1))
    worldlines_dict = {"up": worldline.Worldline(trajectories)}
    permutations = {"up": worldline.PermutationState.identity(particles)}
    cfg = worldline.WorldlineConfiguration(worldlines=worldlines_dict, permutations=permutations)
    occupancy_masks = _build_occupancy_masks(params, cfg)

    grid = lattice.momentum_grid(params.lattice_size)
    energies = lattice.dispersion(grid, params.hopping)
    half_step_factor = np.exp(-0.5 * params.delta_tau * energies).ravel()

    draws = [
        np.array([[0.1, 0.7], [0.4, 0.9]], dtype=np.float64),
        np.array([[0.3, 0.2], [0.8, 0.6]], dtype=np.float64),
    ]
    rng = ArrayRNG([d.copy() for d in draws])

    state = updates.MonteCarloState(
        configuration=cfg,
        phase=1.0 + 0.0j,
        log_weight=0.0,
        rng=rng,
        occupancy_masks=occupancy_masks,
        half_step_factor=half_step_factor,
    )

    zero_magnetization = np.zeros(
        (params.time_slices, params.lattice_size, params.lattice_size), dtype=np.float64
    )
    monkeypatch.setattr(
        updates,
        "_compute_local_magnetization",
        lambda *_args, **_kwargs: zero_magnetization,
    )

    diagnostics = updates._gibbs_update_auxiliary(params, aux_field, state)

    expected_fields = [
        np.where(draw < 0.5, 1, -1).astype(np.int8) for draw in draws
    ]
    total_flips = 0
    for slice_index, expected in enumerate(expected_fields):
        actual = aux_field.slices[slice_index].spatial_field
        assert np.array_equal(actual, expected)
        total_flips += int(np.count_nonzero(expected == 1))
    assert diagnostics["auxiliary_site_flips"] == float(total_flips)


def test_gibbs_update_real_fft_zero_phase(monkeypatch) -> None:
    params = config.load_parameters(
        {
            "lattice_size": 2,
            "beta": 2.0,
            "delta_tau": 1.0,
            "hopping": 1.0,
            "interaction": 4.0,
            "fft_mode": "real",
            "auxiliary_mode": "uniform_plus",
        }
    )
    aux_field = auxiliary.generate_auxiliary_field(params, seed=4)
    configuration = simulation._initialize_configuration(params, np.random.default_rng(0))
    mc_state = simulation._initialize_mc_state(
        params,
        aux_field,
        configuration,
        np.random.default_rng(1),
    )
    initial_phase = mc_state.phase

    magnetization = np.full(
        (params.time_slices, params.lattice_size, params.lattice_size),
        0.2,
        dtype=np.float64,
    )
    monkeypatch.setattr(
        updates,
        "_compute_local_magnetization",
        lambda *_args, **_kwargs: magnetization,
    )

    draws = [
        np.array([[0.1, 0.9], [0.3, 0.8]], dtype=np.float64),
        np.array([[0.4, 0.2], [0.7, 0.6]], dtype=np.float64),
    ]
    mc_state.rng = ArrayRNG([d.copy() for d in draws])

    diagnostics = updates._gibbs_update_auxiliary(params, aux_field, mc_state)

    assert diagnostics["auxiliary_phase_delta"] == 0.0
    assert np.isclose(mc_state.phase, initial_phase)


def test_metropolis_sweep_weight_matches_recompute(monkeypatch) -> None:
    params = config.load_parameters(
        {
            "lattice_size": 2,
            "beta": 2.0,
            "delta_tau": 1.0,
            "hopping": 1.0,
            "interaction": 4.0,
            "auxiliary_mode": "uniform_plus",
        }
    )
    aux_field = auxiliary.generate_auxiliary_field(params, seed=7)
    configuration = simulation._initialize_configuration(params, np.random.default_rng(0))
    mc_state = simulation._initialize_mc_state(
        params,
        aux_field,
        configuration,
        np.random.default_rng(1),
    )
    schedule = updates.UpdateSchedule(worldline_moves=0, permutation_moves=0)

    magnetization = np.full(
        (params.time_slices, params.lattice_size, params.lattice_size),
        0.35,
        dtype=np.float64,
    )
    monkeypatch.setattr(
        updates,
        "_compute_local_magnetization",
        lambda *_args, **_kwargs: magnetization,
    )

    draws = [
        np.array([[0.1, 0.4], [0.7, 0.8]], dtype=np.float64),
        np.array([[0.2, 0.9], [0.5, 0.3]], dtype=np.float64),
    ]
    mc_state.rng = ArrayRNG([d.copy() for d in draws])

    diagnostics, phase_samples = updates.metropolis_sweep(
        params,
        aux_field,
        mc_state,
        schedule,
        measurement_interval=1,
    )

    assert diagnostics["auxiliary_slice_changes"] > 0.0
    assert phase_samples[-1] == mc_state.phase

    recomputed_log, recomputed_phase = simulation._compute_weight_and_phase(
        params,
        aux_field,
        configuration,
    )
    assert np.isclose(mc_state.log_weight, recomputed_log)
    assert np.isclose(mc_state.phase, recomputed_phase)
