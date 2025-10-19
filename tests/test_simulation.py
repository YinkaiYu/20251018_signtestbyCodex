import json
import numpy as np
from dataclasses import replace
from pathlib import Path

from worldline_qmc import auxiliary, config, simulation


def make_params(**overrides):
    base = {
        "lattice_size": 2,
        "beta": 1.0,
        "delta_tau": 0.5,
        "hopping": 1.0,
        "interaction": 0.0,
        "sweeps": 0,
        "thermalization_sweeps": 0,
    }
    base.update(overrides)
    return config.load_parameters(base)


def test_run_simulation_zero_sweeps() -> None:
    params = make_params()
    aux_field = auxiliary.generate_auxiliary_field(params)
    result = simulation.run_simulation(params, aux_field)

    assert result.samples == 0
    assert result.diagnostics["total_sweeps"] == 0.0
    assert set(result.measurements.keys()) == {"re", "im", "abs"}
    assert set(result.variances.keys()) == {"re", "im", "abs"}


def test_run_simulation_collects_samples(tmp_path: Path) -> None:
    params = make_params(sweeps=2, thermalization_sweeps=1, seed=123)
    params = replace(params, log_path=tmp_path / "log.jsonl")
    aux_field = auxiliary.generate_auxiliary_field(params)
    result = simulation.run_simulation(params, aux_field)

    assert result.samples == 2
    assert result.diagnostics["total_sweeps"] == 3.0
    assert result.diagnostics["measurement_sweeps"] == 2.0

    log_path = tmp_path / "log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    entries = [json.loads(line) for line in lines]
    assert entries[-1]["is_measurement"] is True


def test_initialize_configuration_fermi_sea() -> None:
    params = make_params(lattice_size=4, beta=0.5, delta_tau=0.25, interaction=0.0)
    rng = np.random.default_rng(0)
    configuration = simulation._initialize_configuration(params, rng)

    worldline = configuration.worldlines["up"]
    first_slice = worldline.trajectories[0]
    assert np.all(worldline.trajectories == first_slice)

    grid = simulation.lattice.momentum_grid(params.lattice_size)
    energies = simulation.lattice.dispersion(grid, params.hopping)
    ordering = np.argsort(energies)
    expected = ordering[: worldline.particles]
    assert np.array_equal(np.sort(first_slice), np.sort(expected))
