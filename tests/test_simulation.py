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


def test_run_simulation_collects_samples() -> None:
    params = make_params(sweeps=2, thermalization_sweeps=1, seed=123)
    aux_field = auxiliary.generate_auxiliary_field(params)
    result = simulation.run_simulation(params, aux_field)

    assert result.samples == 2
    assert result.diagnostics["total_sweeps"] == 3.0
    assert result.diagnostics["measurement_sweeps"] == 2.0
