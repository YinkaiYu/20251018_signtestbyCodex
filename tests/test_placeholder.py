from worldline_qmc import config


def test_simulation_parameters_time_slices_roundtrip() -> None:
    params = config.SimulationParameters(
        lattice_size=4,
        beta=1.0,
        delta_tau=0.1,
        hopping=1.0,
        interaction=0.0,
    )
    assert params.time_slices == 10

