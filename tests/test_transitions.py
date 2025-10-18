import math

import numpy as np

from worldline_qmc import auxiliary, config, lattice, transitions


def _make_params() -> config.SimulationParameters:
    payload = {
        "lattice_size": 2,
        "beta": 1.0,
        "delta_tau": 0.25,
        "hopping": 1.0,
        "interaction": 0.0,
    }
    return config.load_parameters(payload)


def test_transition_amplitude_diagonal_matches_free_dispersion() -> None:
    params = _make_params()
    aux = auxiliary.generate_auxiliary_field(params, seed=42)
    grid = lattice.momentum_grid(params.lattice_size)
    energies = lattice.dispersion(grid, params.hopping)

    k = 1
    amplitude = transitions.transition_amplitude(
        params, aux, spin="up", time_slice=0, k_from=k, k_to=k
    )

    expected = math.exp(-params.delta_tau * energies[k])
    assert np.isclose(amplitude, expected)


def test_transition_amplitude_off_diagonal_zero_for_u_zero() -> None:
    params = _make_params()
    aux = auxiliary.generate_auxiliary_field(params, seed=17)

    amp = transitions.transition_amplitude(
        params, aux, spin="down", time_slice=0, k_from=0, k_to=1
    )
    assert np.isclose(amp, 0.0)
