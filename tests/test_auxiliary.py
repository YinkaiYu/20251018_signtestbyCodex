import numpy as np

from worldline_qmc import auxiliary, config


def _make_params(**overrides: object) -> config.SimulationParameters:
    defaults = {
        "lattice_size": 2,
        "beta": 1.0,
        "delta_tau": 0.5,
        "hopping": 1.0,
        "interaction": 0.0,
        "seed": 7,
    }
    defaults.update(overrides)
    return config.load_parameters(defaults)


def test_auxiliary_field_u_zero_recovers_structure() -> None:
    params = _make_params(interaction=0.0, beta=1.0, delta_tau=0.25)
    aux_field = auxiliary.generate_auxiliary_field(params, seed=21)

    assert aux_field.time_slices == params.time_slices

    volume = params.volume
    w_up = aux_field.w(0, "up")
    w_down = aux_field.w(0, "down")

    assert np.isclose(w_up[0, 0], volume)
    assert np.isclose(w_down[0, 0], volume)
    assert np.allclose(w_up.flat[1:], 0.0, atol=1e-10)
    assert np.allclose(w_down.flat[1:], 0.0, atol=1e-10)


def test_auxiliary_magnitude_phase_consistency() -> None:
    params = _make_params(interaction=4.0, beta=1.0, delta_tau=0.25)
    aux_field = auxiliary.generate_auxiliary_field(params, seed=5)

    w = aux_field.w(0, "up")
    magnitude = aux_field.magnitude(0, "up")
    phase = aux_field.phase(0, "up")
    reconstructed = magnitude * np.exp(1j * phase)

    assert np.allclose(w, reconstructed)


def test_auxiliary_seed_reproducibility() -> None:
    params = _make_params(interaction=2.0, beta=1.0, delta_tau=0.2)

    field_a = auxiliary.generate_auxiliary_field(params, seed=123)
    field_b = auxiliary.generate_auxiliary_field(params, seed=123)

    slice_a = field_a.slices[0].spatial_field
    slice_b = field_b.slices[0].spatial_field

    assert np.array_equal(slice_a, slice_b)


def test_auxiliary_real_fft_mode_produces_real_weights() -> None:
    params = _make_params(interaction=2.0, fft_mode="real")
    aux_field = auxiliary.generate_auxiliary_field(params, seed=5)

    assert aux_field.fft_mode == "real"
    w = aux_field.w(0, "up")
    assert np.allclose(np.imag(w), 0.0, atol=1e-12)


def test_uniform_plus_auxiliary_field_is_deterministic() -> None:
    params = _make_params(interaction=4.0, auxiliary_mode="uniform_plus", beta=1.0, delta_tau=0.25)
    aux_field = auxiliary.generate_auxiliary_field(params, seed=11)

    for slice_cache in aux_field.slices:
        assert np.all(slice_cache.spatial_field == 1)

    volume = params.volume
    coupling = params.auxiliary_coupling

    w_up = aux_field.w(0, "up")
    w_down = aux_field.w(0, "down")

    assert np.isclose(w_up[0, 0], volume * np.exp(coupling))
    assert np.isclose(w_down[0, 0], volume * np.exp(-coupling))
    assert np.allclose(w_up.flat[1:], 0.0, atol=1e-12)
    assert np.allclose(w_down.flat[1:], 0.0, atol=1e-12)


def test_checkerboard_auxiliary_field_patterns() -> None:
    params = _make_params(
        lattice_size=4,
        auxiliary_mode="checkerboard",
        interaction=2.0,
        beta=1.0,
        delta_tau=0.25,
    )
    aux_field = auxiliary.generate_auxiliary_field(params, seed=0)

    expected = np.array(
        [
            [1, -1, 1, -1],
            [-1, 1, -1, 1],
            [1, -1, 1, -1],
            [-1, 1, -1, 1],
        ],
        dtype=np.int8,
    )
    for slice_cache in aux_field.slices:
        assert np.array_equal(slice_cache.spatial_field, expected)
