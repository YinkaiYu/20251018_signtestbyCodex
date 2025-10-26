import numpy as np
import pytest

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


@pytest.mark.parametrize("fft_mode", ["complex", "real"])
def test_apply_site_update_matches_fft(fft_mode: str) -> None:
    params = _make_params(lattice_size=2, beta=1.0, delta_tau=0.5, interaction=4.0, fft_mode=fft_mode)
    aux_field = auxiliary.generate_auxiliary_field(params, seed=3)

    slice_index = 0
    site_flat = 0
    old_value = aux_field.site_value(slice_index, site_flat)
    new_value = -old_value
    phase_vector = aux_field.site_phase_vector(site_flat)
    if fft_mode == "real":
        phase_vector = phase_vector.real

    coupling = aux_field.auxiliary_coupling
    delta_up = float(np.exp(coupling * new_value) - np.exp(coupling * old_value))
    delta_down = float(np.exp(-coupling * new_value) - np.exp(-coupling * old_value))

    aux_field.apply_site_update(
        slice_index,
        site_flat,
        new_value,
        phase_vector,
        delta_up,
        delta_down,
    )

    spatial = aux_field.slices[slice_index].spatial_field.astype(float)
    exp_up = np.exp(coupling * spatial)
    exp_down = np.exp(-coupling * spatial)
    w_up = np.fft.fftn(exp_up).conj()
    w_down = np.fft.fftn(exp_down).conj()
    if fft_mode == "real":
        w_up = np.real(w_up)
        w_down = np.real(w_down)

    assert np.allclose(aux_field.w(slice_index, "up"), w_up)
    assert np.allclose(aux_field.w(slice_index, "down"), w_down)
