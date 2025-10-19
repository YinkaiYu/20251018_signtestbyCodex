import json
import math
from pathlib import Path

import numpy as np
import pytest

from worldline_qmc import config


def test_load_parameters_from_mapping() -> None:
    mapping = {
        "lattice_size": 4,
        "beta": 2.0,
        "delta_tau": 0.5,
        "hopping": 1.0,
        "interaction": 4.0,
        "sweeps": 10,
        "thermalization_sweeps": 2,
        "worldline_moves_per_slice": 5,
        "permutation_moves_per_slice": 3,
        "output_path": "outputs/run.json",
        "log_path": "outputs/diag.jsonl",
        "fft_mode": "real",
        "initial_state": "random",
        "custom_note": "test",
    }

    params = config.load_parameters(mapping)

    assert params.lattice_size == 4
    assert params.time_slices == 4
    assert params.volume == 16
    assert params.particles_per_spin == 8
    expected_lambda = float(np.arccosh(math.exp(mapping["delta_tau"] * mapping["interaction"] / 2)))
    assert math.isclose(params.auxiliary_coupling, expected_lambda)
    assert params.extra["custom_note"] == "test"
    assert isinstance(params.output_path, Path)
    assert isinstance(params.log_path, Path)
    assert params.fft_mode == "real"
    assert params.initial_state == "random"


def test_load_parameters_from_json(tmp_path: Path) -> None:
    payload = {
        "lattice_size": 2,
        "beta": 1.0,
        "delta_tau": 0.25,
        "hopping": 1.0,
        "interaction": 2.0,
        "seed": 11,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    params = config.load_parameters(path)

    assert params.seed == 11
    assert params.time_slices == 4


@pytest.mark.parametrize(
    "beta, delta_tau",
    [
        (1.0, 0.3),
        (2.0, 0.3333333),
    ],
)
def test_load_parameters_rejects_non_integer_time_slice(beta: float, delta_tau: float) -> None:
    mapping = {
        "lattice_size": 4,
        "beta": beta,
        "delta_tau": delta_tau,
        "hopping": 1.0,
        "interaction": 4.0,
    }

    with pytest.raises(ValueError):
        config.load_parameters(mapping)


def test_load_parameters_rejects_negative_moves() -> None:
    mapping = {
        "lattice_size": 4,
        "beta": 1.0,
        "delta_tau": 0.25,
        "hopping": 1.0,
        "interaction": 4.0,
        "worldline_moves_per_slice": -1,
    }

    with pytest.raises(ValueError):
        config.load_parameters(mapping)


def test_load_parameters_invalid_fft_mode() -> None:
    mapping = {
        "lattice_size": 4,
        "beta": 1.0,
        "delta_tau": 0.25,
        "hopping": 1.0,
        "interaction": 4.0,
        "fft_mode": "invalid",
    }

    with pytest.raises(ValueError):
        config.load_parameters(mapping)
