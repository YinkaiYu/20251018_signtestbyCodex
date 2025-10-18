import json
from pathlib import Path

from worldline_qmc import cli


def test_cli_runs_and_writes_output(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "result.json"
    payload = {
        "lattice_size": 2,
        "beta": 1.0,
        "delta_tau": 0.5,
        "hopping": 1.0,
        "interaction": 0.0,
        "sweeps": 1,
        "thermalization_sweeps": 0,
        "seed": 42,
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "measurements" in data
    assert "diagnostics" in data
    assert data["samples"] == 1
