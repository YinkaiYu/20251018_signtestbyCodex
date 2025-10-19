import json
from pathlib import Path

import experiments.plot_sign_vs_U as plot_u
import experiments.plot_sign_vs_beta_L as plot_beta_l
import experiments.run_average_sign as run_avg


def test_run_average_sign_script(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    args = [
        "--output-dir",
        str(output_dir),
        "--sweeps",
        "1",
        "--thermalization",
        "0",
        "--u-values",
        "0",
        "--beta-values",
        "8",
        "--l-values",
        "8",
        "--seed",
        "5",
    ]
    monkeypatch.setenv("MPLBACKEND", "Agg")
    exit_code = run_avg.main(args)
    assert exit_code == 0

    u_json = json.loads((output_dir / "average_sign_vs_U.json").read_text(encoding="utf-8"))
    assert len(u_json) == 1
    beta_l_json = json.loads((output_dir / "average_sign_vs_beta_L.json").read_text(encoding="utf-8"))
    assert len(beta_l_json) == 1
    assert any((output_dir / "logs_u").glob("*.jsonl"))
    assert any((output_dir / "logs_beta_l").glob("*.jsonl"))

    # Plotting scripts should create figures from the generated JSON data.
    plot_u_path = output_dir / "plot_u.png"
    plot_beta_path = output_dir / "plot_beta.png"
    plot_u.main([
        "--data",
        str(output_dir / "average_sign_vs_U.json"),
        "--output",
        str(plot_u_path),
    ])
    plot_beta_l.main([
        "--data",
        str(output_dir / "average_sign_vs_beta_L.json"),
        "--output",
        str(plot_beta_path),
    ])
    assert plot_u_path.exists()
    assert plot_beta_path.exists()
