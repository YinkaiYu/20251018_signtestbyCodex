import json
from pathlib import Path

import experiments.plot_sign_vs_U as plot_u
import experiments.plot_sign_vs_beta_L as plot_beta_l
import experiments.run_sign_vs_U as run_u
import experiments.run_sign_vs_beta_L as run_beta_l


def test_run_average_sign_scripts(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")

    # U sweep
    output_u = tmp_path / "out_u"
    args_u = [
        "--output-dir",
        str(output_u),
        "--sweeps",
        "1",
        "--thermalization",
        "0",
        "--u-values",
        "0",
        "--lattice-size",
        "4",
        "--beta",
        "4",
        "--seed",
        "5",
    ]
    assert run_u.main(args_u) == 0
    u_json_path = output_u / "average_sign_vs_U.json"
    u_json = json.loads(u_json_path.read_text(encoding="utf-8"))
    assert len(u_json) == 1
    assert any((output_u / "logs_u").glob("*.jsonl"))

    # beta/L sweep
    output_beta = tmp_path / "out_beta"
    args_beta = [
        "--output-dir",
        str(output_beta),
        "--sweeps",
        "1",
        "--thermalization",
        "0",
        "--beta-values",
        "8",
        "--l-values",
        "8",
        "--interaction",
        "20",
        "--seed",
        "10",
    ]
    assert run_beta_l.main(args_beta) == 0
    beta_json_path = output_beta / "average_sign_vs_beta_L.json"
    beta_l_json = json.loads(beta_json_path.read_text(encoding="utf-8"))
    assert len(beta_l_json) == 1
    assert any((output_beta / "logs_beta_l").glob("*.jsonl"))

    # Plotting scripts should create figures from the generatedJSON data.
    plot_u_path = output_u / "plot_u.png"
    plot_beta_path = output_beta / "plot_beta.png"
    plot_u.main([
        "--data",
        str(u_json_path),
        "--output",
        str(plot_u_path),
    ])
    plot_beta_l.main([
        "--data",
        str(beta_json_path),
        "--output",
        str(plot_beta_path),
    ])
    assert plot_u_path.exists()
    assert plot_beta_path.exists()
