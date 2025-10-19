"""Average sign parameter studies and visualization.

This script runs the simulation for two scenarios:
1. Fixed L=12, beta=12 while scanning the interaction strength U.
2. Fixed U=20 while scanning beta and lattice size L (defaults: L∈{4,6,8,12}).

Results (JSON + plots) are stored under ``experiments/output`` by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from worldline_qmc import auxiliary, config, simulation


DEFAULT_OUTPUT_DIR = Path("experiments/output")
DEFAULT_DELTA_TAU = 1.0 / 32.0
DEFAULT_SWEEPS = 16
DEFAULT_THERMALIZATION = 4
DEFAULT_HOPPING = 1.0
DEFAULT_FFT_MODE = "complex"


@dataclass
class RunSpec:
    lattice_size: int
    beta: float
    interaction: float
    seed: int


def build_parameters(
    spec: RunSpec,
    sweeps: int,
    thermalization: int,
    *,
    fft_mode: str,
) -> config.SimulationParameters:
    payload = {
        "lattice_size": spec.lattice_size,
        "beta": spec.beta,
        "delta_tau": DEFAULT_DELTA_TAU,
        "hopping": DEFAULT_HOPPING,
        "interaction": spec.interaction,
        "sweeps": sweeps,
        "thermalization_sweeps": thermalization,
        "seed": spec.seed,
        "fft_mode": fft_mode,
    }
    return config.load_parameters(payload)


def run_experiment(
    specs: Iterable[RunSpec],
    sweeps: int,
    thermalization: int,
    log_dir: Path | None = None,
    fft_mode: str = DEFAULT_FFT_MODE,
) -> List[dict]:
    results: List[dict] = []
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    for idx, spec in enumerate(specs):
        params = build_parameters(spec, sweeps, thermalization, fft_mode=fft_mode)
        if log_dir is not None:
            log_path = log_dir / f"run_{idx:03d}.jsonl"
            params = replace(params, log_path=log_path)
        aux_field = auxiliary.generate_auxiliary_field(params)
        result = simulation.run_simulation(params, aux_field)
        results.append(
            {
                "lattice_size": spec.lattice_size,
                "beta": spec.beta,
                "interaction": spec.interaction,
                "seed": spec.seed,
                "measurements": result.measurements,
                "variances": result.variances,
                "diagnostics": result.diagnostics,
                "samples": result.samples,
            }
        )
    return results


def plot_u_sweep(data: List[dict], output_path: Path) -> None:
    interactions = [entry["interaction"] for entry in data]
    avg_sign = [entry["measurements"]["re"] for entry in data]
    plt.figure(figsize=(6, 4))
    plt.plot(interactions, avg_sign, marker="o")
    plt.xlabel("Interaction U")
    plt.ylabel("Re S")
    plt.title("Re S vs U (L=32, beta=32)")
    if avg_sign:
        y_min = min(avg_sign)
        y_max = max(avg_sign)
        if y_max == y_min:
            margin = max(0.05 * abs(y_max), 0.05)
            plt.ylim(y_min - margin, y_max + margin)
        else:
            span = y_max - y_min
            margin = span * 0.1
            plt.ylim(y_min - margin, y_max + margin)
    plt.grid(True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_beta_l_sweep(data: List[dict], output_path: Path) -> None:
    # Group by lattice size and plot |S| vs beta for each L.
    grouped: dict[int, list[tuple[float, float]]] = {}
    for entry in data:
        grouped.setdefault(entry["lattice_size"], []).append(
            (entry["beta"], entry["measurements"]["abs"])
        )
    plt.figure(figsize=(6, 4))
    for lattice_size, values in sorted(grouped.items()):
        values.sort(key=lambda pair: pair[0])
        betas, avg_sign = zip(*values)
        plt.plot(betas, avg_sign, marker="o", label=f"L={lattice_size}")
    plt.xlabel("Beta")
    plt.ylabel("Re S")
    plt.title("Re S vs Beta and L (U=20)")
    if grouped:
        all_values = [value for values in grouped.values() for _, value in values]
        y_min = min(all_values)
        y_max = max(all_values)
        if y_max == y_min:
            margin = max(0.05 * abs(y_max), 0.05)
            plt.ylim(y_min - margin, y_max + margin)
        else:
            span = y_max - y_min
            margin = span * 0.1
            plt.ylim(y_min - margin, y_max + margin)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average sign studies")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS)
    parser.add_argument("--thermalization", type=int, default=DEFAULT_THERMALIZATION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--u-values",
        type=float,
        nargs="*",
        default=[0.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        help="Interaction values for U sweep (default L=12, beta=12)",
    )
    parser.add_argument(
        "--beta-values",
        type=float,
        nargs="*",
        default=[4.0, 6.0, 8.0, 12.0],
        help="Beta values for the (beta, L) sweep (U=20)",
    )
    parser.add_argument(
        "--l-values",
        type=int,
        nargs="*",
        default=[4, 6, 8, 12],
        help="Lattice sizes for the (beta, L) sweep (U=20)",
    )
    parser.add_argument(
        "--fft-mode",
        choices=["complex", "real"],
        default=DEFAULT_FFT_MODE,
        help="Select complex FFT (with phases) or real part only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scenario 1: vary U at fixed L and beta.
    u_specs = [
        RunSpec(lattice_size=12, beta=12.0, interaction=u_value, seed=args.seed + idx)
        for idx, u_value in enumerate(args.u_values)
    ]
    u_results = run_experiment(
        u_specs,
        args.sweeps,
        args.thermalization,
        log_dir=output_dir / "logs_u",
        fft_mode=args.fft_mode,
    )
    u_json_path = output_dir / "average_sign_vs_U.json"
    u_json_path.write_text(json.dumps(u_results, indent=2), encoding="utf-8")
    plot_u_sweep(u_results, output_dir / "average_sign_vs_U.png")

    # Scenario 2: vary beta and L at fixed U.
    beta_l_specs: List[RunSpec] = []
    for l in args.l_values:
        for idx, beta_value in enumerate(args.beta_values):
            beta_l_specs.append(
                RunSpec(
                    lattice_size=l,
                    beta=beta_value,
                    interaction=20.0,
                    seed=args.seed + 100 + len(beta_l_specs),
                )
            )
    beta_l_results = run_experiment(
        beta_l_specs,
        args.sweeps,
        args.thermalization,
        log_dir=output_dir / "logs_beta_l",
        fft_mode=args.fft_mode,
    )
    beta_l_json_path = output_dir / "average_sign_vs_beta_L.json"
    beta_l_json_path.write_text(json.dumps(beta_l_results, indent=2), encoding="utf-8")
    plot_beta_l_sweep(beta_l_results, output_dir / "average_sign_vs_beta_L.png")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
