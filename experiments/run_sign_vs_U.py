"""Generate average-sign data for varying interaction U.

Supports sweeping multiple lattice sizes and beta values.

Outputs:
  - average_sign_vs_U.json
  - logs_u/L{L}_beta{beta}_U{U}.jsonl per parameter point
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List

from worldline_qmc import auxiliary, config, simulation

DEFAULT_OUTPUT_DIR = Path("experiments/output_u")
DEFAULT_DELTA_TAU = 1.0 / 32.0
DEFAULT_SWEEPS = 16
DEFAULT_THERMALIZATION = 4
DEFAULT_HOPPING = 1.0
DEFAULT_FFT_MODE = "complex"
DEFAULT_MEASUREMENT_INTERVAL = 32
DEFAULT_LATTICE_SIZE = 12
DEFAULT_BETA = 12.0
DEFAULT_AUXILIARY_MODE = "random"


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
    measurement_interval: int,
    auxiliary_mode: str,
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
        "measurement_interval": measurement_interval,
        "auxiliary_mode": auxiliary_mode,
    }
    return config.load_parameters(payload)


def run_experiment(
    specs: List[RunSpec],
    sweeps: int,
    thermalization: int,
    output_dir: Path,
    *,
    fft_mode: str,
    measurement_interval: int,
    auxiliary_mode: str,
) -> List[dict]:
    results: List[dict] = []
    log_dir = output_dir / "logs_u"
    log_dir.mkdir(parents=True, exist_ok=True)

    for idx, spec in enumerate(specs):
        params = build_parameters(
            spec,
            sweeps,
            thermalization,
            fft_mode=fft_mode,
            measurement_interval=measurement_interval,
            auxiliary_mode=auxiliary_mode,
        )
        tag = f"L{spec.lattice_size}_beta{spec.beta}_U{spec.interaction}"
        log_path = log_dir / f"{idx:03d}_{tag}.jsonl"
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


def _expand_lattice_beta_pairs(
    lattice_sizes: List[int],
    beta_values: List[float],
) -> List[tuple[int, float]]:
    if len(lattice_sizes) == len(beta_values):
        return list(zip(lattice_sizes, beta_values))

    if len(lattice_sizes) == 1:
        return [(lattice_sizes[0], beta) for beta in beta_values]

    if len(beta_values) == 1:
        return [(l, beta_values[0]) for l in lattice_sizes]

    # Fall back to full cross product when lengths differ.
    return [(l, beta) for l in lattice_sizes for beta in beta_values]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average sign vs U data generator")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS)
    parser.add_argument("--thermalization", type=int, default=DEFAULT_THERMALIZATION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--u-values",
        type=float,
        nargs="*",
        default=[0.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        help="Interaction values to sample",
    )
    parser.add_argument(
        "--lattice-sizes",
        "--lattice-size",
        type=int,
        nargs="+",
        default=[DEFAULT_LATTICE_SIZE],
        help="Lattice size values to sample (combine with beta values).",
    )
    parser.add_argument(
        "--beta-values",
        "--beta",
        type=float,
        nargs="+",
        default=[DEFAULT_BETA],
        help="Beta values to sample (combine with lattice sizes).",
    )
    parser.add_argument(
        "--fft-mode",
        choices=["complex", "real"],
        default=DEFAULT_FFT_MODE,
        help="Choose complex FFT (with phases) or real cosine component.",
    )
    parser.add_argument(
        "--auxiliary-mode",
        choices=["random", "uniform_plus", "checkerboard"],
        default=DEFAULT_AUXILIARY_MODE,
        help=(
            "Select auxiliary field sampling: random ±1, uniform +1, "
            "or checkerboard staggered ±1."
        ),
    )
    parser.add_argument(
        "--measurement-interval",
        type=int,
        default=DEFAULT_MEASUREMENT_INTERVAL,
        help="Record S(X) after this many Metropolis attempts.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lattice_beta_pairs = _expand_lattice_beta_pairs(
        args.lattice_sizes,
        args.beta_values,
    )

    specs: List[RunSpec] = []
    for pair_idx, (lattice_size, beta_value) in enumerate(lattice_beta_pairs):
        for u_idx, u_value in enumerate(args.u_values):
            seed_offset = pair_idx * len(args.u_values) + u_idx
            specs.append(
                RunSpec(
                    lattice_size=lattice_size,
                    beta=beta_value,
                    interaction=u_value,
                    seed=args.seed + seed_offset,
                )
            )

    results = run_experiment(
        specs,
        args.sweeps,
        args.thermalization,
        output_dir,
        fft_mode=args.fft_mode,
        measurement_interval=args.measurement_interval,
        auxiliary_mode=args.auxiliary_mode,
    )

    (output_dir / "average_sign_vs_U.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
