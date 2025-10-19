"""Generate average-sign data for varying interaction U.

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
    parser.add_argument("--lattice-size", type=int, default=DEFAULT_LATTICE_SIZE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument(
        "--fft-mode",
        choices=["complex", "real"],
        default=DEFAULT_FFT_MODE,
        help="Choose complex FFT (with phases) or real cosine component.",
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

    specs = [
        RunSpec(
            lattice_size=args.lattice_size,
            beta=args.beta,
            interaction=u_value,
            seed=args.seed + idx,
        )
        for idx, u_value in enumerate(args.u_values)
    ]

    results = run_experiment(
        specs,
        args.sweeps,
        args.thermalization,
        output_dir,
        fft_mode=args.fft_mode,
        measurement_interval=args.measurement_interval,
    )

    (output_dir / "average_sign_vs_U.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
