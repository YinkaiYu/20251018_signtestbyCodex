"""Command line interface for momentum-space worldline QMC."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from . import auxiliary, config, simulation


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run momentum-space worldline QMC with fixed auxiliary fields.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to JSON configuration file (see note.md for parameters).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write simulation results as JSON.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Optional path to write per-sweep diagnostics (JSON lines).",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        help="Override the number of measurement sweeps (after thermalization).",
    )
    parser.add_argument(
        "--thermalization",
        type=int,
        help="Override the number of thermalization sweeps.",
    )
    parser.add_argument(
        "--worldline-moves",
        type=int,
        help="Override worldline moves per slice for scheduling purposes.",
    )
    parser.add_argument(
        "--permutation-moves",
        type=int,
        help="Override permutation moves per slice for scheduling purposes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override RNG seed used for auxiliary field and updates.",
    )
    parser.add_argument(
        "--fft-mode",
        choices=["complex", "real"],
        help="Override FFT mode (complex phases vs real cosine component).",
    )
    parser.add_argument(
        "--initial-state",
        choices=["fermi_sea", "random"],
        help="Override initial worldline configuration strategy.",
    )
    parser.add_argument(
        "--measurement-interval",
        type=int,
        help="Record S(X) every N Metropolis attempts (default: once per sweep).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a summary of measurements and diagnostics to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    params = config.load_parameters(args.config)
    if args.sweeps is not None:
        params = replace(params, sweeps=args.sweeps)
    if args.thermalization is not None:
        params = replace(params, thermalization_sweeps=args.thermalization)
    if args.seed is not None:
        params = replace(params, seed=args.seed)
    if args.worldline_moves is not None:
        params = replace(params, worldline_moves_per_slice=args.worldline_moves)
    if args.permutation_moves is not None:
        params = replace(params, permutation_moves_per_slice=args.permutation_moves)
    if args.fft_mode is not None:
        params = replace(params, fft_mode=args.fft_mode)
    if args.initial_state is not None:
        params = replace(params, initial_state=args.initial_state)
    if args.measurement_interval is not None:
        params = replace(params, measurement_interval=args.measurement_interval)
    if args.output is not None:
        params = replace(params, output_path=args.output)

    log_path = args.log
    if log_path is None and (params.output_path or args.output):
        output_path = params.output_path or args.output
        if output_path is not None:
            log_path = output_path.with_name(output_path.name + ".log.jsonl")
    if log_path is not None:
        params = replace(params, log_path=log_path)

    aux_field = auxiliary.generate_auxiliary_field(params)
    schedule = None  # Allow simulation.run_simulation to resolve default moves.
    result = simulation.run_simulation(params, aux_field, schedule=schedule)

    output_path = params.output_path or args.output
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)

    if args.verbose or output_path is None:
        measurements = result.measurements
        diag = result.diagnostics
        summary = (
            f"Samples={result.samples} | Re={measurements['re']:.6f} "
            f"Im={measurements['im']:.6f} | |S|={measurements['abs']:.6f} | "
            f"Acc_k={diag['momentum_acceptance']:.3f} "
            f"Acc_p={diag['permutation_acceptance']:.3f}"
        )
        print(summary)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
