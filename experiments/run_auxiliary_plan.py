"""Automate the auxiliary-field sampling experiments described in AGENTS.md.

This script executes three stages:

1. Baseline run using the provided configuration.
2. Auxiliary-move intensity scan for several multiples of the lattice volume.
3. A short interaction (U) sweep for sanity checks.

Each run reuses the worldline QMC simulation entry points, writes individual
JSON artifacts, and records an overall summary for later plotting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from worldline_qmc import auxiliary, config, simulation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run auxiliary-field planning sweeps.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the base JSON configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/auxiliary_plan_runs"),
        help="Directory for run artifacts (default: experiments/auxiliary_plan_runs).",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        help="Override measurement sweeps for every run.",
    )
    parser.add_argument(
        "--thermalization",
        type=int,
        help="Override thermalization sweeps for every run.",
    )
    parser.add_argument(
        "--measurement-interval",
        type=int,
        help="Override measurement interval for every run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override RNG seed used for all runs.",
    )
    parser.add_argument(
        "--aux-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Multipliers of L^2 for auxiliary moves per slice (default: 0.5 1 2).",
    )
    parser.add_argument(
        "--u-values",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 20.0],
        help="Subset of interaction strengths to sanity-check (default: 0, 10, 20).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run diagnostics to stdout.",
    )
    return parser.parse_args(argv)


def sanitize(label: str) -> str:
    return label.replace(" ", "_").replace("/", "-")


def run_simulation(
    params: config.SimulationParameters,
    label: str,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aux_field = auxiliary.generate_auxiliary_field(params)
    result = simulation.run_simulation(params, aux_field)
    record = {
        "label": label,
        "parameters": {
            "lattice_size": params.lattice_size,
            "beta": params.beta,
            "delta_tau": params.delta_tau,
            "interaction": params.interaction,
            "sweeps": params.sweeps,
            "thermalization_sweeps": params.thermalization_sweeps,
            "measurement_interval": params.measurement_interval,
            "momentum_proposal": params.momentum_proposal,
            "fft_mode": params.fft_mode,
            "auxiliary_mode": params.auxiliary_mode,
            "auxiliary_moves_per_slice": params.auxiliary_moves_per_slice,
            "seed": params.seed,
        },
        "measurements": result.measurements,
        "diagnostics": result.diagnostics,
        "samples": result.samples,
    }

    path = output_dir / f"{sanitize(label)}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def apply_overrides(
    params: config.SimulationParameters,
    **overrides: object,
) -> config.SimulationParameters:
    filtered = {k: v for k, v in overrides.items() if v is not None}
    if not filtered:
        return params
    return replace(params, **filtered)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_params = config.load_parameters(args.config)
    base_params = apply_overrides(
        base_params,
        sweeps=args.sweeps,
        thermalization_sweeps=args.thermalization,
        measurement_interval=args.measurement_interval,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, object]] = []

    # Stage 1: Baseline
    baseline_label = "baseline_default"
    record = run_simulation(base_params, baseline_label, args.output_dir / "baseline")
    record["stage"] = "baseline"
    summary.append(record)
    if args.verbose:
        print(f"[baseline] {record['measurements']} diag={record['diagnostics']}")

    # Stage 2: Auxiliary intensity scan
    volume = base_params.volume
    for multiplier in args.aux_multipliers:
        label = f"aux_multiplier_{multiplier:g}"
        if multiplier <= 0:
            moves = 0
        else:
            moves = max(1, int(round(multiplier * volume)))
        params = apply_overrides(base_params, auxiliary_moves_per_slice=moves)
        rec = run_simulation(params, label, args.output_dir / "auxiliary_scan")
        rec["stage"] = "auxiliary_scan"
        rec["multiplier"] = multiplier
        summary.append(rec)
        if args.verbose:
            print(f"[aux_scan:{multiplier:g}] {rec['measurements']} diag={rec['diagnostics']}")

    # Stage 3: Interaction sanity sweep
    for u in args.u_values:
        label = f"interaction_{u:g}"
        params = apply_overrides(base_params, interaction=u, auxiliary_moves_per_slice=0)
        rec = run_simulation(params, label, args.output_dir / "interaction_sanity")
        rec["stage"] = "interaction_sanity"
        rec["interaction"] = u
        summary.append(rec)
        if args.verbose:
            print(f"[interaction:{u:g}] {rec['measurements']} diag={rec['diagnostics']}")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if args.verbose:
        print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
