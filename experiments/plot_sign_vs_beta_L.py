"""Plot Re S vs beta for multiple lattice sizes from JSON data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_data(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot(data: List[dict], output: Path, title: str | None = None) -> None:
    grouped: Dict[int, List[Tuple[float, float]]] = {}
    entry_lookup: Dict[Tuple[int, float], dict] = {}
    for entry in data:
        grouped.setdefault(entry["lattice_size"], []).append(
            (entry["beta"], entry["measurements"]["re"])
        )
        entry_lookup[(entry["lattice_size"], entry["beta"])] = entry

    if title is None:
        interactions = {entry["interaction"] for entry in data}
        if len(interactions) == 1:
            (U,) = interactions
            title = f"Re S vs Beta and L (U={U})"
        else:
            title = "Re S vs Beta and L"

    plt.figure(figsize=(6, 4))
    all_values: List[float] = []
    for lattice_size, values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        betas, avg_vals, errors = [], [], []
        for beta_val, avg_val in values:
            entry = entry_lookup[(lattice_size, beta_val)]
            betas.append(beta_val)
            avg_vals.append(avg_val)
            var = entry["variances"]["re"]
            samples = entry.get("samples", 0)
            err = np.sqrt(var / samples) if samples > 0 else 0.0
            errors.append(err)
            all_values.append(avg_val)
        plt.errorbar(
            betas,
            avg_vals,
            yerr=errors,
            marker="o",
            capsize=4,
            label=f"L={lattice_size}",
        )

    plt.xlabel("Beta")
    plt.ylabel("Re S")
    plt.title(title)
    if all_values:
        span = max(abs(val) for val in all_values)
        margin = max(span * 0.2, 0.02)
        plt.ylim(-span - margin, span + margin)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Re S vs beta and L from JSON data")
    parser.add_argument("--data", required=True, type=Path, help="Path to JSON data file")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path")
    parser.add_argument("--title", type=str, help="Optional custom title")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_data(args.data)
    plot(data, args.output, title=args.title)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
