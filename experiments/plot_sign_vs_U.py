"""Plot Re S vs interaction U from generated JSON data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_data(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot(data: List[dict], output: Path, title: str | None = None) -> None:
    interactions = [entry["interaction"] for entry in data]
    avg_sign = [entry["measurements"]["re"] for entry in data]
    errors = []
    bounds = []
    for entry in data:
        var = entry["variances"]["re"]
        samples = entry.get("samples", 0)
        err = np.sqrt(var / samples) if samples > 0 else 0.0
        errors.append(err)
        bounds.append((entry["measurements"]["re"] - err, entry["measurements"]["re"] + err))

    if title is None:
        lattices = {entry["lattice_size"] for entry in data}
        betas = {entry["beta"] for entry in data}
        if len(lattices) == 1 and len(betas) == 1:
            (L,) = lattices
            (beta,) = betas
            title = f"Re S vs U (L={L}, beta={beta})"
        else:
            title = "Re S vs U"

    plt.figure(figsize=(6, 4))
    plt.errorbar(interactions, avg_sign, yerr=errors, marker="o", capsize=4)
    plt.xlabel("Interaction U")
    plt.ylabel("Re S")
    plt.title(title)
    if bounds:
        y_min = min(b[0] for b in bounds)
        y_max = max(b[1] for b in bounds)
        if y_max == y_min:
            margin = max(0.02, abs(y_max) * 0.1)
            y_min -= margin
            y_max += margin
        else:
            span = y_max - y_min
            margin = max(0.02, span * 0.1)
            y_min -= margin
            y_max += margin
        plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Re S vs U from JSON data")
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
