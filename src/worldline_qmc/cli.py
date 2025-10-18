"""
Command line interface for running simulations.

Stage 5 will expose configuration options and orchestrate simulations via the
`run_simulation` entry point.
"""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    raise NotImplementedError("Stage 5 implements the CLI parser.")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    raise NotImplementedError("Stage 5 implements the CLI driver.")

