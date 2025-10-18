"""
Random number utilities.

Encapsulates NumPy Generator management to ensure reproducibility across
stages. Stage 1 will flesh out seeding helpers and distribution-specific
sampling.
"""

from __future__ import annotations

import numpy as np


def make_generator(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy random Generator."""

    return np.random.default_rng(seed)

