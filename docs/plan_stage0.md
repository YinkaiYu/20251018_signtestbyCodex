# Stage 0 Planning

## Requirements Recap
- Follow the formulation in `note.md` for momentum-space worldline QMC with fixed auxiliary field.
- Half-filled square lattice with periodic boundary conditions; lattice size `L` implies `N_sigma = L^2 / 2`.
- Discrete imaginary time slices `L_tau = beta / delta_tau`; momentum transfer kernels `M_{l, sigma}` depend on precomputed `W_{l, sigma}(q)`.
- Monte Carlo samples `K_sigma` (worldlines) and `P_sigma` (permutations) using `|w(X)|`; observables focus on the complex phase/sign `S(X)`.

## Proposed Package Layout

```
src/worldline_qmc/
    __init__.py
    config.py          # Read/validate physical and algorithmic parameters.
    lattice.py         # Lattice utilities (momenta grids, dispersion epsilon_k).
    auxiliary.py       # Generate auxiliary field s_il and precompute W_{l,sigma}(q).
    worldline.py       # Data structures for K_sigma and permutations P_sigma.
    transitions.py     # Evaluate M_{l, sigma}(k' <- k) and related helpers.
    updates.py         # Monte Carlo move proposals and acceptance logic.
    measurement.py     # Accumulate sign/phase statistics and diagnostics.
    simulation.py      # Orchestrate sweeps, scheduling, and measurement loops.
    cli.py             # Command line entry point for Stage 5.
    rng.py             # Random number utilities with seeding support.
tests/
    __init__.py
    conftest.py        # Shared fixtures (e.g., small lattice configs).
    test_placeholder.py
```

The package targets a `uv`-managed environment with Python ≥ 3.10, `numpy`, and `scipy`.

## Core Data Structures (Sketch)
- `SimulationParameters`: dataclass storing lattice size, `beta`, `delta_tau`, hopping `t`, interaction `U`, total slices, seeds, and move scheduling counts.
- `AuxiliaryFieldSlice`: holds `s_il` for a single time slice and cached FFT results for `W_{l, sigma}(q)`.
- `AuxiliaryField`: container across `L_tau` slices with convenience access to magnitudes/phases.
- `Worldline`: per-spin representation of occupied momenta across slices (`shape=L_tau x N_sigma`) with Pauli-safe manipulation helpers.
- `PermutationState`: per-spin permutation representation (e.g., `numpy.ndarray` storing integers) with parity tracking.
- `MonteCarloState`: aggregate of auxiliary field reference, worldlines, permutations, cached weights, and current complex phase accumulator.
- `MeasurementAccumulator`: running sums for `S(X)` (real/imag), magnitude, counts, and optional autocorrelation metadata.

## Inputs and Outputs
- **Inputs**: configuration file or CLI options specifying lattice parameters, interaction strength, Trotter step, number of sweeps, RNG seed, and output targets.
- **Outputs**: serialized measurement summaries (JSON/CSV), optional diagnostic logs (acceptance ratios, occupancy checks), and reproducibility metadata (parameters, seed, timestamp).

## Near-Term Tasks (Stages 1-2 Preview)
- Implement configuration parsing and validation logic in `config.py`.
- Build auxiliary field generation and FFT precomputation workflow in `auxiliary.py`.
- Flesh out `Worldline` and permutation containers with deterministic initialization and Pauli checks.

