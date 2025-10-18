# Stage 2 Plan – Worldline Representation

## Objectives
- Encode fermion worldlines `K_σ` as momentum-index trajectories respecting Pauli exclusion on each time slice.
- Implement permutation tracking with parity evaluation for the sign factors `sgn(P_σ)`.
- Provide reliable computation of transfer matrix elements `M_{l,σ}(k' ← k)` using the auxiliary-field caches.
- Supply deterministic tests on small lattices covering Pauli checks, permutation parity, and the transition amplitudes in the non-interacting limit.

## Data Representation
- **Momentum indexing**: label momentum points by their FFT index `(i_x, i_y)`; store flattened indices `i = i_x * L + i_y` for compactness. Conversion helpers will recover 2D indices when interacting with FFT arrays.
- **Worldline storage**: `Worldline.trajectories` remains a `numpy.ndarray` of shape `(L_τ, N_σ)` with integer momentum indices. Methods verify uniqueness per slice and support safe in-place updates.
- **Permutation state**: `PermutationState.values` is a length-`N_σ` integer array mapping particle labels. Validity checks ensure it is a permutation of `[0, …, N_σ-1]`, and parity is computed via cycle decomposition.

## Transition Amplitudes
- Retrieve energies with memoized dispersion tables derived from `lattice.momentum_grid` and `lattice.dispersion`.
- Map momentum differences to FFT grid offsets to index `W_{l,σ}(q)` correctly, using modular arithmetic on `(i_x, i_y)`.
- Implement `transition_amplitude` to evaluate
  ```
  M = exp(-0.5 * Δτ * (ε_{k'} + ε_k)) * W_{l,σ}(q) / V
  ```
  with `q = k - k'` expressed on the discrete grid.

## Testing
- **Permutation parity**: verify identity yields `+1` and a simple transposition yields `-1`.
- **Pauli safety**: attempt to insert an already-occupied momentum and expect a `ValueError`.
- **Successful update**: confirm that updates modify trajectories and preserve occupancy structure.
- **Transition amplitude checks**: with `U = 0`, confirm that diagonal transfers reduce to `exp(-Δτ ε_k)` and off-diagonal transfers vanish due to zero Fourier weight.

## Documentation
- Append Stage 2 progress notes to `AGENTS.md` upon completion.
- Reference this plan from the Stage 2 update.
