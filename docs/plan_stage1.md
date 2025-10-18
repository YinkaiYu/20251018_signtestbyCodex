# Stage 1 Plan – Configuration & Auxiliary Field

## Goals
- Flesh out configuration handling with validation and derived quantities.
- Implement auxiliary field sampling and Fourier precomputation of `W_{l,σ}(q)`.
- Provide deterministic testing for both components using small lattice examples.

## Configuration Handling
- **Input formats**: support loading from a mapping (already validated) or a JSON file on disk. Raise `ValueError` for unsupported file suffixes.
- **Parameters**: lattice size `L`, inverse temperature `beta`, time step `delta_tau`, hopping amplitude `t`, interaction strength `U`, RNG seed, sweep counts, and move scheduling numbers.
- **Derived quantities**:
  - `time_slices = round(beta / delta_tau)` with a tolerance check to ensure `beta` is an integer multiple of `delta_tau`.
  - `volume = L^2`, `num_particles = volume // 2` for each spin.
  - Precompute `lambda = arccosh(exp(delta_tau * U / 2))`; handle the non-interacting limit (`U = 0`) gracefully.
- **Validation**: enforce positive `L`, `beta`, `delta_tau`, and non-negative sweep counts. Confirm that `volume` is even (required for half-filling).
- **Testing**: create fixture dictionaries with known results, verify JSON loading, derived slices, and error paths.

## Auxiliary Field
- **Sampling**: draw `s_{i,l} ∈ {±1}` independently for each space-time site using the shared RNG helper, seeded via `SimulationParameters.seed`.
- **Precomputation**:
  - Compute `lambda` from configuration once.
  - For each time slice, evaluate `exp(± lambda * s_il)` and perform a 2D FFT to obtain `W_{l,σ}(q)` on the lattice momentum grid.
  - Cache both the complex values and their magnitudes/phases for efficient access.
- **Momentum bookkeeping**:
  - Use `np.fft.fftfreq(L) * 2π` to define momentum components.
  - Ensure FFT conventions match the definition `W_{l,σ}(q) = Σ_i exp(i q · r_i) exp(±λ s_il)` by conjugating the forward FFT result.
- **Testing**:
  - With `U = 0`, confirm that `W_{l,σ}(0) = L^2` and off-zero components vanish (within numerical tolerance).
  - With a fixed RNG seed on a small lattice (e.g., `L = 2`), compare against manually computed `W` values.

## Documentation Updates
- Append a Stage 1 progress section to `AGENTS.md`.
- Expand `README.md` environment instructions if needed (e.g., how to invoke configuration loaders).

## Outstanding Questions
- None identified; revisit once implementation surfaces ambiguities.
