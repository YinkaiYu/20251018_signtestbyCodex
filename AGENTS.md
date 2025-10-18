# Codex Implementation Guide

## Project Overview
- Objective: Implement the momentum-space worldline QMC Monte Carlo described in `note.md`, focusing on sampling the fermionic world lines and permutations while keeping the auxiliary field fixed.
- Primary references: Keep `note.md` authoritative for physics formulas, update it (and this guide) whenever new clarifications arise.
- Programming language: Python (version ≥ 3.10 recommended for typing and dataclasses).

## Development Workflow
- Work in clearly scoped stages; finish, review, and test each stage before moving on.
- After every completed stage: run targeted checks/tests, confirm results with the user if needed, then create a dedicated Git commit.
- Open questions or ambiguities should pause implementation until clarified by the user.
- Maintain self-contained, readable code and documentation so the project can be resumed without prior context.

## Environment Setup
1. Use `uv` to manage dependencies and virtual environment (fallback to `mamba` only if Python packages require compiled extensions that `uv` cannot handle).
2. Minimum tooling:
   - Python ≥ 3.10
   - `numpy`, `scipy` (for FFT if needed), and any utility libraries identified during planning.
3. Document exact setup commands inside this repository (e.g., `uv venv`, `uv pip install numpy`).

## Phased Implementation Plan
1. **Stage 0 – Planning & Scaffolding**
   - Confirm requirements from `note.md`, list data structures, modules, and required inputs/outputs.
   - Draft initial Python package layout (modules, placeholder functions, TODOs).
   - Deliverables: project skeleton, updated plans, Git commit.
2. **Stage 1 – Configuration & Auxiliary Field Handling**
   - Implement configuration parsing (lattice size, β, Δτ, interaction strength, etc.).
   - Implement auxiliary field generation and precomputation of `W_{l,σ}(q)` and related data, matching `note.md`.
   - Include unit tests or scripted checks for FFT outputs.
   - Git commit on completion.
3. **Stage 2 – Worldline Representation**
   - Encode fermion worldlines `K_σ` and permutations `P_σ`, including Pauli constraints.
   - Provide helpers for checking occupancy, computing transition matrix elements `M_{l,σ}`.
   - Tests: deterministic small-lattice consistency checks.
   - Git commit on completion.
4. **Stage 3 – Monte Carlo Updates**
   - Implement Metropolis updates for `K_σ` (local momentum updates) and permutation swaps, using acceptance rules from `note.md`.
   - Ensure incremental phase tracking for `S(X)` and maintainability.
   - Develop diagnostic logging for acceptance ratios.
   - Tests: short MC runs on toy parameters; verify invariants (Pauli compliance, normalization) via assertions.
   - Git commit on completion.
5. **Stage 4 – Measurement & Output**
   - Implement averaging of the sign/phase observable `S(X)` and any additional statistics.
   - Provide serialization (e.g., JSON or CSV) for sampling results.
   - Add unit or integration tests validating measurement accumulation on controlled inputs.
   - Git commit on completion.
6. **Stage 5 – CLI / Run Script**
   - Build a command-line interface or script for executing simulations with configurable parameters.
   - Ensure documentation for invoking main entry points and interpreting results.
   - Final verification run, review, and Git commit.

## Testing & Validation
- Prefer `pytest` for unit/integration tests; document how to run them (`uv run pytest`).
- For stochastic components, include deterministic seeds or sanity-check statistics (mean acceptance, sign estimates) on small systems.
- Keep TODOs or follow-up tasks documented if statistical validation requires longer runs outside quick tests.

## Documentation
- Update `note.md` whenever implementation details affect the theoretical description or vice versa.
- Maintain this `AGENTS.md` alongside development to reflect any plan adjustments, newly added stages, or environment changes.

## Communication Guidance
- If uncertainties arise, halt and contact the user with specific questions.
- Summaries and code comments should assume no prior conversation history, enabling future contributors to continue seamlessly.

---

## Stage 0 – Planning & Scaffolding (2025-10-18)
- Confirmed requirements in `note.md`; key data structures and modules captured in `docs/plan_stage0.md`.
- Established Python package skeleton under `src/worldline_qmc` with placeholder modules for upcoming stages.
- Added `tests/test_placeholder.py` to keep the pytest harness active during scaffolding.
- Declared dependencies (`numpy`, `scipy`, `pytest`) in `pyproject.toml`; CLI documentation resides in `README.md`.

### Environment Setup (uv)
1. `uv venv`
2. `uv pip install -e .[dev]`

Future updates to this plan should timestamp new sections to preserve progress history.

## Stage 1 – Configuration & Auxiliary Field (2025-10-18)
- Documented detailed objectives in `docs/plan_stage1.md`, including validation rules and FFT conventions.
- Implemented `config.load_parameters` with JSON/mapping support, derived quantities, and extensive validations (`src/worldline_qmc/config.py`).
- Added lattice helpers for momentum grids and dispersion (`src/worldline_qmc/lattice.py`).
- Implemented auxiliary-field sampling and Fourier caches with reproducible seeding (`src/worldline_qmc/auxiliary.py`).
- Introduced targeted tests in `tests/test_config.py` and `tests/test_auxiliary.py` (all passing via `uv run pytest`).
- Added `.gitignore` to drop Python bytecode artifacts and removed tracked `__pycache__/` directories.

## Stage 2 – Worldlines & Transitions (2025-10-18)
- Captured momentum-index conventions and testing targets in `docs/plan_stage2.md`.
- Implemented permutation parity, Pauli-safe worldline updates, and momentum index helpers in `src/worldline_qmc/worldline.py`.
- Implemented transition amplitudes using auxiliary-field caches and memoized dispersions in `src/worldline_qmc/transitions.py`.
- Added deterministic tests covering worldline behavior and transition amplitudes (`tests/test_worldline.py`, `tests/test_transitions.py`).
- All Stage 2 tests pass via `uv run pytest` (17 tests).
