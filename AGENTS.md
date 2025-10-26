# Codex Collaboration Guide

## Purpose
- Serve as the working agreement for collaboration, coordination, and ongoing maintenance of the Fermionic Worldline QMC project.
- Capture the state of the repository, preferred workflows, and experiment practices so future work can resume without additional context.

## Communication & Language
- Day-to-day discussion may be in Mandarin; official documentation (README, AGENTS, notebooks, commit messages) remains in English.
- When clarification is needed, pause implementation, raise the question explicitly, and record the resolution here if it affects future work.

## Repository Snapshot
- Canonical upstream: <https://github.com/YinkaiYu/Fermionic-Worldline-QMC.git>
- Primary goal: simulate the momentum-space worldline QMC formulation at half filling with dynamical auxiliary fields, focusing on average-sign measurements.
- Current data products: high-statistics checkerboard/FFT=complex sweep stored under `experiments/output_checkerboard_complex_highsweep/`.

## Development Workflow
1. **Sync & branch** – Ensure local and server clones track the upstream `master`. Create topic branches as needed; keep history linear (rebase preferred).
2. **Environment** – Use Python ≥3.10. Default tooling is `uv`; when building wheels fails, switch to micromamba/conda (`qmc311` environment on the cluster).
3. **Implementation** – Add concise comments only where non-obvious. Follow modular design in `src/worldline_qmc/`. Maintain ASCII unless physics notation requires otherwise.
4. **Testing** – Run `pytest` before committing. For major changes, add or update targeted tests (unit or experiment scripts).
5. **Documentation** – Update README, AGENTS, and `note.md` whenever behavior, workflow, or physics formulas change. Record new datasets or scripts in the Experiments section.
6. **Commit & review** – Commit logically grouped changes with descriptive messages. Avoid bundling generated data unless curated (e.g., aggregated results).

## Experiment Workflow (Local)
- Use `experiments/run_sign_vs_U.py` or related scripts for quick sweeps; archive previous outputs to `experiments/archive/<timestamp>/` before rerunning.
- Use `experiments/run_auxiliary_plan.py` to regenerate the baseline/auxiliary-intensity/interaction sanity suite whenever sampler parameters change.
- Maintain derived datasets (plots/JSON) under `experiments/output_*`. Only version-control curated results (e.g., `..._highsweep`); keep raw runs ignored.
- When new datasets are produced, document parameters, acceptance stats, and locations in README + AGENTS.

## Cluster Workflow Summary
- **Sync the repo** – Clone/pull from upstream. Use `rsync` only for large data folders.
- **Environment** – `micromamba activate qmc311`; install project with `pip install -e .[dev]`.
- **Job scaffold** – Keep parameter tables in `jobs/configs/`, Slurm scripts in `jobs/scripts/`, stdout/err logs in `jobs/logs/`.
- **Slurm template** – Load micromamba, activate environment, `cd` into repo, read parameters via `sed`. Write outputs to `experiments/slurm_runs/${SLURM_JOB_ID}/L{L}_beta{beta}`.
- **Submit/monitor** – `sbatch jobs/scripts/run_*.sbatch`, watch with `squeue -u $USER` and `tail -f jobs/logs/*.out`. Use `--array` for large sweeps.
- **Collect results** – After completion, `rsync` the job directory back to the workstation, preserving hierarchy for plotting/aggregation.
- **Version control** – `experiments/slurm_runs/` is ignored; only aggregate folders (e.g., `experiments/output_checkerboard_complex_highsweep/`) are committed.

## Documentation Maintenance
- README: external-facing overview, installation, usage, data products. Update whenever functionality or datasets change.
- AGENTS: collaboration notes, workflows, and cluster instructions. Keep current; append major decisions or workflow changes.
- note.md: theoretical formulas and acceptance rules. Sync with any algorithmic changes.
- Commit documentation updates alongside code/data changes; log them in the “Documentation Maintenance” subsection above.

## Testing & QA
- `pytest` is mandatory before merge/push. For stochastic updates, hold seeds fixed in tests and ensure acceptance logic matches `note.md`.
- Track performance or acceptance regressions via the JSONL diagnostics (especially for high-U sweeps).

## Historical Reference
- Detailed stage-by-stage notes from the initial implementation are retained below (Stage 0–15). See `docs/plan_stage*.md` for the original planning artifacts.
- Use this log to trace past decisions or locate planning documents; new milestones should follow the updated workflow described above.

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

## Stage 3 – Monte Carlo Updates (2025-10-18)
- Documented update strategy and testing plan in `docs/plan_stage3.md`.
- Added worldline permutation utilities (`inverse`, `swap`) in `src/worldline_qmc/worldline.py` to support boundary handling.
- Implemented Metropolis sweep with momentum and permutation moves in `src/worldline_qmc/updates.py`, including note-referenced acceptance/phase increments.
- Added deterministic Monte Carlo unit tests using stubbed transition amplitudes (`tests/test_updates.py`) and expanded permutation tests (`tests/test_worldline.py`).
- All tests pass after the update via `uv run pytest` (21 tests).

## Stage 4 – Measurement & Output (2025-10-18)
- Recorded measurement plan in `docs/plan_stage4.md`, emphasizing `S(X)` statistics per `note.md`.
- Implemented accumulator with variance diagnostics in `src/worldline_qmc/measurement.py`, including explicit references to the $S(X)$ definition.
- Added measurement unit tests `tests/test_measurement.py` covering averaging, variance, and invalid inputs.
- Test suite (`uv run pytest`) now includes 24 passing cases.

## Stage 5 – CLI & Simulation Orchestration (2025-10-18)
- Documented CLI and orchestration goals in `docs/plan_stage5.md`.
- Implemented full simulation loop in `src/worldline_qmc/simulation.py`, including initialization, scheduling, and diagnostics aligned with `note.md` formulas.
- Added command-line interface in `src/worldline_qmc/cli.py` for running simulations and exporting JSON results.
- Created integration tests `tests/test_simulation.py` and `tests/test_cli.py`; full suite (`uv run pytest`) passes with 27 tests.

## Stage 6 – Repository Audit & Alignment (2025-10-19)
- Cross-checked implementations against `note.md`; local momentum and permutation Metropolis ratios follow the listed `\mathcal{R}_k`/`\mathcal{R}_p` expressions, and `S(X)` accumulation matches the phase definition.
- Added inline comments in `simulation.py` highlighting the correspondence with the product `w(X)=Π_l M_{l,σ}` and boundary terms.
- Remaining optional improvements from `note.md` (importance-sampled momentum proposals, loop/"洗牌" updates) are not implemented yet; current sampler relies on uniform proposals only.
- No extraneous generated files tracked; `.gitignore` covers caches. Repo ready for further extensions.

## Stage 7 – Usage & Documentation Refresh (2025-10-19)
- Expanded `README.md` with module-to-formula mapping, configuration details, programmatic usage example, and CLI/JSON output description.
- Added `.gitignore` rule for `experiments/output/` to keep generated data out of version control.
- Created template configuration `experiments/config_samples/quick_config.json` for quick CLI trials.

## Stage 8 – Average Sign Experiments (2025-10-19)
- Added Matplotlib dependency and initial sweep script for the requested parameter studies (`U`, `β`, `L`).
- Script exported JSON datasets and PNG plots under `experiments/output/`; optional CLI overrides controlled sweep counts and parameter ranges.
- Added regression tests (`tests/test_experiments.py`) ensuring the data generator runs with minimal sweeps and uses `Agg` backend for headless environments.

## Stage 9 – Sampler Improvements (2025-10-19)
- Simulation now initializes worldlines in the zero-temperature Fermi sea (`initial_state='fermi_sea'`) and keeps them constant along imaginary time.
- Introduced configurable FFT modes (`fft_mode='complex'` or `'real'`) so that `W_{l,σ}(q)` can retain full phases or only its cosine component.
- Validation for new configuration flags lives in `config.load_parameters`; auxiliary-field cache stores the selected mode for downstream inspection.

## Stage 10 – Logging & CLI Upgrades (2025-10-19)
- `simulation.run_simulation` writes per-sweep diagnostics (JSONL) when `log_path` is provided; CLI auto-generates a log next to the output JSON unless overridden.
- README expanded with module-to-formula mapping, configuration options (`fft_mode`, `initial_state`, `log_path`), and updated experiment description focusing on `Re S`.

## Stage 11 – Updated Experiments (2025-10-19)
- Data generation split into `experiments/run_sign_vs_U.py` (default `L=12, β=12` sweep over `U`) and `experiments/run_sign_vs_beta_L.py` (`U=20`, sweeping `β` 与 `L ∈ {4,6,8,12}`) with multi-sample measurement support (`measurement_interval` configurable).
- Each run logs to `logs_u/` / `logs_beta_l/` using descriptive filenames (`L{L}_beta{β}_U{U}.jsonl`) and exposes `--fft-mode`, `--measurement-interval`, et al.
- Fresh experimental outputs generated for complex and real FFT modes (64 sweeps, 16 thermalization sweeps) stored in `experiments/results/{complex,real}/` accompanied by JSONL diagnostics.
- Plotting is handled by `plot_sign_vs_U.py` / `plot_sign_vs_beta_L.py`, enabling visualization tweaks without rerunning simulations.

## Stage 12 – Low-U Refinement Sweep (2025-10-20)
- Updated experiment expectations to cover finer interaction resolution near `U=0`. Target list: `U = [0, 0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80, 1.00]`.
- Increased sampler effort for production-quality runs: `--sweeps 64`, `--thermalization 16`, `--measurement-interval 8`, retaining `Δτ = 1/32` and `L = β ∈ {4, 6, 8, 10}`.
- Noted in README’s experiment section so future reruns use the same command presets for both FFT modes (complex / real).

## Stage 13 – Auxiliary Field Modes (2025-10-20)
- Added `auxiliary_mode` configuration with `"random"` (default), `"uniform_plus"` (deterministic +1), and `"checkerboard"` (staggered ±1) options to probe auxiliary-field dependence.
- `generate_auxiliary_field` now routes through `_sample_spatial_field`, supporting deterministic slices without touching RNG state.
- Experiment scripts accept `--auxiliary-mode`; README documents how to run uniform or checkerboard sweeps alongside the standard random-field runs.

## Experiment Workflow Reference (updated 2025-10-20)
- **Plan & communicate** – Confirm desired parameter grids (U, β, L, FFT/auxiliary modes, measurement settings) with the user; record changes immediately in README and this guide to keep future runs reproducible.
- **Archive before reruns** – Move any existing `experiments/output*` directories into a timestamped folder under `experiments/archive/` so new artifacts stay isolated and history remains inspectable.
- **Run generation scripts** – Invoke `uv run python experiments/run_sign_vs_U.py` (or other drivers) with the agreed `--sweeps`, `--thermalization`, `--measurement-interval`, `--fft-mode`, `--auxiliary-mode`, and (when needed) `--auxiliary-moves-per-slice`, customizing `--output-dir` per scenario (complex/real, uniform/checkerboard, etc.). Use deterministic seeds when comparing modes.
- **Plot immediately** – Regenerate figures with `experiments/plot_sign_vs_U.py`, saving PNGs alongside their JSON sources for quick visual review.
- **Validate outputs** – Spot-check JSON/diagnostics (sample counts, acceptance ratios) and summarize notable metrics; add pytest coverage when new configuration branches appear.
- **Document & commit** – Update README/AGENTS with new procedures or findings, run `uv run pytest`, then commit the code and documentation changes together with a concise message. Generated data directories remain ignored unless explicitly versioned.

## Stage 14 – Enhanced Sampler Efficiency (2025-10-20)
- Measurement accumulator now bins samples per sweep via `push_bin`, yielding error bars that better respect autocorrelation.
- Momentum updates cache log-magnitude/phase pairs from transition matrix elements and use sweep-level occupancy masks for O(1) Pauli checks.
- Permutation moves include swaps, short cycles, and small shuffles with parity tracking, improving configuration mixing.
- Acceptance tests rely on log-ratio comparisons, reducing redundant `np.exp` evaluations; README and unit tests updated accordingly.

## Stage 15 – |W|-Weighted Momentum Proposals (2025-10-20)
- Introduced `momentum_proposal` configuration flag (`"w_magnitude"` default, `"uniform"` fallback) and precomputed per-slice proposal tables built from `|W_{l,σ}(q)|`.
- Momentum Metropolis updates now draw proposals via these tables and add the `\log P_l(q_{\text{old}}) - \log P_l(q_{\text{new}})` correction to maintain detailed balance while keeping the stored log-weight purely physical.
- Added regression coverage (`tests/test_updates.py::test_momentum_update_weighted_proposal`) validating the selective cancellation of `|W|` factors and phase preservation.
- README and `note.md` refreshed to document the new proposal mode and acceptance-ratio bookkeeping.

## Stage 16 – Dynamical Auxiliary Field Sampling (2025-10-20)
- Auxiliary spins `s_{il}` are now updated inside each sweep via a Metropolis kernel that recomputes the affected `W_{l,σ}(q)` entries incrementally. `auxiliary_mode` only seeds the initial slice; subsequent configurations are sampled.
- Added `auxiliary_moves_per_slice` to `SimulationParameters`/CLI so the auxiliary update budget can be tuned independently of worldline/permutation moves (default: one attempt per lattice site).
- Cached a global `phase_table` (`e^{iq·r}`) to enable `O(V)` updates of `W_{l,σ}` per accepted flip, and wired proposal-table refreshes so `|W|`-weighted momentum moves stay consistent with the latest auxiliary configuration.
- Extended `tests/test_auxiliary.py`, `tests/test_updates.py`, and `tests/test_simulation.py` to cover FFT-consistent site updates, acceptance bookkeeping, and end-to-end runs with auxiliary dynamics. README, `note.md`, and this guide now describe the new workflow.

## Cluster Workflow Reference (updated 2025-10-20)
- **Sync the repo** – On the server run `git clone` (or `git pull`) so that source files stay aligned with the local workspace. Use `rsync` only for large data folders that are intentionally excluded from version control.
- **Prepare the environment** – Create and activate the micromamba environment `qmc311` (Python 3.11 with numpy/scipy/matplotlib), then install the project with `pip install -e .[dev]`. Verify once with `pytest`. Every job or interactive run starts with `micromamba activate qmc311`.
- **Job scaffold** – Keep `jobs/configs/` for parameter tables (`L beta U seed` rows), `jobs/scripts/` for Slurm submission scripts, and `jobs/logs/` for stdout/stderr. This structure lets `sed -n` read the right row per array index.
- **Slurm script template** – Load micromamba (`eval "$(micromamba shell hook --shell=bash)"`), activate the environment, `cd` into the repo, and read the parameter table. Write outputs to `experiments/slurm_runs/${SLURM_JOB_ID}/L{L}_beta{beta}` and logs to `jobs/logs/`. Use job names such as `yyk_worldlineQMC` on partition `fat6348`.
- **Submit and monitor** – Submit with `sbatch jobs/scripts/run_*.sbatch`. Track progress via `squeue -u $USER` and `tail -f jobs/logs/<file>`. When sweeping many points, use `--array` to iterate over parameter rows.
- **Collect results** – After completion, copy back the entire job directory with `rsync -avP master01:.../experiments/slurm_runs/<jobid>/ experiments/slurm_runs/<jobid>/`. Preserve the hierarchy so plotting/aggregation scripts (e.g., `experiments/plot_sign_vs_U.py`) can consume the data unchanged.
- **Version control** – `experiments/slurm_runs/` is ignored in `.gitignore`; only curated aggregates (e.g., `experiments/output_checkerboard_complex_highsweep/`) are checked in. Confirm the repo is clean before committing summarised outputs.
- **Remote repository** – Canonical upstream lives at `https://github.com/YinkaiYu/Fermionic-Worldline-QMC.git`. Keep both local and server clones pointing to this origin so code and documentation stay synchronized.
