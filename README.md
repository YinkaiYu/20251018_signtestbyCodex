# Fermionic Worldline QMC

Fermionic Worldline QMC simulates the momentum-space worldline formulation of the half-filled Hubbard model with a dynamical auxiliary Hubbard–Stratonovich field.  The implementation follows the derivations collected in `note.md`, samples fermionic worldlines, permutations, and auxiliary spins, and measures the complex average sign/phase.

The canonical upstream lives at <https://github.com/YinkaiYu/Fermionic-Worldline-QMC.git>.  Issues, feature requests, and data contributions should reference this repository.

## Highlights

- Momentum-space worldline Monte Carlo with explicit permutation sampling.
- Auxiliary field precomputation (`W_{l,σ}(q)`) via FFT, including optional checkerboard/uniform initial patterns.
- Metropolis sampling of the full space-time auxiliary field with incremental FFT updates and automatic refresh of momentum proposals.
- Log-domain Metropolis updates with optional `|W|`-weighted momentum proposals.
- Incremental complex phase accumulation and sweep-level binning for error bars.
- Experiment scripts for parameter sweeps and plotting utilities for quick diagnostics.

## Installation

### Using `uv` (local workstation)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[dev]
```

### Using micromamba / conda (cluster-friendly)

```bash
micromamba create -n qmc311 -c conda-forge python=3.11 numpy scipy matplotlib pip
micromamba activate qmc311
pip install -e .[dev]
```

After activation, `pytest` should report a clean pass across the full suite, confirming the environment.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/worldline_qmc/` | Core sampler, measurement, and CLI modules. |
| `experiments/` | Parameter sweep scripts, plotting utilities, sample configs. |
| `experiments/output_checkerboard_complex_highsweep/` | Aggregated high-statistics dataset (`sweeps=1024`, FFT=complex, auxiliary checkerboard). |
| `experiments/slurm_runs/` | (Ignored) scratch space for per-job outputs from cluster runs. |
| `docs/plan_stage*.md` | Historical planning notes for earlier development stages. |
| `note.md` | Physics derivations and acceptance rules referenced by the code. |

## Quick Start (CLI)

1. Create a configuration JSON (see `experiments/config_samples/quick_config.json` for a template).
2. Run the CLI:

   ```bash
   python -m worldline_qmc.cli --config config.json --output result.json --log result.jsonl
   ```

The CLI writes measurements/diagnostics to `result.json`, spill per-sweep diagnostics to `result.jsonl`, and echoes summary statistics to stdout.
Use `--auxiliary-moves` when you want to explicitly set the number of auxiliary spin-flip proposals per slice (defaults to one per site when omitted).

## Programmatic Usage

```python
from worldline_qmc import auxiliary, config, simulation

params = config.load_parameters({
    "lattice_size": 4,
    "beta": 4.0,
    "delta_tau": 0.25,
    "hopping": 1.0,
    "interaction": 4.0,
    "sweeps": 20,
    "thermalization_sweeps": 10,
    "seed": 1,
})

aux_field = auxiliary.generate_auxiliary_field(params)
result = simulation.run_simulation(params, aux_field)
print(result.to_dict())
```

## Configuration Reference

`SimulationParameters` accepts the following keys:

- `lattice_size` (`L`) – Linear size of the square lattice (`V = L²`).
- `beta` (`β`), `delta_tau` (`Δτ`) – Imaginary-time extent and slice length (`L_τ = β/Δτ`, enforced integer).
- `hopping` (`t`), `interaction` (`U`) – Hubbard model couplings (`λ = arccosh(exp(Δτ U / 2))`).
- `sweeps`, `thermalization_sweeps` – Measurement and warm-up sweep counts.
- `worldline_moves_per_slice`, `permutation_moves_per_slice` – Override default proposal budgets (defaults scale with particle count).
- `momentum_proposal` – `"w_magnitude"` (importance sampling using `|W|`, default) or `"uniform"`.
- `fft_mode` – `"complex"` (retain phases) or `"real"` (cosine component only).
- `auxiliary_mode` – `"random"`, `"uniform_plus"`, or `"checkerboard"`; controls the initial auxiliary slice prior to sampling.
- `auxiliary_moves_per_slice` – Overrides the number of auxiliary spin-flip proposals per imaginary-time slice (defaults to one attempt per site when unset).
- `initial_state` – `"fermi_sea"` or `"random"`.
- `measurement_interval` – Attempts between phase samples (defaults to one full sweep).
- `seed`, `output_path`, `log_path` – RNG seed and optional file outputs.

Unknown config keys are preserved in `SimulationParameters.extra` for downstream metadata.

## Experiments

The `experiments/` directory provides ready-made scripts:

- `run_sign_vs_U.py` – Sweep interaction strength for one or more `(L, β)` pairs.
- `run_sign_vs_beta_L.py` – Sweep lattice size / β for fixed U.
- `run_auxiliary_plan.py` – Automate baseline checks, auxiliary-move scans, and small-U sanity sweeps for the dynamical auxiliary-field sampler.
- `plot_sign_vs_U.py`, `plot_sign_vs_beta_L.py` – Visualize JSON results.

Example high-statistics sweep (checkerboard auxiliary, FFT=complex):

```bash
python experiments/run_sign_vs_U.py \
  --output-dir experiments/slurm_runs/${SLURM_JOB_ID}/L${L}_beta${BETA} \
  --sweeps 1024 --thermalization 64 \
  --measurement-interval 8 \
  --fft-mode complex \
  --auxiliary-mode checkerboard \
  --u-values 0 1 2 5 10 15 20 25 30 \
  --lattice-sizes 4 6 8 10 \
  --beta-values 4 6 8 10
```

Cluster workflows, including micromamba environment setup and `sbatch` templates, are detailed in `AGENTS.md` (“Cluster Workflow Reference”).

## Data Products

- `experiments/output_checkerboard_complex_highsweep/average_sign_vs_U.json` – Combined dataset from `1024`-sweep checkerboard/complex runs (36 parameter points).
- `experiments/output_checkerboard_complex_highsweep/average_sign_vs_U.png` – Plot showing `Re ⟨S⟩` with standard-error bars; generated via `plot_sign_vs_U.py`.

Per-job raw outputs (JSON + JSONL) are stored under `experiments/slurm_runs/<jobid>/L{L}_beta{β}/` and should be archived or aggregated before committing.

## Testing

Continuous validation uses `pytest`:

```bash
pytest            # in any configured environment
```

The suite covers parameter validation, auxiliary-field FFTs, transition amplitudes, momentum/permutation updates, measurement accumulation, simulation orchestration, CLI behavior, and experiment helpers.

## Further Reading

- `note.md` – Physics background, acceptance ratios, and phase-update formulas.
- `AGENTS.md` – Collaboration/workflow guide (language policy, development routines, cluster instructions).
- `docs/plan_stage*.md` – Historical notes from the initial staged implementation.
