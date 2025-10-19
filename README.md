# Momentum-Space Worldline QMC

This repository implements the momentum-space worldline QMC algorithm described in `note.md`. The code keeps the auxiliary Hubbard-Stratonovich field fixed and samples fermionic worldlines (`K_σ`) together with permutations (`P_σ`) to estimate the average sign

Development follows the staged plan recorded in `AGENTS.md`. Each stage has a companion planning note in `docs/plan_stage*.md`. Below is a breakdown of the modules and their connection to the formulas in `note.md`.

## Module Overview & Formula Mapping

| Module | Description | Key references to `note.md` |
| --- | --- | --- |
| `worldline_qmc/config.py` | Parses simulation parameters into a `SimulationParameters` dataclass, checks `L`, `β`, `Δτ`, etc. | Parameter definitions preceding Eq. for `L_τ = β / Δτ` |
| `worldline_qmc/lattice.py` | Provides momentum grid (`np.fft.fftfreq`) and dispersion `ε_k = -2t( cos k_x + cos k_y )`. | Free-fermion dispersion expressions in the opening section |
| `worldline_qmc/auxiliary.py` | Generates auxiliary field `s_{i,l} = ±1` and computes `W_{l,σ}(q)` via FFT. | Definitions of `W_{l,σ}(q)` and λ = arccosh(exp(Δτ U / 2)) |
| `worldline_qmc/worldline.py` | Encodes worldlines and permutations, enforces Pauli exclusion. | Discrete worldline representation and `k_{l,σ}^{(n)}` permutations |
| `worldline_qmc/transitions.py` | Evaluates `M_{l,σ}(k' ← k) = exp[-Δτ(ε_{k'}+ε_k)/2] W_{l,σ}(k-k')/V`. | Equation defining the transfer matrix elements `M_{l,σ}` |
| `worldline_qmc/updates.py` | Implements Metropolis updates using `𝓡_k`, `𝓡_p` ratios and phase increments `ΔΦ`. | Acceptance ratios and phase increment formulas (Section on Monte Carlo updates) |
| `worldline_qmc/measurement.py` | Accumulates the complex phase observable `S(X)` as defined in `note.md`. | Definition of `S(X)` and accumulation of `Φ(X)` |
| `worldline_qmc/simulation.py` | Orchestrates initialization (Fermi-sea default), sweeps, measurement logging, returns diagnostics. | Product representation of `w(X)` and boundary links through `P_σ` |
| `worldline_qmc/cli.py` | Provides a CLI driver that loads configs, runs simulations, and writes JSON output. | Automates the workflow implied throughout `note.md` |

For further detail reference the staged notes (`docs/plan_stage*.md`) which include expectations, design sketches, and test strategies for each milestone.

## Quick Start

1. 准备配置文件（JSON），包含 `note.md` 中列出的物理参数，例如（`experiments/config_samples/quick_config.json` 提供了一个模板）：

```json
{
  "lattice_size": 2,
  "beta": 1.0,
  "delta_tau": 0.5,
  "hopping": 1.0,
  "interaction": 0.0,
  "sweeps": 10,
  "thermalization_sweeps": 5,
  "seed": 42
}
```

需要测试“无相位”情形时，可以在配置中加入 `"fft_mode": "real"`（同时也可设置 `"initial_state": "random"` 以恢复旧的随机初态）。

2. 运行模拟并输出结果：

```bash
uv run python -m worldline_qmc.cli --config config.json --output result.json --verbose
```

结果将写入 `result.json`，并在终端显示测量摘要。若未指定 `--log`，CLI 会默认生成 `result.json.log.jsonl` 记录逐 sweep 诊断。

## Configuration Fields

`SimulationParameters` accepts the following keys:

- `lattice_size` (`L`) – square lattice linear dimension.
- `beta` (`β`), `delta_tau` (`Δτ`) – define `L_τ = β / Δτ`; validation enforces integral slice count.
- `hopping` (`t`), `interaction` (`U`) – Hubbard model parameters. `λ = arccosh(exp(Δτ U / 2))` appears in auxiliary-field generation.
- `sweeps`, `thermalization_sweeps` – number of measurement and warm-up sweeps.
- `worldline_moves_per_slice`, `permutation_moves_per_slice` – optional overrides for scheduling; default heuristics follow the number of particles per spin and time slices.
- `seed` – RNG seed controlling auxiliary field sampling and Metropolis proposals.
- `output_path` – optional JSON path for CLI output.
- `log_path` – optional JSON-lines diagnostics file；CLI 默认基于输出路径命名。
- `fft_mode` – `"complex"`（保留 FFT 相位）或 `"real"`（仅使用实部、无相位），方便比较符号表现。
- `initial_state` – `"fermi_sea"`（零温费米海，随虚时保持不变）或 `"random"`。

`config.load_parameters` accepts a JSON file or dictionary. Unknown fields go into `SimulationParameters.extra` for downstream analysis metadata.

## Running Programmatically

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

## Output Structure

The JSON produced by the CLI or `SimulationResult.to_dict()` includes:

- `measurements`: averages of `S(X)` components (`re`, `im`, `abs`).
- `variances`: sample variances (no autocorrelation correction) for diagnostics.
- `diagnostics`: counts/acceptance ratios for momentum & permutation moves, plus sweep totals.
- `samples`: number of measurement samples accumulated (`sweeps`).

## Experiments & Visualization

`experiments/run_average_sign.py` reproduces the parameter studies described in the project request（默认输出 `Re S` 数据与日志，**不直接绘图**）：

数据生成拆分为两个脚本：

```bash
# L=12, β=12, varying U
uv run python experiments/run_sign_vs_U.py \
    --output-dir experiments/output_u \
    --sweeps 64 --thermalization 16 \
    --fft-mode complex --measurement-interval 32

# U=20, varying β 与 L
uv run python experiments/run_sign_vs_beta_L.py \
    --output-dir experiments/output_beta_L \
    --sweeps 64 --thermalization 16 \
    --fft-mode complex --measurement-interval 32
```

`--u-values`、`--beta-values`、`--l-values`、`--fft-mode`、`--measurement-interval`、`--seed` 可定制采样点；日志存放于 `logs_u/`、`logs_beta_l/`。

绘图使用独立脚本：

```bash
uv run python experiments/plot_sign_vs_U.py \
    --data experiments/output_u/average_sign_vs_U.json \
    --output experiments/output_u/average_sign_vs_U.png

uv run python experiments/plot_sign_vs_beta_L.py \
    --data experiments/output_beta_L/average_sign_vs_beta_L.json \
    --output experiments/output_beta_L/average_sign_vs_beta_L.png
```

图像展示 `Re S` 与标准误差（来自方差/有效样本数），y 轴按数据最小/最大值自动设置并留出 10% 余量。可使用 `--title` 覆盖默认标题。

## Tests

Test suite is managed with `pytest`. Install dependencies with `uv` and run:

```bash
uv pip install -e .[dev]
uv run pytest
```

Tests cover configuration parsing, auxiliary-field FFTs, worldline/permutation constraints, transition amplitudes, Metropolis updates, measurement accumulation, simulation orchestration, and CLI behavior. Stage-specific plans detail targeted test intentions.
