# Momentum-Space Worldline QMC

This repository hosts a staged implementation of the momentum-space worldline QMC Monte Carlo described in `note.md`. The simulation keeps the auxiliary field fixed and samples fermionic world lines and permutations to study the average sign.

Development proceeds according to `AGENTS.md`. Each stage introduces focused functionality, supporting tests, and documentation updates. See `docs/plan_stage0.md` for the current scaffold and planned components.

## Quick Start

1. 准备配置文件（JSON），包含 `note.md` 中列出的物理参数，例如：

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

2. 运行模拟并输出结果：

```bash
uv run python -m worldline_qmc.cli --config config.json --output result.json --verbose
```

结果将写入 `result.json`，并在终端显示测量摘要。
