# Stage 5 Plan – CLI / Run Script

## 目标
- 提供命令行入口以加载配置、运行 Monte Carlo、输出测量结果。
- 复用前序阶段模块（配置、辅助场、更新、测量），确保易于扩展。
- 支持 JSON 输出，便于后续自动化分析。

## 实现要点
- 在 `simulation.py` 中实现完整的模拟流程：
  - 初始化 worldline/permutation、计算初始 `S(X)` 与对数权重。
  - 循环执行 Metropolis sweep，记录接受率，总扫数包括热化阶段。
  - 使用 `MeasurementAccumulator` 记录采样期的观测量。
  - 返回 `SimulationResult`，含测量均值、方差、诊断信息。
- 在 `cli.py` 中构建 `worldline-qmc` CLI：
  - 参数：`--config`（JSON），可选 `--output`、覆盖 sweeps/seed/move 数。
  - 运行模拟后保存结果 JSON，并在终端打印摘要。
- 参考 `note.md`：在涉及公式位置添加注释，以便审阅。

## 测试
- 单元：`simulation.run_simulation` 在小体系（`L=2`，少量 sweep）下输出合理数据。
- CLI：在 `tests/test_cli.py` 中调用 `cli.main`，使用临时配置文件，验证输出文件结构与成功返回。

## 文档
- 更新 `AGENTS.md`，描述 CLI 的使用方式及输出内容。
- 若默认输出位置依赖 `params.output_path`，在 `README.md` 增补调用示例。
