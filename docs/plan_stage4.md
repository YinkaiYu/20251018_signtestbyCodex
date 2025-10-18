# Stage 4 Plan – Measurement & Output

## 目标
- 根据 `note.md` 中对 $S(X)$ 的定义，实现符号/相位观测量的积累与统计。
- 集成 Monte Carlo 更新后的增量相位，维护样本均值及误差估计。
- 提供结果序列化接口（JSON/CSV）以便后续 Stage 5 的 CLI 使用。

## 实现要点
- 在 `measurement.py` 中编写 `MeasurementAccumulator`：
  - 记录样本数 `N`，累计 `S(X)` 的实部、虚部和模长。
  - 可选记忆平方和以供方差/误差评估（使用阻塞或自相关修正留待未来扩展）。
  - 提供 `push(sign_phase: complex)` 与 `averages()`，输出 `{"re": ..., "im": ..., "abs": ...}`。
  - 验证 `|S(X)|=1`（允许浮点误差）。
- 将测量与 Monte Carlo 状态衔接：Stage 3 的 `phase` 可直接作为当步贡献。
- 设计结果导出函数（Stage 4 完成度允许简单 JSON dump），Stage 5 再封装到 CLI。

## 测试计划
- 单元测试：对 `MeasurementAccumulator` 推入已知复杂数序列，验证均值、模长、样本数。
- 集成测试：构建小型 Monte Carlo 循环（使用假定的更新状态），确保测量模块与 `MonteCarloState` 协作，输出可序列化字典。
- 验证 `push` 对输入 `NaN`/`inf` 做基本防护，避免污染统计。

## 文档
- 完成后在 `AGENTS.md` 添加 Stage 4 记录。
- 若观测量公式在代码中直接使用，添加引用到 `note.md` 相应段落。
