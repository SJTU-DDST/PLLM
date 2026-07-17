# Iteration 3 / Round 1 / Reviewer

**总体判断：6/10，Borderline。** 控制面接入提高了可演示性，但实时界面也增加了把投影值误读成真实释放量的风险。

## 可信贡献

1. Expert catalog、planner 与 API 已形成可测试的无 GPU 闭环。
2. API 同时输出 evidence、`data_plane_ready` 与 `executable`，具备声明边界的基础。
3. 仍保持原始 Top-22，不把预测 expert 用作替代计算。

## 主要问题

**I3.R1.1 Major：容量域存在硬件语义风险。** 若独显机器使用系统 `MemTotal` 规划 expert residency，系统可能在 VRAM 已满时仍建议完整驻留。必须区分 coherent UMA 与 discrete VRAM，并在 API 中暴露选择口径。

**I3.R1.2 Major：40 层柱状图容易被理解为真实 cache heatmap。** 当前没有 physical slots、route tracer 或 resident-byte telemetry。界面必须持续显示 `CONTROL PLANE ONLY / NOT EXECUTABLE`，报告也必须将柱状图称为投影。

**I3.R1.3 Major：实时 planner 不能触发隐式状态切换。** 论文需要说明 monitor loop 是否会调用 vLLM、更新 controller state 或释放权重，并用测试锁定 recommendation-only guardrail。

## 次要问题

- 移动端 evidence 标签需要检查长文本换行。
- 应把 API 路径和 evidence 枚举写入实验报告。

## 可证伪条件

若 planner 无法正确选择容量域，或者 UI 隐去数据面未就绪状态，则当前产品集成不应作为可信贡献。

## 建议结论

**Borderline**
