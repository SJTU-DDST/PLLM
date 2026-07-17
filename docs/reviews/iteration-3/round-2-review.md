# Iteration 3 / Round 2 / Reviewer

**总体判断：7/10，Weak Accept for prototype。** 第一轮的容量域与展示边界已经修复，但控制输入仍由手工假设构成。

## 可信贡献

1. UMA/独显容量 scope 已由代码和测试明确区分。
2. recommendation-only guardrail 在 API 与 UI 两处可见。
3. domain-shift synthetic failure 被保留，没有筛掉不利结果。

## 主要问题

**I3.R2.1 Major：95% byte hit 和 5% false-prefetch 不是在线测量。** 实时状态即使来自 NVML，planner 的 cache 参数仍是假设。所有自动建议必须携带 `hypothetical_control_input_not_model_measurement`，不能称为 calibrated policy。

**I3.R2.2 Major：当前 `elastic_resident` 只是动作名。** 因为 controller state 没有物理转换，论文不得写成 vLLM 已进入 elastic mode。建议将状态与 plan 分开：运行时 state 仍来自可执行 vLLM 动作，expert action 仅存在于 projection 对象。

**I3.R2.3 Major：创新性仍依赖未实现数据面。** API 和前端不是算法证据。论文结论应明确，是否存在优于 full/hibernate 的 Pareto 区间仍由真实 trace、I/O 和前台实验决定。

## 次要问题

- 建议报告全量测试数，并区分静态页面测试与离屏截图。
- 手工 API 输入应使用不同 evidence 字段。

## 可证伪条件

若真实 byte hit 明显低于假设，planner 应退出 elastic；否则 phase-boundary 主张不成立。

## 建议结论

**Weak Accept for prototype, Reject as completed system paper**
