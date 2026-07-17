# Iteration 2 / Round 1 / Author

- `I2.R1.1`：接受。论文明确只称 synthetic empirical coverage，并保留“交换性下 marginal”限制；真实数据必须按 request split 且包含多个 calibration requests。
- `I2.R1.2`：接受。Test coverage 0 被写入论文和实验报告，作为 drift fallback 反例，不调参隐藏。
- `I2.R1.3`：接受。Simulator 输出 `prediction_over_budget_records`；planner把 false-prefetch debt 纳入带宽，observed-shift 场景直接 hibernate。

本轮不反驳 reviewer；这些失败正是 no-GPU simulator 应提前暴露的问题。
