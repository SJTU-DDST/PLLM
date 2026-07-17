# Iteration 2 / Round 1 / Reviewer

**总体判断：5/10。** Predictor、conformal set 和 planner 已有可运行代码，但当前 trace 完全合成，不能为论文算法有效性背书。

## 主要问题

`I2.R1.1 Critical`：Calibration 的 960 records 来自同一 request 内相关 token，不满足 exchangeability。Coverage 1.0 不能写成统计保证。

`I2.R1.2 Critical`：切换 synthetic domain 后 test coverage 为 0，说明 predictor 不具备可迁移性。作者不能只报告 calibration。

`I2.R1.3 Major`：Prediction set 平均 263，远大于 32/64/128 slot budget，conformal coverage 与 cache feasibility 发生冲突。

**建议：Weak Reject。**
