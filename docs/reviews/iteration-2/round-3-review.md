# Iteration 2 / Round 3 / Reviewer

**总体判断：7/10，Weak Accept for control-plane iteration。** 当前代码能作为 planner 和 accounting 原型，但不能证明 conformal prediction 对 Nemotron 有效。

## 剩余问题

`I2.R3.1 Major`：真实实现需要 online rolling coverage 与 domain-shift detector；当前只有离线 fallback。

`I2.R3.2 Major`：Planner 使用离散 slot profiles 和简单阈值。论文必须把“优于 simple threshold”保留为待验证，而不是算法结果。

`I2.R3.3 Minor`：API/UI 必须同时显示 `data_plane_ready=false` 和 evidence source，防止评委模式误读。

**建议：本轮通过，下一轮做产品集成但继续标记模拟数据。**
