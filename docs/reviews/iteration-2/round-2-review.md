# Iteration 2 / Round 2 / Reviewer

**总体判断：6/10。** 证据口径改善，但 cache 字节账需要确认“预取过、随后在使用前被驱逐”的 expert 不能算 useful hit。

## 主要问题

`I2.R2.1 Critical`：Ready-by-deadline 必须以 actual expert 使用时仍在 slot 为准，而不是发起过 prefetch 即算命中。

`I2.R2.2 Major`：Planner 的 false-prefetch bytes 应按 token 汇总 40 层，不能把 per-layer record 当 per-token。

`I2.R2.3 Major`：Hypothetical 95% hit 场景必须在 JSON 里机器可读标记，不能与 observed plan 混淆。

**建议：Borderline。**
