# Iteration 2 / Round 2 / Author

- `I2.R2.1`：接受并修复。Simulator 在 actual use 时重新检查 cache；prefetched-but-evicted bytes 计入 false/wasted prefetch，actual expert 重新计 blocking miss。该问题由初版单元测试实际触发。
- `I2.R2.2`：接受。Planner 输入将 per-record false bytes 乘以 40 MoE layers，结果中 observed shift 为约 9.6GiB/token debt并进入 hibernate。
- `I2.R2.3`：接受。Hypothetical plan 的 `evidence` 固定为 `hypothetical_control_input_not_model_measurement`；文件顶层列出 forbidden claims。

修订没有改变算法结果，只纠正 accounting 和 evidence metadata。
