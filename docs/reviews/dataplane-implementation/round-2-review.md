# Data Plane / Round 2 / Reviewer

**总体判断：6/10，Borderline。** Physical resize 和 runtime format 边界已澄清，但启动路径可能仍读取全部 59GiB routed checkpoint，削弱 SparkLoad 价值。

## 主要问题

**DP.R2.1 Major：减少 allocation 不等于减少冷启动 I/O。** 默认 fastsafetensors iterator 可能读取所有 experts 后才由 weight loader 丢弃。Elastic loader 必须在 `get_tensor` 前过滤 nonresident experts，或如实报告全量读放大。

**DP.R2.2 Major：初始 rows 不能使用未初始化内存做 Marlin transform。** 若 routed checkpoint tensors 被全部过滤，必须确定性初始化临时 rows；runtime package 覆盖要发生在第一次 kernel 调用前。

**DP.R2.3 Major：Sleep Level 2 会使旧 mapping 指向已丢弃 allocation。** 深度休眠前必须撤销 mappings，reload 后重新注册 generation。

## 可证伪条件

若 `blktrace/iostat` 显示 128-slot 启动仍读取近完整 59GiB routed checkpoint，或 Level 2 后 mapping generation 未变化，则快速恢复路径不成立。

## 建议结论

**Borderline**
