# Iteration 5 Round 1 Rebuttal

**作者立场：接受 Reject / Major Revision。** 当前实现修复了部分方法缺陷，但真实 route、在线 resize、Blender QoS 与 exact resume 尚无 GPU 证据，不能据此推翻审稿结论。

## A. 当前已修复

| 评审项 | 回应与证据 |
| --- | --- |
| C2 同窗偏差 | `DecodeRouteWindow` 只用过去窗口构造热集，并在下一完整窗口计分；planner 只读取 `next_window` held-out envelope。 |
| C2 fused batch `unique()` | runtime 以逐 token Top-22 rows 观测；仅 actual miss load 去重，不再把统计样本压成集合。 |
| C3/M2 统一 K 与 horizon | `HorizonAwareLayerPlanner` 求逐层 `{K_l}`，计入 shrink copy、未来 expansion、kernel rebuild、release deadline 与剩余 decode token。 |
| C4 prefill 冲突 | non-full profile 下，只有 idle 且无活跃请求才扩回 512；前台 reserve 或 elastic decode 活跃时请求进入 replay queue。 |
| C6 batch 非线性 | 71 上完成 1/2/4/8/16/22/32-object、100 次、跨完整 20,480-object index 的 strided 测量。 |
| C7 pipe 数据面 | 审稿后新增 shared-MR：子进程直接把父进程共享 mmap 注册为 MR，RDMA READ 后 stdout 只传 16B descriptor。100 次实测 steady p95 为 0.477/0.784/1.513/2.863/26.223/42.710/43.897ms，steady 26.8--72.4Gb/s；仍需 H2D 分解。 |
| M5 RDMA 覆盖 | 最大 batch-32 曲线跨索引验证 3,200 个 source objects，计时前 package SHA、返回后 byte-exact。 |
| m1 route 魔数 | 删除 `prompt_start - 8`；按 vLLM 0.25.1 scheduler 源码严格检查 prompt rows 与 `completion_tokens-1`。 |
| C5 状态检查 | resize 前后比较 allocation fingerprint、bytes 和每个去重 storage 的首/中/尾 content samples。它是 sampled guard，不是 exact resume。 |

以上回答“算法与控制路径是否存在”，不回答 GPU 性能是否成立。

## B. 必须等待 GPU 验证

- first-50 MQA/NQA/TQA full-route capture 尚未产生，任何 `K_l<512` Pareto 区间仍未知。
- `100GiB/s` compaction、`0.75GiB/s` expansion 和 `5ms/layer` rebuild 仍是保守配置，不是 Nemotron 校准结果。
- admission 需要并发 1/2、短请求、前台 active 与公平性验证。
- state island 需要连续 100 次 token-boundary resize、greedy token equality 和完整状态/ledger 验证。
- shared-MR 的终点是 Python memoryview；package parse、H2D、mapping publish 和 kernel stall 尚未分解。
- expert eviction 不减少 Top-22 FLOPs，必须和 Level-0 decode duty cycle 在同一 Blender trace 上联合验证。
- `.cpu().tolist()` trace 同步需 native/full-512/full-512+trace 三组隔离。

当前“风险上界”只是过去最多八个 transition 的最差观测，应称为 **empirical held-out envelope**，不是统计意义的 `Q_{1-delta}` 保证。

## C. 当前不能反驳

1. 没有 exact deep resume；sampled fingerprint 不覆盖完整 KV/Mamba、block table、sequence ledger 与 RNG。
2. shared-MR 仍是 host-staged，尚不是 RDMA-to-GPU 完整数据面。
3. 顶会创新性必须由新 Pareto 区间、对照算法和形式分析支撑，当前只能称可证伪的系统假设。
4. expert eviction 不释放 MoE FLOPs；连续 decode rate limiter 尚未完成。
5. 远端 59.063GiB routed image 不含 15.72GiB non-routed weights，不能替代完整 Level-2 reload。
6. SSE 首文本 chunk 不是严格 engine phase boundary；绕过 PLLM proxy 不受支持。
7. 单模型、单 vLLM 版本不能支持一般性外推。

## Round 2 可证伪实验

| 实验 | 对照 | 通过条件；否则结论 |
| --- | --- | --- |
| Trace 开销 | native、full-512 EER、full-512+trace | TPOT/吞吐损失 `<5%`，否则在线 route collection 不可用 |
| LongBench first-50 | MQA/NQA/TQA；LRU/window-LFU/static held-out | 只在 next-window 计分；同窗 hit 不计 |
| Profile planner | uniform K vs per-layer `{K_l}` | 至少一组释放目标容量且 TPOT `<5x`，否则 H1 失败 |
| Horizon 消融 | 无 horizon、无 transition、完整 planner | 短请求必须拒绝不可摊销 resize |
| 在线转换 | 512→`{K_l}`→512，100 次 | 实际释放、无 OOM、全 wall time、state/greedy tokens 相等 |
| RDMA 端到端 | pipe vs shared-MR vs NVMe | 分解 RDMA/shared view/package/H2D/mapping；Python view 不作为最终 GPU 结果 |
| Blender | no-LLM/full/Level-0/PhaseEER/Level-2 | 前台吞吐 `>=90%` no-LLM，LLM TPOT `<5x`，恢复快于 Level-2 |
| DGX Spark | 同一策略与场景 | 报告 `MemAvailable`、PSI、功耗、渲染吞吐；RTX VRAM 不替代 UMA |

**Round 1 最终回应：** 当前树把固定 K、同窗预测与遗漏迁移成本推进为逐层 horizon-aware 规划器，并把远端 miss 从 pipe 升级到 direct shared host MR；但这些仍是待 GPU 证伪的机制，不是已成立的新系统结论。
