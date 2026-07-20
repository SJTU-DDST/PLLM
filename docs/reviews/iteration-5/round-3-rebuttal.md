# Iteration 5 Round 3 Rebuttal

感谢审稿人继续检查实现细节。我们接受这些意见对异步一致性、请求隔离、在线 fallback 和评估口径的要求。当前修改解决了大部分软件正确性问题，但没有产生新的 GPU、LongBench 或 Blender 结果，因此我们不请求基于机制实现推翻 Reject 结论。

## 修改摘要

当前版本新增：

- request、route、capacity 三重 generation 及执行前复验；
- `min_tokens - exact delta token IDs` 后向下取整到 128 token 的 horizon lower bound；
- PhaseEER 单序列、单请求执行域；
- request boundary 同时清空 decode 与 prefill history；
- actual blocking expert-load debt 与 controller fallback；
- Pareto frontier、parent pointer 和 250 组 exhaustive-oracle cases；
- 按 layer bytes 计算在线 byte hit；
- 异构 `{K_l}` 的 request-local token-total p50/p95/p99/CVaR95；
- capacity-aware pending action；
- shared-MR chunk iterator。

这些修改仍属于机制与 CPU/mock 证据，不等同于 GPU 性能结论。

## 逐项回应

### C1：async key 缺少 request/route/capacity generation

**接受并已修改。**

异步 key 现在包含：

\[
G=(G_{request},G_{route},G_{capacity},H_{bucket},
K^{current},R_{target},N_{decode})
\]

新请求递增 `request_generation`；每次 sealed route window 递增 `route_generation`；capacity envelope 变化递增 `capacity_generation`。异步结果只有 key 与当前请求完全一致时才会发布。[expert_control.py](/home/cong/PLLM/pllm/expert_control.py:147)

执行 resize 前，controller 再比较三重 generation、当前 horizon 与计划 horizon，并通过 uncached runtime status 复验 request/route generation。[controller.py](/home/cong/PLLM/pllm/controller.py:728)

**剩余边界：** 这是一套 fail-closed 乐观并发检查，不是跨 sensor、controller 和 vLLM 的事务。capacity sensor 在最后检查与动作之间仍可能变化；下一轮会纠正，但尚未证明严格实时一致性。

### C2：并发和 `n>1` 污染 request-local predictor

**接受并已修改。**

- EER 启动脚本固定 `--max-num-seqs 1`。
- OpenAI proxy 拒绝 `n != 1`。
- 已存在活动请求时，新请求不会进入 engine。
- 新 prefill 清空 decode counts、validation、prefill tail 和 prefill counts。
- offline evaluator 只在单个请求内部形成 previous→next transitions。

证据见 [controller.py](/home/cong/PLLM/pllm/controller.py:372)、[decode_residency.py](/home/cong/PLLM/pllm/decode_residency.py:127) 和 [run_vllm_eer.sh](/home/cong/PLLM/scripts/run_vllm_eer.sh:21)。

**剩余边界：** 直接绕过 PLLM proxy 的客户端不受保护；并发 admission 仍需压力测试。本文不再把单请求 locality 外推到 continuous batching。

### C3：缺少 actual miss-debt fallback

**接受并已修改。**

runtime 对每层 `data_plane.ensure()` 计时。该 wall interval 覆盖 remote/local source、package parse、CPU→GPU slot copy 和 mapping publish；在最后一个 MoE 层聚合成 token load debt。实际 debt 超过预算后，controller 根据 capacity envelope 执行：

- 仅计算压力：Level-0 yield；
- 必须释放容量：Level-2 hibernate。

实现位于 [vllm_eer_runtime.py](/home/cong/PLLM/pllm/vllm_eer_runtime.py:595) 和 [controller.py](/home/cong/PLLM/pllm/controller.py:680)。

**剩余边界：** fallback 由 250ms controller poll 观察，不是 runtime 内立即抢占；计时不包含随后 fused-MoE kernel 的执行。真实 GPU tail 和触发延迟仍未测量。

### C4：property test 太弱

**接受并已加强。**

planner 现在每个 reclaim bucket 保存非支配 frontier，使用 parent pointer 回溯。支配条件包含 reclaim、objective、mean miss、miss bytes、miss latency、immediate time 和 total transition。

测试生成 250 个随机三层异构实例，对每个实例枚举全部 `3^3` profiles，并比较：

- 是否存在可行解；
- hit、I/O、deadline 和 slowdown 约束；
- 最优 slowdown；
- reclaim 与 fallback。

见 [test_decode_residency.py](/home/cong/PLLM/tests/test_decode_residency.py:274)。

**反驳范围：** 该测试足以反驳 Round 2 的已知 bucket-pruning 反例，但不是数学证明，也没有穷举生产规模 40 层输入。

### C5：byte hit 被误算成 object hit

**核心 planner 已修复。**

planner 先按每层实际 routed tensor bytes 计算：

\[
B_{miss}=\sum_l m_l\frac{B_l}{512},\qquad
h_{byte}=1-\frac{B_{miss}}{\sum_l22B_l/512}
\]

因此不同层不再等权平均 object hit。[decode_residency.py](/home/cong/PLLM/pllm/decode_residency.py:784)

**剩余边界：** request-local route window 在单层内部仍以 expert count 统计，依赖同层 experts 等尺寸；offline uniform-K 表中的旧 `byte_hit_rate` 也使用平均 expert bytes。异构 evaluator 已明确输出 `object_hit_rate` 与 `miss_bytes`。序列化 header/framing bytes 不进入模型权重 byte hit，而进入实测 transport cost。

### C6：离线 total stall 只支持 uniform K

**接受并已修改。**

`heldout_layer_plan_summary()` 接受完整 `slots_by_layer`，在每个 request-local next window 上逐 token 计算：

\[
S_t(\{K_l\})=\sum_l g(m_{l,t}(K_l))
\]

并报告 p50、p95、p99 和 CVaR95。默认窗口已与 runtime 统一为 256。[benchmark_decode_routes.py](/home/cong/PLLM/scripts/benchmark_decode_routes.py:222)

**剩余边界：** 当前 `g` 是 host shared-MR source curve，不含真实 H2D、kernel rebuild 或 GPU contention。在线 planner仍使用 worst-observed additive surrogate；异构 token-total distribution目前用于离线验收和否证，而非正式概率保证。

### C7：无压力时 planner pending 会错误 yield

**接受并已修改。**

pending result 仍携带 `action=yield` 以表示没有可执行 elastic plan，但 controller 只有在 capacity plan 为 `elastic_resident` 时才执行该快速 yield。`full_resident` 且无容量压力时，pending 不产生动作。[controller.py](/home/cong/PLLM/pllm/controller.py:831)

若 capacity 确实要求释放，foreground fast path 先 Level-0 yield，后台 planner 完成后再选择 elastic 或 hibernate。

CPU synthetic stress 的诚实结果是：

- Pareto 同步求解 p50：`2294.9ms`；
- peak states：`4775`；
- async dispatch：`1.53ms`；
- 后台结果可见时间：`2368.3ms`。

因此我们只主张“planner 不阻塞前台检测”，不主张 planner 本身足够快。

### C8：`>32` objects 缺少稳定 staging 语义

**机制已修改，实机证据仍不足。**

`iter_many()` 每次最多暴露 32 个 memoryviews；dataplane 在 generator 推进、shared slots 被复用前完成 package decode 和 sink write。[expert_store.py](/home/cong/PLLM/pllm/expert_store.py:593)、[expert_dataplane.py](/home/cong/PLLM/pllm/expert_dataplane.py:160)

mock 测试覆盖 batch size 2、总计 5 objects，确认三个 chunk 在 staging 复用前被消费。

**接受剩余质疑：** 尚无真实 RoCE `44` miss、`512` expansion 或 RDMA→GPU chunk pipeline 数据。不能据 mock 声称大批量恢复 no-copy 或高性能。

## 尚不能反驳

### GPU expansion

`512→{K_l}→512` 的 allocator peak、source parse、H2D、kernel rebuild、实际释放量和失败恢复尚未完成。destructive expansion 仍只能称为 implemented mechanism。

### LongBench

新的 256-token request-local route capture、异构 profile replay及在线 PhaseEER first-50 MQA/NQA/TQA 尚无结果。若大多数短 QA 不足两个完整窗口，应报告 N/A，而不是改小窗口或跨请求拼接。

### Blender

B0 无 LLM、B1 full resident、B2 Level-0、B3 PhaseEER、B4 Level-2 的随机交错正式实验尚未运行。当前只有场景、OptiX 和前台 PID 检测链路，不能主张前台达到 90%。

### DGX Spark

现有 host/RDMA 和 RTX 数据不能证明 UMA 上的 `MemAvailable`、PSI、功耗或恢复收益。

## 修订后的 Claim Boundary

**可以主张：**

- stale async plan 由三重 generation 和执行前复验 fail closed；
- PhaseEER 当前限定为一个请求、一个 sequence；
- Pareto planner通过有限随机实例的 exhaustive oracle 对照；
- 在线系统能观察 actual blocking-load debt并形成可执行 fallback；
- 异构 profile 可在 request-local route 上计算 token-total host-source tail；
- chunk iterator消除了 mock 多批路径的强制 `bytes` materialization。

**不可以主张：**

- 已找到可用的 `K_l<512` Pareto 区间；
- `<5x` GPU TPOT 已满足；
- RDMA→GPU expansion 已加速；
- LongBench F1/吞吐不变；
- Blender 达到无后台性能 90%；
- exact KV/Mamba deep resume；
- DGX Spark UMA 收益成立。

## 结论

Round 3 修改使 PhaseEER 从“可能执行错误或过期计划的启发式”推进为具备 request isolation、stale-plan rejection、finite-domain oracle checks 和 actual-debt fallback 的可证伪原型。我们认为这些修改足以回应主要软件正确性问题，但不足以回应论文最关键的经验问题。

在 GPU transition、LongBench 和 Blender 数据完成前，本文仍应维持 Major Revision，而不是把机制完整度当作性能与学术结论。
