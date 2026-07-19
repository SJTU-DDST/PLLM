# Iteration 5 Round 2 Rebuttal

**审稿结论：Reject / Major Revision，4/10**

我们接受该结论。Round 1 后的方法已经从同窗、统一 K 的启发式推进到 request-local held-out、逐层 horizon planner，但审稿人指出当前 bucket DP、tail estimator 和控制闭环仍不足以支撑论文核心结论。本文不报告尚未完成的 GPU 结果，也不把计划中的修改写成现有贡献。

## 已完成修正

两项修改已进入当前实现：

1. **保守且可靠的 decode horizon。**  
   现在使用
   \[
   H_{lb}=\max(0,\texttt{min\_tokens}-N_{\text{exact generated token ids}})
   \]
   PLLM 强制向 vLLM 请求 `return_token_ids`，按 delta token IDs 递减，并在返回客户端前移除用户未请求的字段。缺少 token IDs、退化到文本 chunk 或未提供 `min_tokens` 时，立即令 `H=0`，禁止 elastic resize。[api.py](/home/cong/PLLM/pllm/api.py:247)、[controller.py](/home/cong/PLLM/pllm/controller.py:367)

2. **请求边界隔离。**  
   新 prefill 会清空 prediction counts、recency、validation history 和未封口窗口；离线分析分别在每个请求内部形成 previous→next transitions，不再连接两个短请求。[decode_residency.py](/home/cong/PLLM/pllm/decode_residency.py:119)、[benchmark_decode_routes.py](/home/cong/PLLM/scripts/benchmark_decode_routes.py:91)

其余意见均接受为待完成修改。

## Critical Issues

### C1：单状态 bucket DP 不是约束优化

**接受。** 当前实现最多区分同一 bucket 的 under-target 与 target-satisfied 状态，仍可能让低 objective、高 miss bytes 的状态淘汰最终可行状态。[decode_residency.py](/home/cong/PLLM/pllm/decode_residency.py:598)

将改为每个 reclaim bucket 保存非支配 Pareto frontier。状态至少包含：

\[
z=(R,T_{now},T_{restore},B_{miss},\mathbf S,parent)
\]

其中 `S` 是 held-out token 的累计 stall vector。只有当一个状态在 reclaim 不低、迁移时间不高、miss bytes 不高且每个 token stall 都不高时，才允许支配另一个状态。profile 路径改用 parent pointer 回溯，不再复制 40 层 records tuple。

新增 exhaustive property test：对 `L≤6`、每层 `≤4` 个 profile 枚举全部组合，验证 frontier planner 与 brute force 的可行集合、最优目标和 fallback 完全一致。

### C2：transition 和 release deadline 不完整

**接受。** 当前 `immediate_seconds` 只统计 shrink，漏掉从异构 profile 即时扩容；当 `K_l=current_l<512` 时也漏计未来 full-prefill rebuild。

修改为显式转移矩阵：

\[
T_{now}=\sum_l T_l(K_l^{cur}\rightarrow K_l),\qquad
T_{prefill}=\sum_l T_l(K_l\rightarrow512)
\]

`release_deadline` 约束完整 `T_now`；horizon 摊销同时包含 `T_now+T_prefill`。任何 `source != target` 都计一次 allocator/kernel rebuild。GPU 实验后用各层实测 transition matrix 替换当前带宽配置估计。

### C3：`sum_l g(p95_l)` 不是 token-total p95

**接受。** 逐层 p95 相加破坏跨层相关性，既可能过保守，也可能低估同步长尾。

route evaluator 将为每个 request-local held-out token 保留：

\[
S_t(\{K_l\})=\sum_l g_l(m_{l,t}(K_l))
\]

并独立检查总 miss bytes/I/O budget。planner 直接约束 `S_t` 的 empirical p50、p95、p99 和 CVaR95；默认使用 CVaR95 排序、p99 作为 hard tail guard。Pareto state 的 stall vector 按 token 对齐累加，因此不会再由 per-layer quantile 推导 total quantile。GPU `g_l` 未完成前，只能标注为 host-side estimate。

### C4：`max_tokens - chunk` 不能作为 horizon

**接受并已修复。** `max_tokens` 已从 horizon 路径移除；文本 chunk 不再被视为 token。没有显式 `min_tokens` 或精确 delta token IDs 时，planner 得到 `H=0` 并拒绝 resize。

边界是：当前可靠 horizon 只支持经 PLLM proxy 的流式 vLLM 请求。非流式请求和绕过代理的请求不启用 elastic planning。

### C5：planner fallback 未闭环

**接受。** 当前 `_maybe_auto_resize()` 只执行 `decode_elastic`，planner 返回的 `yield` 不会驱动 controller；外层 policy 的 yield 是另一条独立决策，不能算闭环。

将增加统一 dispatcher：

| Planner action | Controller action |
| --- | --- |
| `observe/full` | 保持 full resident |
| `decode_elastic` | Level-0 quiesce → resize → wake |
| `yield` | Level-0 `mode=keep` |
| `hibernate` | Level-2；在 exact serializer 完成前标为 abort/replay |

异步结果携带 request/profile generation，过期结果不得执行。测试必须验证 planner yield 实际调用 Level-0，而非只出现在 API JSON 中。

## Major Issues

### M1：planner 约 0.21s

**接受。** 该延迟不适合 250ms monitor 的同步关键路径。除 parent pointer 外，planner 将移到异步 worker，并按 route generation、capacity envelope 和 horizon 缓存结果。前台 focus 事件先执行独立的快速 Level-0 yield；只有完成且未过期的 planner 结果才能转入 elastic 或 hibernate。修改前不声称 planner 满足 `<500ms` 控制 SLO。

### M2：不得跨请求拼窗口

**顺序请求已修复；并发仍保留限制。** 在线 reset 和离线 request-local transitions 已完成。但 fused concurrent decode 目前没有 request ID 标注，仍可能混合两个并发请求。因此 Round 2 中 route learning 只在 `decode_requests==1` 时启用；并发实验先保持 full/yield。不得由单请求结果外推并发 locality。

### M3：offline 64、runtime 256 不一致

**接受。** offline 默认将改为 256，与 runtime 对齐。64/128 仅作为窗口敏感性消融。若 first-50 QA 中单请求不足两个 256-token 窗口，应报告 `insufficient evidence`，不能跨请求补足样本。

### M4：503 不是自动 FIFO

**接受。** 当前只是 server-side replay record 加手工重放入口。将增加单消费者 FIFO：前台 reserve 解除后自动恢复 full prefill profile并执行 queued jobs。

原 HTTP/SSE 连接已返回 503，不能宣称透明续传。OpenAI API 用户获得 replay ID 并轮询结果；只有 PLLM 原生客户端可用持久 job stream 自动重新连接。

### M5：超过 32 objects 会复制 shared staging

**接受。** 当前为避免 mmap slot 被下一批覆盖，会把 `>32` 的 memoryviews 转成 `bytes`。[expert_store.py](/home/cong/PLLM/pllm/expert_store.py:571)

将实现 `iter_many()`：每次最多返回 32 个 views，dataplane 在 staging 复用前完成 decode/H2D，再请求下一批。协议边界保持“单个 RDMA batch ≤32”，但任意长度逻辑请求可以 chunk-stream；不得再宣称大批次 zero-copy list。

### M6：GPU expansion 尚未验证

**接受。** destructive expansion、H2D、kernel rebuild、实际释放量、OOM 峰值和状态连续性均为 GPU pending。当前仅能主张代码路径和 CPU/mock 不变量，不能主张恢复加速或 `<5x` TPOT。

## 修订后的主张

Round 2 论文只保留以下核心表述：

> PhaseEER 是一个待验证的 request-local、horizon-aware、per-layer residency algorithm。它以 empirical token-total tail、完整 profile transition 和前台容量 SLO决定 elastic 是否可行；不可行时执行可验证的 yield/hibernate fallback。

不再主张 bucket DP 是严格约束优化，不再使用 per-layer p95 代表 total p95，不再把 503 称为自动恢复，也不把 host shared-MR 吞吐外推为 GPU reload 性能。

## Round 3 验收条件

| 验收项 | 可证伪条件 |
| --- | --- |
| Planner 正确性 | 小规模 exhaustive enumeration 与 Pareto planner 完全一致 |
| Planner 开销 | foreground 快路径不等待 planner；异步 p95 与状态数同时报告 |
| Route 证据 | first-50 MQA/NQA/TQA，256-token request-local windows；不足即 N/A |
| Tail SLO | 报告 token-total p50/p95/p99/CVaR95，不再相加 per-layer quantiles |
| Transition | 实测每层 `current→target→512` wall time、bytes、rebuild 和释放量 |
| Fallback | 无可行 profile 时实际触发 Level-0/2，并记录动作延迟 |
| RDMA→GPU | 分解 RDMA、staging decode、H2D、mapping publish 和 resumed TPOT |
| 前台 QoS | Blender 达到无后台吞吐 90%，同时 LLM TPOT `<5x`；否则 H1 失败 |

**结论：** Round 2 意见揭示的不是措辞问题，而是优化器正确性、风险统计和执行闭环问题。我们接受这些问题。在 Pareto property test、token-total tail、fallback dispatcher 和 GPU transition 数据完成前，PhaseEER 仍是有实现基础但核心假设未成立的原型。
