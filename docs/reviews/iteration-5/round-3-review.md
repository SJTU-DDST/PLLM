# Iteration 5 Round 3 Review

## 总体评价

**评分：4/10，Reject / Major Revision**  
**信心：4/5**

| 维度 | 评分 |
| --- | ---: |
| 创新性 | 5/10 |
| 算法正确性 | 5/10 |
| 系统完整性 | 4/10 |
| 实验充分性 | 2/10 |
| 论述与证据边界 | 6/10 |

Round 2 的两个核心错误已经实质修复：Pareto frontier 不再每个 bucket 只保留单状态，`current→target→full` 也纳入了 destructive expansion 和未来 full-prefill 恢复成本。direct shared host-MR 的跨机数据可信，但只证明到 host memory 的 source path。

当前仍不能接收。异步 planner 可能跨请求复用旧决策，request-local 假设在并发后会失效，论文所称 online miss-debt fallback 尚未实现；最关键的 GPU route、在线 resize、LongBench decode-only 和 Blender QoS 证据仍为空。

## Round 2 问题核验

| 项目 | Round 3 判断 |
| --- | --- |
| Pareto 支配条件 | 对当前加性目标和单调约束基本正确 |
| exhaustive test | 现有测试过弱，不能证明一般正确性 |
| `current→target→full` | 公式层面已修复 |
| `min_tokens` + token IDs | vLLM 0.25.1 streaming 路径基本正确 |
| request-local/single decode | 未闭环 |
| 逐 token total-stall | uniform-K 离线统计正确，逐层计划未覆盖 |
| online worst-observed surrogate | 边界写得更诚实，但没有在线违约检测 |
| async fast yield/fallback | 接线完成，但存在 stale-result 和误 yield |
| `>32` chunk iterator | 代码路径存在，真实 shared-MR 验证不足 |
| GPU/LongBench/Blender | 仍未完成 |

## Critical Issues

### C1. 异步 planner 会跨请求执行过期结果

[expert_control.py](/home/cong/PLLM/pllm/expert_control.py:277) 的 key 只包含 completed-window 数、32-token horizon bucket、slots 和容量 bucket，不包含 request generation、route profile 内容或 profile generation。[同步缓存](/home/cong/PLLM/pllm/expert_control.py:199) 有相同问题。

我构造了两个具有相同 key、不同 profile 的请求：request A 完成后，request B 立即得到了带有 A 标记的缓存结果。Round 2 rebuttal 承诺的“request/profile generation”并未实现。

必须增加单调 `request_generation`、`route_generation`、capacity generation，并在 future 完成和执行 resize 前再次核验。Horizon 不应按 32 token 粗粒度复用，至少必须重新验证当前计划在精确 `H` 下仍满足 SLO。

### C2. request-local 与 single-decode 不变量仍会被并发破坏

[DecodeRouteWindow.set_phase()](/home/cong/PLLM/pllm/decode_residency.py:125) 重置 decode history，但没有清空 `_prefill_tail` 和 `_prefill_counts`，短 prompt 会继承上一请求的 prefill ranking。

[controller.py](/home/cong/PLLM/pllm/controller.py:339) 只在系统原先完全 idle 时执行 reset。第二个请求与现有 decode 重叠时不会重置或永久 invalidate predictor；并发结束、`decode_requests` 回到 1 后，混合历史又可能被用于 elastic plan。此外，“一个 request”不等于“一个 decode sequence”，`n>1` 仍会绕过 single-decode gate。

需要在任何 overlap、`n>1`、绕过代理或 phase 不连续时将当前 generation 标记为不可学习，直到所有请求结束并由下一次干净 prefill 重新开始。

### C3. `<5x` 风险门没有运行时违约闭环

当前在线 surrogate 是各层 `g(max_observed_miss_l)` 的求和。这能上界历史窗口中的 host-source cost，但不是未来分布界，也不包含 H2D、Marlin rebuild 或 GPU stall。

更严重的是，论文 [限制部分](/home/cong/PLLM/paper/HiberFlow-ACM六页稿.md:414) 称 domain shift 由 online miss debt 和 fallback 处理，但代码中没有 miss-debt accumulator、违约阈值或由实际 miss stall 触发的 yield/hibernate。32-object p99 已达 250.9ms，planner 却仍使用 p95 `g(m)`。

必须在线记录逐 token actual miss、load wall time、累计 debt，并在超过计划预算时立即停止继续 paging。否则 `<5x` 只能称为历史输入上的启发式筛选。

### C4. 中心论文结论仍没有真实证据

`results/decode_residency/` 仍为空。论文也明确承认 GPU route、逐层 resize、H2D/Marlin TPOT、正式 LongBench 和 Blender 三组实验均 pending。

这不是可以用软件回归替代的部分。论文的核心命题是存在优于 full residency 和 hibernate 的 elastic Pareto 区间；当前唯一真实 paging 结果仍是 EER-256 的超过 226 倍退化。

此外，MQA/NQA/TQA 通常只生成很短答案，而当前算法至少需要两个完整 256-token decode window。必须报告“有多少请求实际进入 elastic”，并增加长代码生成或长文本生成 workload，不能仅用短答案 QA 宣称验证 decode planner。

## Major Issues

### M1. Pareto 实现合理，但仓库 exhaustive test 不充分

[支配条件](/home/cong/PLLM/pllm/decode_residency.py:850) 对当前约束是安全的：reclaim 越大越好，其余最终使用的成本维度越小越好。我额外进行了 1,000 个包含异构 current slots、miss、deadline、I/O 和 transition 的小实例穷举，未发现与 brute force 不一致。

但仓库的 [oracle test](/home/cong/PLLM/tests/test_decode_residency.py:270) 只有 25 个三层、零 miss、全 full-current 实例，只比较 reclaim，不比较完整 `{K_l}`、可行集合和 fallback。应扩展至 `L≤6`、每层最多四个 option、至少 10,000 个固定种子，并覆盖所有约束边界。

当前 bucket 内保存完整 frontier，且没有 frontier cap，因此 bucket 本身并未造成近似；论文“bucket 化仍是近似”的表述不准确。真正的问题是最坏复杂度可能指数增长，现有 40 层 artifact 已达到 p50 2.295s。

### M2. “byte-hit”约束实际按 object miss 计算

planner 虽读取 `byte_hit_rate_lower_bound`，最终却用 `1-sum(mean_misses)/(L×TopK)` 重新计算 hit rate。不同层 expert bytes 不一致时，这不是 byte hit。应使用 `miss_bytes / total_access_bytes`，或明确证明 Nemotron 各层对象等长并将指标改名为 object hit。

### M3. 离线 total-stall 只支持 uniform K

[heldout_next_window_summary()](/home/cong/PLLM/scripts/benchmark_decode_routes.py:91) 正确地对每个 held-out token 计算 `sum_l g(m_l,t)`，没有错误相加逐层 quantile；但它只接受统一 `slots`，不能评估论文核心的逐层 `{K_l}`。

需要输入 planner 产生的 `slots_by_layer`，逐 transition 重放实际计划，同时报告 p50/p95/p99、CVaR、surrogate/actual 比值和违约率。

### M4. async fast yield 会在无容量压力时误暂停

pending plan 固定返回 `yield`，[controller fallback](/home/cong/PLLM/pllm/controller.py:783) 不检查 capacity action。因此即使当前为 `full_resident`、无需释放资源，首次异步求解也可能触发 Level 0。

fast yield 必须只由真实 foreground/capacity pressure 触发。后台 planner pending 本身不是暂停 vLLM 的理由。

### M5. `>32` iterator 逻辑成立，但证据不足

[RDMAPoolExpertStore.iter_many()](/home/cong/PLLM/pllm/expert_store.py:900) 和 [TieredExpertSource.iter_many()](/home/cong/PLLM/pllm/expert_store.py:1030) 能在完整 remote tier 上逐批消费，当前同步 H2D 也能在 shared MR 覆盖前完成读取。

但现有测试只用 batch size 2 的内存 mock；跨机数据最大 batch 仍为 32。必须增加 64/88-object shared-MR 测试，验证跨 chunk 顺序、内容 hash、staging 覆盖和混合 tier fallback。RDMA CMake 可以编译，但当前 `ctest` 显示没有注册测试。

### M6. 论文仍有几处超出证据的措辞

[表 R5](/home/cong/PLLM/paper/HiberFlow-ACM六页稿.md:34) 将 GPU-row resize 列为“Validated mechanism”，但新路径没有真实 GPU 验收，应改为“Implemented / GPU pending”。

[第 301 行](/home/cong/PLLM/paper/HiberFlow-ACM六页稿.md:301) 的 `>32` 消费保证目前来自代码推理和 mock，不是 direct shared-MR 实测。第 414 行的 online miss-debt fallback 也应改成“尚未实现”。

## 创新性判断

当前仍有明显的 A+B+C 风险。Pareto DP、本地窗口预测、expert paging、persistent QP 和 phase guard 单独都不是新算法。

潜在可发表贡献是一个可证伪的系统结论：在前台容量 deadline 下，prefill/full、decode/elastic 和 hibernate 之间存在由 route locality、transition horizon 和实际 source tail 共同决定的 phase boundary。但只有真实证明非空 elastic Pareto 区间、且优于 full/yield/hibernate 后，这才会从工程组合上升为 MLSys 贡献。

## 接受条件

1. 修复异步 generation 和 request-local 不变量，并加入跨请求 stale-result、并发、`n>1` 和短 prompt 测试。
2. 建立在线 actual-stall/debt fallback，验证违约后能在 deadline 内 yield 或 hibernate。
3. 完成真实 Nemotron route、逐层 `{K_l}` resize/restore、greedy equality、TPOT 和释放量实验。
4. 使用 plan-specific `{K_l}` 计算逐 token total-stall，并与 online surrogate 做覆盖率对照。
5. 完成 LongBench 加长生成 workload，以及 full/all-phase/decode-only/uniform-K/per-layer-K 消融。
6. 完成 Blender B0/B1/B2 随机交错实验，证明前台性能而不只是显存释放。
7. 完成 host→GPU/UMA slot 和 `>32` shared-MR 实验；host endpoint 数据不得外推为 GPU speedup。
8. 收紧论文中的 validated、fallback、byte-hit 和 bucket-optimality表述。

## 验证记录

当前 Python 回归约 108 项全部通过；RDMA 三个目标编译成功，但无 `ctest`。独立的 1,000 例随机小规模穷举未发现 Pareto DP 与当前数学模型不一致；异步跨请求缓存则已得到确定性反例。
