# Iteration 4 / Round 3 / Reviewer

**总体判断：6/10，Weak Reject（顶会系统论文）；8.5/10，Strong Accept（Hackathon 研究原型）。** 第三轮已解决可信度问题：所有已验证、机制 artifact 和研究假设均有明确边界，不再从 RDMA bytes 推导模型恢复，也不从 focus detection 推导 Blender QoS。但论文的两个算法性主张，elastic feasibility frontier 和 hybrid-state exact resume，仍缺少决定性数据或实现。

## 本轮已解决

1. 论文给出 `F_E(K,P,r)` 与 `F_H` 的容量、带宽、deadline 和 compute 可行域，把“换页不行就休眠”改成可证伪的 phase boundary。
2. H1--H5 均显式标注当前状态。尤其 H1、H3、H5 被标为尚未建立，H2 仅有 synthetic 证据。
3. R2 改为 route-preserving functional execution，不再暗示 greedy-output equivalence。
4. Level 2 release/restore、RDMA v1 profile 和 v2 loopback 按字节对象、wall time 与 endpoint 对齐。2.545s 不再被当作 74.8GiB 模型的恢复时间。
5. Blender B0/B1/B2 的场景、参数、重复次数和指标已固定，artifact-to-claim mapping 可让评委追溯原始 JSON。

## 仍然阻止顶会接收的问题

**I4.R3.1 Critical：可行域是正确形式，但还不是已验证算法。** 当前只观测到一个 `F_H=true` 的 Level 2 admission 点。没有真实 route 的 `E[B_miss+B_false]`，也没有 32/64/128/256 slots 的 frontier sweep，无法证明 `F_E` 存在非空、有用的区间。

**I4.R3.2 Critical：HiberCache 仍是 byte carrier，不是 NemotronH 恢复算法。** Mamba recurrent state、Attention KV、RNG/sampler 与 token ledger 没有连入 vLLM executor。故障注入和 uninterrupted greedy equality 均不可运行。这一缺口直接否定了标题中任何“exact hibernation”式强主张，但当前标题已避免该问题。

**I4.R3.3 Major：缺少前台干预的因果数据。** focus bridge 是完整 artifact，但论文的问题是 foreground QoS，不是 focus detection。B0/B1/B2 必须成为下一个实验；否则 PLLM 只是能被前台事件触发的 model sleep controller。

**I4.R3.4 Major：RDMA 还未进入 model restore 关键路径。** 跨机 v1 证明 remote profile carrier 可行，但 GET 落在本地文件，再由 Python/CUDA loader 入 slot。在相同 15.859GB profile 上比较 local NVMe 和 RDMA-to-slot 之前，它是 transport contribution，不是 restore-speed contribution。

**I4.R3.5 Major：评估还停留在单机、单模型、单轮测量。** DGX Spark UMA 没有实测，Level 2 数据没有反复次数、p50/p95 和功耗。这不影响 Hackathon demo 的真实性，但不足以支撑顶会的普适系统结论。

**I4.R3.6 Minor：当前 Markdown 是研究终稿，不是已排版的 ACM 四页稿。** 证据表、完整实验协议和 artifact mapping 使正文明显超过四页排版预算。正式投稿应保留问题、frontier、三个验证结果和关键限制，将协议与 artifact 表移入附录/产品报告。

## 终局优先级

1. 立即跑 Blender B0/B1/B2，先证明前台问题、介入和收益的因果链。
2. 从真实 Nemotron 请求收集 route trace，完成 frontier sweep。H1 失败时应删除 EER 主线，而不是调整口径。
3. 让 RDMA GET 直接填充 pinned staging/slot loader，并与同 bytes 的 NVMe 基线对比。
4. 完成最小 NemotronH serializer：先保证 greedy single-request token equality，再扩展并发与 fault injection。
5. 在 DGX Spark 上重复核心表格，主指标改为 `MemAvailable`、UMA bandwidth 和整机功耗。

## 最终裁决

作为顶会系统论文，**Weak Reject**：研究问题和可证伪边界已经站稳，但最具创新性的两个机制仍缺关键实验。作为 Hackathon 项目，**Strong Accept**：真实 120B 模型、快速 capacity admission、跨机 profile pool、可用 UI 和 Blender artifact 形成了可演示的完整工程骨架，而且论文对证据边界的表述值得信任。下一步不应继续纸面迭代，而应运行上述五类实验。
