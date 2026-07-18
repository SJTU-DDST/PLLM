# Iteration 4 / Round 1 / Reviewer

**总体判断：5.5/10，Weak Reject（顶会系统论文）；7.5/10（Hackathon artifact）。** 本轮最大的进步是把 71 远端内存池和 Blender 前台链路写进了证据表，并主动披露 v1/v2、wall/data-phase 与 preview/full experiment 的差异。但论文仍把三种成熟度完全不同的机制放在同一个标题和摘要中，读者很难判断核心结果究竟是什么。

## 可信贡献

1. 真实 120B NVFP4、128-slot Marlin、Level 1/2 和 60GiB admission 是有价值的系统证据。
2. 15.859GB 跨机 profile 不是 `ib_write_bw` 替代品，而是真实 `.pllmex` 对象；71 侧无磁盘是合理的数据面选择。
3. 作者明确把 49.85/31.62Gb/s wall throughput 与 90.19Gb/s worker-phase diagnostic 分开，没有将后者直接写成端到端吞吐。
4. Blender 工程、OptiX 枚举和 foreground PID 已可重复，但论文没有把 preview 冒充 QoS 实验。

## 主要问题

**I4.R1.1 Critical：标题中的“事务式休眠”仍没有对应实现。** 当前 live-state carrier 只有 byte transaction，`serializer_attached=false`，Mamba/KV/RNG exact resume 未接入。标题和摘要会让读者以为事务恢复是已验证贡献。应删除标题中的已完成语气，或把论文明确改成 artifact/design report。

**I4.R1.2 Critical：论文的主结果不清楚。** EER 首请求 63.94s，Blender QoS 未跑，DGX Spark 未跑，RDMA 尚未直接进入 GPU slot。当前唯一闭环结果是 Level 2 释放后 60GiB allocation 从 OOM 变成功。论文需要建立 claim ladder：哪些是 result，哪些是 mechanism artifact，哪些只是 hypothesis。

**I4.R1.3 Major：RDMA 指标仍存在归因歧义。** PUT 输入可能来自 page cache，GET wall 包含本地文件写，MR/QP setup 是否在每个指标内也不一致。`sum(bytes)/max(worker phase)` 只有在 worker phase 高度重叠时才近似 aggregate throughput。论文需要给出统一时间分解：setup、source、wire、sink、validation，并固定 wall metric 为主指标。

**I4.R1.4 Major：phase-separated epoch 是控制假设，不是新一致性算法。** 删除 checksum 和 read-after-read 可以提高 benchmark，但论文必须说明谁建立 epoch、并发 GET 如何被禁止、71 reboot 如何使 generation 失效。否则“commit header”容易被误解为完整事务语义。

**I4.R1.5 Major：前台感知仍没有端到端因果证据。** GNOME focus 可用不等于 PLLM 提高 Blender 性能。至少需要 full-resident、PLLM、no-background 三组渲染，报告检测延迟、hibernate latency、samples/s、GPU memory 和恢复时间。

## 次要问题

- `61 passed` 应给出测试选择规则；排除失败测试的原因不能只写“旧环境假设”。
- v1 跨机结果与 v2 本机结果应放在不同表格，避免暗示 v2 已达到跨机线速。
- 当前系统是离散 GPU 服务器验证，标题中的 unified-memory AI PC 仍依赖未来 DGX Spark 实验。

## 可证伪条件

若 Blender 三组对照显示前台吞吐没有改善，或直接 GPU-slot reload 后 wall latency仍由本地 sink/Marlin 重建主导，则 RDMA pool 不是系统瓶颈的有效解法。若 Mamba/KV/RNG 无法保持 greedy token equality，则“事务式休眠”主张必须删除。

## 下一轮必须修改

1. 重写标题、摘要和贡献，建立 validated result / artifact / hypothesis 三层 claim ladder。
2. 给出统一的数据移动时间分解与 RDMA contract。
3. 将 Blender 和 exact resume 明确列为待运行 protocol，不再和已验证结果并列。

## 建议结论

**Weak Reject as a systems paper; strong engineering direction for a hackathon.**
