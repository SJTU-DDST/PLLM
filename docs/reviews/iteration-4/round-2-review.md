# Iteration 4 / Round 2 / Reviewer

**总体判断：6.5/10，Borderline（研究原型）；8/10（Hackathon）。** 标题降级和 claim ladder 显著改善了论文可信度。作者现在明确承认“容量 admission 与 profile 搬运”是闭环结果，exact resume、Blender QoS 和 DGX Spark 是假设。论文不再靠措辞把未实现部分伪装成贡献。

## 已解决的问题

1. 事务式休眠被标为未完成协议，Level 2 原生 drop/reload 不再冒充 exact state commit。
2. RDMA 使用统一的 `T_wall` 分解，wall throughput 被设为主指标。
3. phase-separated epoch 被正确表述为外部调度 contract，而非新一致性算法。
4. v1 跨机结果和 v2 本机 smoke test 已明确分离。

## 仍然存在的主要问题

**I4.R2.1 Critical：论文最有学术潜力的 H1--H4 仍没有真实 router trace。** 当前 predictor 的 domain-shift 结果来自 synthetic trace，EER 真实运行依赖 blocking miss，而首请求读取约 58GB、耗时 63.94s。没有真实 route locality，无法证明中间 elastic 区间存在。claim ladder 解决了可信度，但没有解决论文贡献强度。

**I4.R2.2 Major：R2 的“exact-route”不等于端到端模型等价。** 代码以 actual Top-22 为权威并 fail closed，这是正确的不变量；但 independent baseline 本身不 deterministic，尚无逐层 expert output 或 greedy token baseline。建议把 R2 改成“route-preserving functional execution”，避免读者把它理解为输出等价证明。

**I4.R2.3 Major：R3 与 Level 2 restore 不可直接比较。** RDMA 搬运的是 15.859GB 的 128-slot profile，Level 2 local restore 重载约 74.8GiB 且包含 Marlin/runtime rebuild。论文需要给出 byte-normalized 和 critical-path 表，并明确 remote profile 尚未进入 GPU slots。否则读者很容易用 2.545s 对 41.9s 得出错误 speedup。

**I4.R2.4 Major：前台机制只有 sensing，没有 intervention result。** A1 验证了 Blender focus PID，但没有验证 focus→policy→sleep 的 action latency。即使暂时不跑完整 render，也应把可复现实验协议、触发阈值和预期状态迁移放入 artifact section，使下一次运行可以直接填表。

**I4.R2.5 Major：学术核心需要更紧的形式化。** 当前“当 paging 不可行时 hibernate”接近直觉。论文应明确 feasibility frontier：给定 resident bytes、miss bytes/token、token rate、foreground deadline 和 I/O slack，哪些动作可行；planner 的创新候选是跨越该 frontier，而不是 A+B+C 的组件集合。

## 次要问题

- 测试例外应在工程报告中说明，四页论文只需给 artifact commit 和测试总览。
- 应增加 artifact-to-claim mapping，列出每个 JSON、命令和证据类别。
- “在线校准 prediction set”仍比当前 synthetic 实现强，摘要应改成目标机制或 prototype implementation。

## 下一轮必须修改

1. 把 phase boundary 写成可行域与不可行域，不把直觉阈值称为算法。
2. 增加 current hypothesis-status table 和 byte-normalized critical path。
3. 增加 Blender 三基线实验协议与 artifact-to-claim mapping。
4. 把 R2 改为 route-preserving functional result，继续保留 exact-output 未验证。

## 可证伪条件

如果真实 trace 下 `E[B_miss+B_false]` 使任何实用 token rate 都越过 I/O frontier，则 EER 应从论文主路径移除，系统退化为 foreground-aware hibernation。若 remote profile 进入 GPU 后重建/同步时间远大于网络时间，则 RDMA 应被表述为 transport optimization，而不是 restore optimization。

## 建议结论

**Borderline as a prototype paper; strong accept for an evidence-disciplined hackathon submission.**
