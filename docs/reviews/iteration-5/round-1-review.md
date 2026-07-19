# Iteration 5 Round 1 严格审稿

**结论：Reject（当前稿件）**  
**评分：3/10；信心：4/5**  
**创新性：2/5；技术可靠性：2/5；实验完整性：1/5；潜在影响：4/5**

## 总评

本文研究桌面前台任务与后台 120B MoE 推理共享 GPU/UMA 资源的问题，提出 prefill 全驻留、decode-only 弹性专家驻留、KV/Mamba state island、Level-0/2 fallback 和远端 RDMA warm expert image。问题重要，工程量显著，作者也诚实报告了 EER-256 的 paging collapse。

但当前核心方案仍是 `phase-aware policy + traditional expert cache/exact miss + vLLM sleep + RDMA warm store`。在目前证据下属于 A+B+C 系统整合，尚未形成新的驻留算法、恢复算法或可推广的系统原理。中心假设所依赖的真实 GPU route、384/448/480/496/504 在线结果、状态连续性和前台 QoS 均未完成。

## 优点

- 证据分层较严谨，没有把 synthetic predictor、standalone RDMA 或 Level-0 流冻结冒充完整恢复。
- 20,480 个 Marlin runtime expert objects、63,435,912,912 bytes 是有价值的真实数据面资产。
- actual Top-22 为唯一权威路由，miss 时等待正确 expert，方向上避免了模型语义近似。
- 记录并保留 EER-256 `>226x` 延迟下界这一负面结果，说明作者愿意接受可证伪结论。
- persistent-QP stream GET 消除了逐 miss 建连和本地落盘，是必要的数据面进步。

## Critical Issues

1. 核心结论没有实验对象。新方案依赖真实 decode route locality，但 GPU route/LongBench 实验尚未运行，无法判断任何 profile 是否存在非空 Pareto 区间。
2. 在线 hit-rate 有同窗评估偏差。当前窗口既形成 hot experts 又计算 projected hit；fused batch 的 Top-22 还先 `unique()`，不是逐 token access 或实际 byte hit。
3. guardrail 忽略 profile transition。512→K 要复制约 44.3--58.1GiB、重建 40 层 kernel 并回收 allocator；504 只释放约 0.92GiB，却可能复制约 58GiB。必须纳入 `T_shrink`、`T_expand`、剩余 decode horizon 和短请求摊销。
4. prefill 全驻留会与桌面 QoS 冲突。新请求无条件扩回 512 可能违反前台 reserve、OOM，或打断并发 decode。需要 admission、排队、合并或拒绝策略。
5. KV/Mamba state island 只是 allocation fingerprint，不证明内容、block table、sequence、RNG 或 deep exact resume。
6. RDMA miss cost model 与实测不一致。单对象和 22-object batch 非线性，当前模型未覆盖逐层依赖、package decode、Python copy 与 H2D。
7. persistent-QP 仍经过 C++ staging、stdout、Python bytes、CPU tensor、CUDA copy；当前结果不能支持快速 full-prefill restore。

## Major Issues

1. 创新边界不足。ActiveEvict、DALI、Least-Stale、SpecMD、ServerlessLLM 与 BlitzScale 已覆盖动态 expert budget、预测/替换和分层恢复；只把预算来源改成前台应用不够。
2. 统一 K 忽略不同层的路由熵和 miss stall，应在总容量约束下求逐层 `K_l`。
3. expert eviction 不减少 Top-22 FLOPs；前台受 SM、功耗或 UMA bandwidth 限制时还需要 decode duty-cycle/yield。
4. actual Top-22 不等于 exact resume，仍需 greedy token equality 或 logit tolerance。
5. full profile 20,480 objects，但 timed stream 只验证 50/220 个对象；远端 pool 也不含 15.72GiB non-routed weights。
6. 论文和实验报告落后于工作树，artifact 尚不可复现为论文所述系统。

## Minor Issues

- `completion_tokens - 1`、`prompt_start - 8` 缺少与 vLLM token index 对齐的源码或实验验证。
- SSE 首个文本 chunk 不是 engine token boundary，绕过 PLLM proxy 无法正确追踪。
- 每个 MoE 调用的 `.unique().cpu()` 和 mapping publish 需要测 instrumentation overhead。
- GB/GiB、checkpoint、runtime expert image 与 remote pool capacity 口径需要统一。
- 单模型、单 vLLM 版本只能作为原型限制，不能据此宣称一般性 phase boundary。

## 必须新增的算法定义

把固定阈值改为 per-layer、horizon-aware、risk-bounded residency algorithm：

$$
\min_{\{K_l,S_l\},r,z}
Q_{1-\delta}\left[\sum_l g_l(m_{l,t})\right]
+\lambda T_{transition}+\mu P_{foreground}.
$$

约束至少包括 resident bytes、release deadline、next-window TPOT、I/O/UMA bandwidth 和 decode duty cycle。预测只能使用过去窗口，评估下一窗口；没有可行解时显式选择 yield/hibernate。只有剩余 decode 时间足以摊销 shrink、miss 和下一次 prefill expansion 时才允许收缩。

## 必须新增的对照与消融

- native vLLM、full-512 EER、full-512 EER+trace。
- LRU、LFU、window-LFU、Least-Stale、static popularity、offline Belady/oracle。
- 统一 K 对比 per-layer `K_l`，以及无 horizon、无风险上界、无 fallback 消融。
- local NVMe、new-QP RDMA、persistent-QP RDMA、batch GET、完整 RDMA→GPU slot。
- no-LLM、full resident、Level-0、Level-2 与 decode profiles 的同一 Blender trace。
- Attention KV、Mamba conv/SSM、RNG/ledger 状态消融。
- 短 QA、至少 512-token 代码生成、并发 1/2 和跨领域 route drift。

## 可证伪验收条件

1. 至少一个 `K<512` profile 同时满足真实容量释放、前台吞吐不低于无后台 90%、LLM TPOT 小于 full resident 5x，且恢复快于 Level-2；否则 elastic 假设失败。
2. profile 在 held-out next-window 达到声明的 p95 miss/TPOT SLO；同窗 hit 不计。
3. shrink+expand 全部计入请求 wall time；短 LongBench 无法摊销时必须拒绝 resize。
4. 连续 100 次 token-boundary resize 中 actual Top-22、状态内容与 greedy token 序列和 full baseline 一致。
5. RDMA 报告随机跨 40 层对象的 p50/p95/p99，以及 RDMA、pipe/decode、H2D 分解。
6. DGX Spark 报告 `MemAvailable`、PSI、功耗和 Blender throughput；独显 VRAM 不替代 UMA 结论。

## 最终判断

当前实现是有价值且工程基础扎实的 hackathon prototype，但顶级系统/MLSys 标准下仍是 A+B+C。可能脱离该评价的贡献应是：面向突发前台容量 SLO，基于未来窗口风险上界和剩余 decode horizon，联合求解逐层专家容量、实际 miss batch、decode duty cycle 与 hibernation boundary，并在 DGX Spark UMA 上证明新的 Pareto 区间。在真实 route 和在线 GPU 数据出现前，应称为待验证算法设计。
