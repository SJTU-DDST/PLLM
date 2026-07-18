# PLLM HiberFlow-EER 项目报告

## 1. 项目定位

PLLM 是面向 DGX Spark 和 NVIDIA 桌面 AI 工作站的前台感知 vLLM 资源运行时。它解决的不是“怎样让大模型独占设备跑得更快”，而是：

> 当用户突然启动游戏、Blender、视频编码或其他高负载任务时，后台 120B MoE 模型怎样迅速让出容量、带宽、算力和功耗，并在前台结束后低成本恢复同一请求？

第一版只控制 vLLM，不暂停训练任务、未知 CUDA 进程或未开放 Sleep API 的外部服务。测试模型固定为只读共享目录中的 `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`，不复制、不下载。

最新方案由三档连续动作构成：

1. **Full Resident**：前台空闲时保持完整模型和正常 token rate。
2. **Elastic Expert Residency**：中等压力下保留模型原始 Top-22 路由，只让部分 routed experts 驻留在 UMA，其余位于 NVMe 或远端主机。
3. **Transactional Hibernation**：压力过大或 expert miss 流量不可承受时，在 token 边界提交 live state，丢弃不可变权重并深度休眠。

项目暂不研究动态 Top-k。任何模式都不减少或替换 router 选择的 22 个 experts；预测错误只允许产生等待和额外 I/O，不能改变模型输出。

## 2. 用户演示闭环

正式 Demo 使用一条可观察的因果链：

1. Nemotron 正在通过 vLLM 流式生成 CUDA/DOCA 代码。
2. 用户启动 Blender 或游戏，GNOME focus、进程 exec 和 NVML activity 触发前台资源需求。
3. 悬浮窗显示需要释放的物理容量、允许的后台算力和决策 deadline。
4. PLLM 先降低 decode duty cycle，再把 expert cache 从 full 收缩到 128/64 slots per layer；UI 展示实际 resident bytes、byte hit、prefetch 和 blocking miss。
5. 若预计 SSD/RDMA 流量会破坏前台 QoS，系统停止换页，在 token 边界 commit 并进入 Level-2 hibernation。
6. 前台吞吐、`MemAvailable`、PSI、功耗与 I/O 实时显示，并与无后台模型基线比较。
7. 前台结束后，系统渐进 warm experts 或从 NVMe/ConnectX-7 恢复完整权重，原请求继续输出。

评委模式必须区分 `LIVE`、`MOCK` 和 `HISTORICAL REPLAY`。没有运行过的真实模型数据不能伪装成实时曲线。

## 3. 系统架构

```text
GNOME focus/exec ─┐
NVML/codec/power ─┼─> Foreground-QoS Agent ─> resource envelope R/D/B/C/H
PSI/MemAvailable ─┤                                  |
UPower/profile ───┘                                  v
                                      Full / Elastic / Yield / Hibernate
                                               |              |
                                               v              v
                                  Exact Expert Residency   Token Transaction
                                  slot cache + predictor   KV + Mamba + ledger
                                      |          |              |
                                      v          v              v
                                  local NVMe  CX-7 host-stage  SparkLoad
                                               |
                                               v
                             Flask API + SQLite + Vue + PySide6
```

状态机为：

```text
FULL_RESIDENT <-> ELASTIC_RESIDENT -> YIELDING -> COMMITTING
                                                   |
                                                   v
                                              HIBERNATED
                                                   |
                                                   v
                                               RESTORING
```

## 4. 核心模块

### 4.1 Foreground-QoS Agent

Agent 每 250ms 融合：

- GNOME Shell D-Bus 的焦点 PID、app ID 和进程启动事件；
- NVML 的 GPU、进程级 SM、内存、功耗、NVENC/NVDEC；
- Linux `MemAvailable`、swap、CPU load 和 memory PSI；
- UPower 与 `powerprofilesctl`；
- 历史前台持续时间、误触发和恢复成本。

输出不是一个简单阈值，而是前台资源包络：容量 `R`、释放 deadline `D`、可用数据移动带宽 `B`、后台 compute duty cycle `C` 和预计持续时间 `H`。自然语言策略如“Blender 渲染时优先释放 40GB”只能编译成这些受限字段，最后由白名单 policy guard 执行。

### 4.2 Route-Preserving Elastic Expert Residency

本地 checkpoint 静态分析显示：

- tensor 合计约 74.783GiB；
- strict `experts.<id>` physical expert objects 约 59.063GiB；
- Mamba、Attention、shared experts、latent projection、embedding、router 等非 routed 权重约 15.720GiB；
- 40 个 MoE 层，每层 512 experts，每 token 固定 Top-22；
- 平均 layer-local expert object 为 2.953MiB。

每层保留 64 个 slots 时，理论 routed cache 为 7.383GiB，总权重工作集约 23.103GiB；每层 128 slots 时约 30.486GiB。对应 projection reclaim 为 51.680GiB 和 44.297GiB，但不是已经测得的运行时结果。

预测器使用 route history、expert co-activation、前层表示和 1024 维 latent input，输出 future expert prediction set。actual router 始终权威：

```text
predict -> async prefetch -> actual Top-22 route
                         -> hit: remap logical id to physical slot
                         -> miss: load exact expert and stall
```

系统不使用错误 expert，不修改 Top-k，不做 cache-aware rerouting。当前证据只支持“actual route 不被 predictor 替代”，不支持跨独立请求的逐 token 等价。Conformal-style calibration 只用于控制 held-out trace 上的边际 miss 风险；prompt/domain 漂移时必须扩大集合或退出 elastic mode。

动态 resident budget 不是孤立 cache 参数。控制器同时选择：

- 每层 expert slots；
- prefetch horizon 和 prediction-set 风险；
- NVMe/RDMA/UMA 带宽上限；
- decode token rate/duty cycle；
- 是否升级到 yield/hibernate。

必要条件为：

```text
token_rate * expected(miss_bytes + false_prefetch_bytes)
    <= min(storage bandwidth, staging bandwidth, foreground UMA slack)
```

不满足时继续 paging 没有意义，系统必须降速或休眠。

### 4.3 HiberCache 与 Token Transaction

深度休眠需要保存的不是第二份 75GiB 权重，而是正在演化的 live state：

- prompt 与已生成 token IDs；
- Attention KV block 与未满 tail；
- NemotronH Mamba conv/temporal state；
- sampler RNG、grammar state；
- scheduler block table；
- 带单调序号的输出 ledger 和 client commit boundary。

Attention KV append-only，适合异步 shadow flush；Mamba state 每 token 整体更新，在复制当前约 162MiB state 与从旧 checkpoint replay suffix 间选择。只有 object checksum、manifest 和 epoch commit durable 后才允许释放 model allocations。

现有 vLLM `OffloadingConnector + TieringOffloadingSpec` 是数据承载基础，但并不自动提供 live request transaction。真实 Nemotron 的 Mamba serializer、block-table restore 和 RNG continuity 仍需实现和验证。

### 4.4 SparkLoad 与分层数据路径

DGX Spark CPU/GPU 共用 128GB UMA，因此 CPU offload 不能作为容量层。PLLM 的真实层级是：

```text
expert/model slots in UMA
        <-> local NVMe canonical shards
        <-> remote host memory over ConnectX-7
```

Spark 不支持 GPUDirect RDMA。PLLM 因此实现两条语义不同的 host-staged 路径：

- durable object store：`CX-7 -> registered host buffer -> local SSD atomic cache -> checked Python buffer -> CUDA/UMA slot`，提供 checksum、manifest 和落盘语义；
- volatile remote pool：71 预注册大块 host MR，75 以持久 RC QP 执行 one-sided PUT/GET，71 CPU 与文件系统不进入数据热路径。

当前 volatile GET 仍先写 75 本地 `.pllmex`，再由 Python/CUDA loader 入 slot，因此不是 NIC 直达 GPU，也不是零拷贝。两条路径均受前台感知 I/O governor 限制；直接 pinned-staging-to-slot 是下一阶段的性能工作。

Level 2 恢复直接读取共享模型 checkpoint，不在每次休眠时写出第二份 75GiB 权重。`fastsafetensors`、multithread safetensors 和未来 expert-object loader 的 transform 时间与 I/O 时间分别记录。

### 4.5 产品界面与规划视图

现有 PySide6 置顶悬浮窗显示当前前台应用、系统状态、PLLM 模式、释放 GiB、GPU/UMA/功耗和一键 release/wake。Vue 3 控制中心由 Flask 静态托管，不使用 Node.js。迭代 3 已加入 recommendation-only 的 Expert Residency 规划视图：

- Full/Elastic/Hibernated 状态时间线；
- 40 层 slots、投影 resident/reclaim bytes、miss traffic 和 token-rate cap；
- 空闲、创作与紧急三组资源包络交互；
- NVMe/RDMA/UMA 流量及 I/O governor；
- 决策原因、资源包络与 phase-boundary 告警；
- `CONTROL PLANE ONLY`、`NOT EXECUTABLE` 和 evidence source 标识；
- 实验矩阵和原始 JSON/CSV 下载。

其中每层柱状图表示 planner 的 slot 投影，不是实际 cache heatmap。真实 `predicted set/actual Top-22/hit/miss` 只有接入 route tracer 和 slot data plane 后才允许显示为 live data。

## 5. 创新点与现有工作的差异

“预测 expert 并预取”本身不是创新：MoE-Infinity、ProMoE、ExpertFlow、Fate 和 Pre-Attention Prediction 已经覆盖 activation trace、跨层预测和 proactive cache；SpecMD、FlashMoE 和 ActiveEvict 已覆盖淘汰、SSD 与动态 budget。

PLLM 的可辩护创新是三者的联合问题：

### 5.1 外部前台 SLO 驱动的可收缩 residency

已有 offloading 工作大多在固定 GPU budget 下优化模型 TPOT。PLLM 的 budget 由不可控前台实时改变，并同时限制物理容量、NVMe/RDMA、UMA、算力和功耗。系统研究的是是否存在 `full resident` 与 `full hibernate` 之间的 Pareto 区间。

### 5.2 风险校准但严格精确的专家预取

PLLM 不以 router prediction 替代真实 routing。Prediction set 只决定提前搬什么，actual Top-22 决定必须执行什么。误预测成本被转化为可测的 stall/I/O，而不是隐藏的模型质量变化。风险校准进一步把“集合多大”与前台带宽预算连接起来。

### 5.3 Elastic-to-Hibernate phase boundary

专家换页不是万能模式。PLLM 在线估计 miss debt；一旦所需流量超过前台允许带宽，就降低 decode rate、yield 或事务式休眠 live hybrid request。这个切换把稳态 expert cache 与 Attention/Mamba/token transaction 纳入同一资源包络。

上述贡献目前属于研究设计。只有 expert slot data plane、真实 traces、前台实验和事务正确性全部成立，才足以形成强系统论文；当前不能把 A+B+C 的架构图当作已证明创新。

## 6. DGX Spark 适配价值

- 128GB UMA 允许 120B NVFP4 模型运行，也使 CPU offload 失去物理容量意义。
- 273GB/s 共享带宽同时承载 GPU compute、CPU、NVMe/NIC staging 和前台图形，需要显式 governor。
- 两个 copy engines 为预取重叠提供条件，但不能消除 LPDDR 和存储能耗。
- ConnectX-7 提供远端容量源，但无 GDR，必须 host-stage。
- 140W GB10 功耗包络使 SSD/RDMA expert traffic 的能耗成为一等指标。
- NemotronH 同时包含 Mamba、Attention、LatentMoE 和 NVFP4，是检验混合状态与 fine-grained expert residency 的高难度对象。

## 7. 当前实现与证据边界

### 已在真实模型/硬件上验证

- 完整回归 `69 passed`，CMake、compileall 和 shell syntax 通过；
- 完整导出 20,480 个 Marlin runtime experts、约 60GiB，跨 40 层抽样 checksum 全部通过；
- 128-slot ModelOpt NVFP4/Marlin EER 启动、actual Top-22 blocking load 和真实生成；
- vLLM Level 1/2 与 PLLM API hibernate/wake；Level 2 在 0.131--0.185s 回收约 43--44GiB；
- 恢复后真实 OpenAI proxy 请求 HTTP 200，但本地恢复约 39--42s，冷槽请求约 33s；
- 60GiB 前台 CUDA allocation 从模型常驻 OOM 变为休眠后成功；
- 20MiB RC RDMA PUT/GET integrity、token/path guard、Python store 与 15MiB live-state carrier；
- 75→71 的 4-QP volatile pool 搬运 5,120 个真实 runtime expert objects（15.859GB）：PUT wall 2.545s（49.85Gb/s），GET 到本地文件 wall 4.012s（31.62Gb/s）；
- Blender 5.2 OptiX 场景、GNOME focus PID 与 PLLM workload input 链路；当前只有预览，没有 QoS 对照；
- 实时 NVML/EER API、SSE、Vue 桌面/移动界面与 PySide6 悬浮窗。
- full-resident vLLM 完成 MQA/NQA/TQA 各 50 条：F1 0.3600、总吞吐
  6,734.30 tok/s；EER-256 首条 MQA 在 499s 内未完成，延迟下界 >226.09x，
  累计换入 358.43GiB，构成真实 paging-collapse 反例。

### 已实现但只具部分证据

- route-history/conformal predictor 只有 synthetic trace，不能代表真实 Nemotron 路由局部性；
- Level 0 同一 HTTP stream 可以冻结且不重连，但跨独立请求 bitwise determinism 失败，不能声称 exact token resume；
- live-state carrier 的 SSD/RDMA transaction 已验证，Mamba/KV/RNG serializer 尚未接入；
- 128 slots 的 exact fused Top-22 需要 `max_num_batched_tokens<=5`，路径正确但性能不可接受；
- full 512 experts + Sleep Mode 在 Marlin repack 阶段启动 OOM；Level 2
  恢复 EER 后 slots 为冷状态，`data_plane_ready` 尚不能代表 warm-set ready；
- 自动 destructive resize 仍默认关闭，40 层 fault injection 尚不完整。

### 仍待完成

- 真实 route tracer、predictor calibration、online drift detector 与 EER baseline matrix；
- 优化后 volatile-pool v2 的跨机复测、direct RDMA-to-slot 与同 profile NVMe 基线；
- DGX Spark ConnectX-7/UMA 测试；
- Blender、游戏、NVENC 的真实 throughput/jank/energy；
- NemotronH Mamba/KV/RNG restore 与 greedy token equality。

MR staging copy、纯 verbs phase 与调用者 wall time 继续分开报告。跨机结果来自优化前协议 v1；selective-signaling/inline-header/read-depth-16 的 v2 只完成 124MB 本机 RoCE smoke test，不以本机 94.0/79.1Gb/s 替代跨机复测。详细结果见 `docs/实验报告.md`。

### LongBench QA 对照结论

完整 150 条质量结果只属于 full-resident baseline。真正具备快速释放能力的
PLLM 配置没有完成首条 EER-256 样本，因此其 F1 是 N/A，而不是 0。当前证据
支持“0.210s 回收 73,718MiB”，不支持“开启 PLLM 后保持质量与吞吐”。共享
模型占 74.846GiB，EER runtime experts 额外占 59.079GiB，部署合计
133.926GiB。仅启用 monitor/HiberCache 的全驻留控制组因另一用户作业占用
77,296MiB 而未运行，不能用 baseline 推测其开销；详细逐样本结果见
`docs/LongBench-QA开关PLLM实验.md`。

## 8. 实验计划

迭代 2 的 `results/expert_residency_simulation.json` 仍只验证 control-plane accounting：synthetic calibration coverage 1.0、set size 263，在 domain shift 上 coverage 为 0，不能升级为真实 predictor 结论。当前真实 runtime 已报告 40 层、128 physical slots 和 `data_plane_ready=true`；前端因此可以展示 LIVE 数据面，但 planner 的 256-slot 建议必须与实际 128 slots 分栏。自动 resize 保持默认关闭。

### 8.1 Expert trace 与预测

- workload：code、math、chat、RAG、长上下文和 domain shift；
- 记录每层 actual Top-22、router scores、latent input、token phase；
- 比较 LRU、LFU、Least-Stale、ProMoE-style predictor、fixed top-n 和 calibrated set；
- 指标：byte hit、set size、false bytes、blocking miss、coverage 和 drift recovery。

### 8.2 容量与前台 QoS

- resident slots：full、256、128、64、32 per layer；
- 前台：Blender、graphics trace、NVENC、NVMe media workload；
- 指标：`MemAvailable`、admission success、throughput/jank、功耗、PSI、I/O 和后台 TPOT；
- 与无后台模型前台基线比较，目标为 90%以上，但必须报告实际置信区间。

### 8.3 Phase boundary

比较始终 paging、始终 yield、始终 hibernate 和 PLLM 自适应策略。扫描 token rate、cache budget、SSD/RDMA 带宽和前台持续时间，验证是否存在稳定 elastic Pareto 区间，以及控制器是否在 paging collapse 前退出。

### 8.4 Transaction 与恢复

在每个 token boundary 注入 pause；对 Attention tail、Mamba state、RNG、manifest 和 block table 做 fault injection。Greedy 输出必须逐 token等于 uninterrupted baseline；随机采样必须恢复 RNG；跨连接 only-once 只对支持 token ACK 的 PLLM 客户端声明。

### 8.5 否证条件

以下结果必须如实报告：

- 95% byte hit 需要接近完整 64GiB expert cache；
- SSD/RDMA 流量使前台性能低于 baseline 90%；
- expert slot remap 或 NVFP4 transform 抵消容量收益；
- calibrated set 在 prompt shift 下系统性失效；
- 双源因 UMA contention 慢于最佳单源；
- Mamba/stream transaction 无法保持 token equality。

## 9. 双智能体审稿机制

仓库新增两个职责隔离的研究智能体：

- Reviewer Agent：检查新颖性、已有工作覆盖、证据、硬件事实和可证伪性；
- Rebuttal/Revision Agent：逐条选择接受、部分接受、证据反驳、降级为假设或删除主张，并输出完整修订稿。

编排器默认运行四轮，每轮保存 review、rebuttal 和 manuscript snapshot，不默认覆盖论文。项目三轮工程迭代各自又完成至少三轮 Reviewer/Rebuttal 往返，记录位于 `docs/reviews/iteration-1`、`iteration-2` 和 `iteration-3`。本地 GPU 空闲后可用任意 OpenAI-compatible/vLLM endpoint 重跑自动编排器。

## 10. 比赛提交材料

- `README.md`：安装、运行、Demo 和证据边界；
- `paper/HiberFlow-ACM四页稿.md`：最新论文设计稿；
- `docs/PLLM项目报告.md`：本报告；
- `docs/research/近一年相关工作矩阵.md`：相关工作与差异；
- `docs/实验报告.md`：实验协议与真实数据；
- `docs/部署说明.md`：conda、vLLM、GNOME 和 systemd；
- `docs/演示视频脚本.md`：评委演示流程；
- `docs/reviews/四轮审稿迭代记录.md`：审稿、答辩和修改轨迹；
- `agents/`、`scripts/run_peer_review.py`：双智能体复现实验。

## 11. 结论

PLLM 最新方案不再把资源让渡等同于“暂停并重载整模型”。它利用 Nemotron routed experts 占权重主体的结构，在保持 Top-22 完全不变的前提下提供可收缩 residency；用前台资源包络同时约束 cache、I/O、decode duty cycle 和 hibernation；当 paging 不可行时，再以事务方式保护小型 live state 并丢弃大权重。

这比单独的 expert prediction、Sleep Mode 或 SSD loader 更接近一个完整研究问题。数据面已经在 RTX PRO 上运行并证明快速释放与前台显存 admission，但也暴露出 fused Top-22 batch 上界、约 40 秒恢复和跨请求非确定性。Mamba transaction、真实路由预测、前台吞吐和 DGX Spark 实验仍未完成；创新性最终由这些问题能否形成新算法并取得 Pareto 改善决定。

LongBench 的 paging-collapse 反例进一步表明，仅增加 SSD/RDMA 带宽不能解决
问题：当 68.96% byte hit 导致单请求换入 358.43GiB 时，系统必须在在线
miss-debt 越界后退出 elastic mode。下一版的研究重点因此是可证伪的在线
phase boundary 与 layer-pipelined expert execution，而不是继续扩大静态 cache。

## 参考资料

- [MoE-Infinity](https://arxiv.org/abs/2401.14361)
- [ProMoE](https://arxiv.org/abs/2410.22134)
- [ExpertFlow](https://arxiv.org/abs/2410.17954)
- [Fate](https://arxiv.org/abs/2502.12224)
- [Pre-Attention Expert Prediction](https://arxiv.org/abs/2511.10676)
- [SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching)
- [ActiveEvict](https://openreview.net/pdf?id=UAMZ4tRFn6)
- [FlashMoE](https://arxiv.org/abs/2601.17063)
- [OD-MoE](https://arxiv.org/abs/2512.03927)
- [SSD MoE Offloading Energy Analysis](https://arxiv.org/abs/2508.06978)
- [NVIDIA DGX Spark Hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark CUDA Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
- [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
