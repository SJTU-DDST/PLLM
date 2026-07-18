# HiberFlow-EER：统一内存 AI PC 上面向前台 QoS 的弹性专家驻留与事务式休眠

**HiberFlow-EER: Foreground-SLO-Aware Elastic Expert Residency and Transactional Hibernation on Unified-Memory AI PCs**

> 研究设计与系统原型稿，证据版本对应 PLLM commit `24a7dc6`。本文区分 RTX PRO 实测、synthetic control plane 与 DGX Spark 待验证结果。当前已完成真实 Nemotron/EER、Level 1/2、CUDA admission、75→71 跨机一侧 RDMA 内存池和 GNOME 前台识别；真实 predictor、Mamba exact resume、Blender QoS 对照与 DGX Spark UMA 尚未完成。

**匿名作者**  
**匿名单位**

## 摘要

桌面级 AI 终端需要让一个后台大模型与游戏、Blender、视频编码等前台任务共享容量、内存带宽、算力和功耗。现有系统通常在两个极端间选择：保持模型完整驻留，或者暂停并卸载整个模型。前者可能阻止前台任务分配资源，后者对 75GiB 级模型产生数秒到数十秒恢复延迟。MoE 模型的稀疏激活提供了中间状态，但 DGX Spark 的 CPU 与 GPU 共用 128GB 物理内存，使传统 GPU-to-CPU expert offload 不能真正释放容量。

本文提出 HiberFlow-EER，一种保持原始 MoE Top-k 路由语义的前台感知运行时。系统把前台需求表示为随时间变化的资源包络，并在完整驻留、弹性专家驻留、token-boundary yield 和事务式深度休眠间选择。弹性阶段仅保留部分 routed experts；基于路由历史和中间表示预测后续工作集，从本地 NVMe 或 ConnectX-7 远端内存预取。实际 router 结果始终具有权威性：预测 miss 必须等待正确 expert，不允许用预测 expert 替代，因此该路径不引入模型质量损失。系统使用在线校准的 prediction set 控制 miss 风险，并联合选择 expert cache 容量、预取范围、I/O 上限和 decode duty cycle；当预计 miss 流量超过前台允许的带宽时，系统降低生成速率或转入深度休眠。

深度休眠阶段沿用 HiberFlow 的事务式状态协议：在 token 边界提交 Attention KV、Mamba recurrent state、采样器和输出 ledger，直接丢弃已有 canonical checkpoint 的不可变权重，并从 NVMe 或无 GPUDirect RDMA 的 host-staged RDMA 路径恢复。本文实现了只读取 17 个 safetensors header 的 catalog；它确认 74.783GiB tensor 中严格匹配 `experts.<id>` 的 physical expert objects 为 59.063GiB，非 routed 权重为 15.720GiB。

在 RTX PRO 6000 96GB 上，128-slot EER 能完成真实生成；PLLM Level 2 在 0.185s 内回收 44.17GiB，使原本 OOM 的 60GiB 前台 allocation 成功，但本地恢复仍需 41.39s。75→71 的 16GiB volatile remote-memory pool 以 4 个 RC QP 搬运 5,120 个真实 runtime expert objects（15.859GB）：PUT wall time 为 2.545s（49.85Gb/s），GET 到本地文件为 4.012s（31.62Gb/s），71 侧关键路径不落盘。该数据来自优化前协议；最新 selective-signaling/inline-header v2 只完成 124MB 本机 RoCE smoke test，不能用本机 94.0/79.1Gb/s 替代跨机复测。

实验同时暴露两个关键负面结果：fused Top-22 要求 `22B<=128`，故精确路径将 batched tokens 限制为 5，首个短请求耗时 63.94s；Blender 5.2 OptiX 场景、GNOME focus PID 和 PLLM workload 输入虽已贯通，但正式 foreground render/LLM 对照尚未运行。当前结果证明容量 admission 与远端 profile 搬运可行，不证明 EER 已达到可用吞吐、Blender QoS 已改善或事务式逐 token 等价。

**关键词：** MoE inference；expert offloading；foreground QoS；unified memory；vLLM；NVMe；RDMA；checkpoint/restart

## 1. 问题与核心观点

DGX Spark 能在桌面运行 100B 级模型，但桌面并不是独占服务器。前台程序的需求具有突发性：游戏启动需要立即分配大量图形资源，Blender 渲染同时需要计算和内存带宽，视频创作会使用 NVENC/NVDEC、GPU 和 NVMe。仅观察后台 LLM 的吞吐无法反映用户体验；Sereno 在移动 SoC 上展示了前台卡顿与后台 LLM 吞吐下降之间的明显不对称 [1]。

vLLM Sleep Mode 可以冻结 scheduler 或丢弃权重 [2]，但只有完整驻留和完整休眠两档仍然过于粗糙。目标模型是 fine-grained MoE：每个 token 只使用 512 个 routed experts 中的 22 个。若前台只需要 20--40GiB，立即卸载全部 74.8GiB 会支付不必要的恢复成本；若只暂停计算而保持所有权重，又无法满足容量 admission。

本文研究的问题是：

> 给定前台在 deadline `D` 内要求的物理容量 `R(t)`、可用 I/O/UMA 带宽 `B(t)` 和后台计算份额 `C(t)`，能否在不改变原始 Top-k 路由和输出语义的条件下，动态收缩 MoE expert 工作集；当稳态换入换出不可行时，再事务式休眠 live request？

核心观点不是“expert activation 可以预测”，该方向已有大量工作 [3--8]。本文的研究对象是**外部前台 SLO 驱动的可收缩 expert residency，以及弹性稳态与事务式休眠之间的可行性边界**。

本文贡献被限定为：

1. **统一的前台资源包络。** 将 expert cache 容量、预测集合、decode duty cycle 与 hibernation 放入同一决策问题；系统在容量、带宽或 deadline 不可满足时有可解释的降级边界，而不是始终尝试换页。
2. **保持路由语义的风险校准专家驻留。** 使用 prediction set 预取原始 Top-22 的候选 experts；actual route miss 只导致精确加载和 stall。在线 calibration 控制的是边际 miss 风险，不被夸大为逐 token 条件保证。
3. **UMA/no-GDR 数据路径。** 在 DGX Spark 上把 NVMe 和远端 host memory 作为真实容量层，CPU pages 只作 staging；expert slot cache 与整模型 SparkLoad 共用受前台带宽约束的分块 loader。
4. **弹性驻留到事务式休眠的连续协议。** 中等压力下保留 live state 并逐层换 expert；高压力下提交 Attention/Mamba/sampler/token ledger 后丢弃不可变权重，恢复时保持逻辑 exactly-once stream。

## 2. 动机与定量边界

### 2.1 Nemotron 的可收缩部分

本地只读 checkpoint `/mnt/ssd-storage/shared_models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` 包含 88 层：40 个 Mamba 层、40 个 MoE 层和 8 个 Attention 层。每个 MoE 层有 512 个 routed experts，每 token 固定激活 22 个；本文不修改该数字。

解析 safetensors header 得到：

| 权重类别 | 大小 |
| --- | ---: |
| physical routed expert objects (`experts.<id>.*`) | 59.063GiB |
| 非 routed 权重（Mamba、Attention、shared experts、latent projection、embedding、router 等） | 15.720GiB |
| tensor 合计 | 74.783GiB |
| 17 个 checkpoint 文件合计 | 74.802GiB |

平均每个 layer-local routed expert 约为：

$$S_e=59.063GiB/(40\times512)=2.953MiB.$$

若每层保留相同数量 `K` 的 expert slots，忽略 allocator padding 后：

| 每层 slots `K` | routed cache | 加 15.720GiB 非 routed 权重 | 相对完整权重释放量 |
| ---: | ---: | ---: | ---: |
| 32 | 3.691GiB | 19.411GiB | 55.371GiB |
| 64 | 7.383GiB | 23.103GiB | 51.680GiB |
| 128 | 14.766GiB | 30.486GiB | 44.297GiB |
| 256 | 29.531GiB | 45.251GiB | 29.531GiB |

这些是静态容量算术，不是运行时释放结果。实际实现还要计入 quantization scale、workspace、CUDA graph、KV/Mamba state 和碎片。

### 2.2 换容量会产生带宽债务

一个 token 在 40 个 MoE 层共需要 `40*22=880` 个 expert instances。若全部 miss，按 checkpoint bytes 估算需要约：

$$B_{all-miss}=59.063GiB\times22/512\approx2.538GiB/token.$$

在 10 token/s 下，90%、95% 和 99% 的 byte hit rate 分别仍产生约 2.538、1.269 和 0.254GiB/s miss traffic，错误预取还会增加流量。因此“准确率 99%”不是充分指标；系统必须测量**按 deadline 到达的 byte hit rate、false-prefetch bytes 和 foreground bandwidth loss**。

稳态弹性驻留的必要条件为：

$$r\cdot E[B_{miss}+B_{false}]\le
\min(B_{nvme}+B_{rdma},B_{stage},B_{uma,slack}/\alpha),$$

其中 `r` 是后台 token rate，`alpha` 表示 NVMe/NIC 写 staging、staging 读和 destination 写对 UMA 的放大。若不等式不成立，继续 paging 只会把容量竞争转换成 I/O 与功耗竞争；正确动作是降低 `r` 或 hibernate。SSD expert offloading 的能耗风险已有定量警告 [9]，PLLM 因而把能耗作为约束而不是只优化 TPOT。

### 2.3 UMA 不是 HBM 加 host DRAM

DGX Spark 的 CPU 与 GPU 共用 128GB LPDDR5x 和 273GB/s 带宽 [10]。把 expert 从 CUDA allocation 复制到普通 CPU pages 可能改变 accounting，却没有增加物理 `MemAvailable`，甚至短暂形成第二份权重。故深层 tier 必须是 NVMe 或远端机器。

Spark 不支持 GPUDirect RDMA、nvidia-peermem、DMA-BUF 或 GDRCopy；NVIDIA 建议 verbs 应用使用 `cudaHostAlloc` 后注册 MR [11]。远端 expert 传输必须走 ConnectX-7 到 host staging，再进入可计算的 UMA slot。PLLM 保留两条不同语义的数据路径：durable object store 在远端落盘并执行 manifest/checksum；高性能 pool 在 71 预注册一块 volatile host MR，PUT/GET 由计算节点执行 one-sided RDMA，71 CPU 和文件系统不进入数据路径。当前 reload client 仍把 host staging 写成本地 `.pllmex` 后再交给 Python/GPU loader，因此没有实现 NIC→GPU slot 的零拷贝。论文不以理想 in-process 路径或其他服务器的 GDR 数据替代 Spark 结果。

## 3. HiberFlow-EER 设计

### 3.1 状态机与资源包络

系统状态为：

```text
FULL_RESIDENT <-> ELASTIC_RESIDENT -> YIELDING -> COMMITTING
                                                   |
                                                   v
                                              HIBERNATED
                                                   |
                                                   v
                                               RESTORING
```

Foreground-QoS Agent 每 250ms 融合 GNOME focus/exec、NVML 进程级 SM 与 codec 活动、`MemAvailable`、PSI、swap、供电和前台历史，产生：

$$\mathcal{E}_t=(R_t,D_t,B_t,C_t,H_t),$$

分别表示容量需求、释放 deadline、允许的数据移动带宽、后台 compute duty cycle 和预测前台 horizon。自然语言策略只编译这些受限参数，不能直接终止进程或修改模型语义。

控制器选择动作：

$$a_t=(M_e,h,r,q),$$

其中 `M_e` 为 expert cache budget，`h` 为预取 horizon，`r` 为 token rate，`q` 为 full/elastic/yield/hibernate。目标是最小化后台 TPOT、恢复代价、前台失败风险、能耗与 SSD 写放大，并满足 `MemAvailable`、I/O、compute 和 deadline 约束。

### 3.2 风险校准的 expert prediction set

令 `A_{l,t}` 为原始 router 在层 `l`、token `t` 选择的真实 Top-22，`P_{l,t}` 为在该层执行前预测并预取的集合。预测器可使用请求路由历史、相邻 token 共激活图、前层 hidden state 和 Nemotron 的 1024 维 latent input，但预测器不参与模型输出。

离线 trace 被按 request 划分为 train、calibration 与 test，避免相邻 token 泄漏到不同 split。迭代 2 已实现稀疏 route-history/co-activation predictor：训练阶段统计 layer popularity 和 previous-route-to-next-route transition，推理阶段对 512 experts 排序。对 calibration record 计算覆盖实际 Top-22 所需的最大 rank，并使用 split conformal quantile 生成 `P_{l,t}`，目标是：

$$Pr(A_{l,t}\nsubseteq P_{l,t})\le\delta_l.$$

该保证只在交换性假设下具有有限样本的**边际覆盖**；桌面 workload 或 prompt domain 漂移会破坏假设。当前 synthetic no-GPU 反例恰好展示了这一点：一个 request 的 calibration coverage 为 1.0、平均 set size 为 263，但切换到另一个 synthetic domain 后 test coverage 降为 0。该结果不是 Nemotron accuracy，而是要求运行时监控 empirical miss、漂移时扩大集合或关闭预测的单元级证据，不能宣称逐 token conditional guarantee。

全局风险预算按层分配 `sum_l delta_l <= delta`。分配器结合 layer miss latency、expert bytes、可重叠窗口和历史漂移，为慢层或短 overlap 层分配更低 miss risk。前台收缩 `M_e` 时，系统同时调整 prediction set 和 token rate，而不是通过修改 Top-k 来硬塞进预算。

### 3.3 精确 expert slot cache

每个 MoE 层维护固定物理 slots 和逻辑到物理映射：

```text
logical expert id -> {slot, generation, source, checksum, ready_event}
```

runtime expert object 包含 Marlin 已转换的 NVFP4 packed weights 与 weight scales。当前 W4A16 Marlin 路径在转换时丢弃 per-expert activation input scales，因此对象格式不虚构这两个 tensors；只有 format、model fingerprint、tensor layout 与 SHA-256 全部校验后 slot 才变为 ready。执行流程为：

1. predictor 提前提交 `P_{l,t}` 的异步读；
2. actual router 产生原始 `A_{l,t}`；
3. cache hit 直接把 logical id remap 到 slot；
4. miss 阻塞该层并加载正确 expert；
5. 所有 `A_{l,t}` ready 后运行原始 fused MoE kernel；
6. 根据预测复用价值、加载代价和收缩 deadline 异步 pre-evict。

预测错误只增加 stall、I/O 和 cache pollution，不改变 expert output。若当前 vLLM backend 不能接受 logical-to-slot remap，系统必须回退 full residency 或 hibernate，不能执行错误 expert。

淘汰 utility 为：

$$U_{l,e}=p_{reuse}L_{miss}+p_{near}D_{evict}-\lambda S_e-\mu E_{load},$$

其中 `p_reuse` 来自在线 trace，`p_near` 表示近期被实际路由的概率，`D_evict` 是前台 deadline 内保留该 expert 的机会成本。与 ActiveEvict [8] 的主要区别不是“首次动态 budget”，而是 budget 来自外部前台物理容量 SLO、路由保持精确，并与 hibernation phase boundary 联合。

### 3.4 I/O governor 与算力让渡

expert eviction 释放容量，但 inactive expert 原本不消耗 MoE FLOPs；换出本身不会自动让出算力。PLLM 使用 token bucket 限制 decode iteration，确保降低单次 I/O 后不会因 token/s 上升再次占满 GPU。前台仍活跃时，NVMe、RDMA 和 UMA copy 分别有带宽 token bucket；miss debt 连续超过阈值时依次执行：缩短 prefetch horizon、降低 decode rate、Level-0 yield、事务式 hibernate。

这构成一个可测量的 phase boundary，而不是声称 paging 在所有压力下都优于暂停。

### 3.5 混合状态的事务式深度休眠

当 elastic mode 不可行时，HiberFlow 在 token 边界提交：

$$\Sigma=(H,K,M,G,B,L,c,e),$$

其中 `H` 为 token history，`K` 为 Attention KV，`M` 为 Mamba conv/temporal state，`G` 为 RNG/sampler/grammar，`B` 为 block table，`L` 为带序号输出 ledger，`c` 为客户端提交边界，`e` 为 epoch。权重由 model hash 标识，不重复 checkpoint。

Attention KV 是 append-only block，适合持续 shadow flush；Mamba state 每 token 整体更新，系统在复制当前约 162MiB recurrent state 与从较旧 checkpoint teacher-force replay suffix 间选择。只有 object manifest、checksum 和 commit record durable 后，系统才释放 KV、expert slots、dense weights 和 workspace。

恢复若从 `r<=c` 开始，则对 `L[r+1:c]` teacher-force 重放但不再次发送，随后恢复 RNG 并继续生成。该协议目标是 token ledger 的 logical exactly-once；标准 OpenAI SSE 没有 ACK，跨连接保证仅对支持 `(request_id,token_seq)` 的 PLLM 客户端成立。

### 3.6 SparkLoad 与多源恢复

完整 hibernation 后，SparkLoad 从原始 17 个 safetensors shards 或远端暖副本恢复，不生成第二份 74.8GiB checkpoint。expert mode 与整模型 mode 共用 logical chunk manifest：

- local NVMe：`O_DIRECT/io_uring -> cudaHostAlloc staging -> destination`；
- remote：`ConnectX-7 RDMA -> ibv_reg_mr(cudaHostAlloc) -> destination`；
- page cache：仅在不会压缩前台 `MemAvailable` 时作为 opportunistic source。

调度器按实测 queue delay、有效带宽和 UMA 放大选择 source。双源并不保证更快；若并发路径争用同一 UMA，planner 必须退化到最佳单源。

### 3.7 Marlin 的边界

Marlin 的 query-agnostic token/head selection 与 I/O-compute overlap 仅用于可复用长前缀 KV 的独立实验组，不进入 live expert routing 或正在生成请求的精确状态。Nemotron 权重已经是 NVFP4，本文不再次量化 routed experts。这样可以避免把近似 KV、expert paging 和事务正确性混成一个无法归因的结果。

## 4. 实现状态

第一版只适配 vLLM 0.25.1。现有仓库实现 Foreground-QoS 控制面、Level-0 `mode=keep`、Sleep API 管理、OpenAI proxy、SQLite event log、PySide6/Vue UI、HiberCache、能力探测和 host-staged RDMA。数据面把 Marlin 转换后的每个 expert 保存为带模型指纹、tensor layout 和 SHA-256 的原子 `.pllmex` 对象；弹性启动只为每层分配 `K` 个 ModelOpt NVFP4 physical rows，并在 Marlin kernel 前根据 actual Top-22 同步加载和发布 `expert_map`。完整导出已产生 20,480 objects、63,435,912,912 bytes；128-slot runtime 已完成真实推理与 Level 1/2 恢复。通用 live-state store 的 SSD/RDMA byte transaction 已验证，但 NemotronH Mamba/KV/RNG serializer 尚未接入。

以下表格区分真实模型/硬件验证、synthetic control plane 和未来工作：

| 组件 | 当前状态 | 完成证据 |
| --- | --- | --- |
| expert catalog 与 trace schema | LIVE | 20,480 runtime objects、抽样 checksum |
| 真实 route tracer 与 predictor dataset | 未实现 | 真实 Nemotron trace、请求级 dataset split |
| route-history predictor 与 conformal rank set | synthetic control-plane 已实现 | request split、coverage/set-size tests |
| resource-envelope planner 与控制 API | LIVE control | 真实 vLLM discovery、hibernate/wake、SSE |
| NVFP4 expert slot manager | LIVE functional / slow | 128 slots、真实 generation、fused batch cap 5 |
| per-expert SSD loader | LIVE | 完整 export、blocking miss、约 58GB first-request traffic |
| RDMA durable object path | LIVE object path | 20MiB RC PUT/GET、SHA-256；远端落盘 |
| RDMA volatile profile pool | LIVE standalone | 15.859GB、4 QP、71 侧无磁盘；GPU slot 集成待测 |
| live-state SSD/RDMA carrier | LIVE byte carrier | 15MiB snapshot、4MiB chunks、checksum |
| transactional Mamba serializer | 未实现 | 每 token pause 后 greedy token equality |
| full SparkLoad | LIVE local NVMe | Level 2 reload 41.78s；Spark UMA 待测 |

vLLM EER 通过 opt-in `sitecustomize` 在模型构造前把 `ModelOptNvFp4FusedMoE.create_weights(num_experts=512)` 改为 `K` 个 physical rows，同时保留 512-entry global `expert_map`。标准 loader 只把初始 logical experts 写入映射后的 rows；Marlin 后处理完成后，runtime cache 以 kernel-format rows 覆盖它们。执行时 `topk_ids` 是唯一权威来源：runtime 在同一 worker 中阻塞，直到所有 actual experts 已校验并写入，之后才更新 GPU `expert_map` 并调用原 kernel。若单批 unique experts 超过 slots、对象缺失、checksum 失败或 mapping publish 失败，本次 inference 抛错，不调用 kernel。

动态容量变化要求 `--enforce-eager`。PLLM 先用 vLLM Level 0 在 token boundary quiesce，然后撤销 mapping、释放旧 layer parameters、按 32/64/128/256/512 profile 重建 tensors、从 backing store 恢复 retained experts 并重建 Marlin kernel。40 层 resize 目前是 fail-closed、非跨层原子事务：中途失败会保持 vLLM quiesced，需要重试或重启。首次使用前还必须完整加载一次 512-expert 模型，将约 59GiB routed runtime rows 导出至 `/mnt/ssd-storage/$USER/pllm-experts`；这是额外 SSD 占用，而不是免费的 checkpoint view。

远端 tier 不再使用 `ib_write_bw` 代替数据传输。`pllm-rdma-store` 用 TCP 控制面交换 RC QP 与 MR，执行 checksummed expert package 的 get/put 和远端落盘确认；`pllm-rdma-pool` 则在 71 分配并注册大 MR，每个 client 建立持久 RC QP，用确定性 slot index 直接 one-sided read/write。PLLM 状态机保证 PUT 与 GET phase 分离，分片 client 拥有不相交 slots，因此热路径只依赖同 QP RC ordering：payload WR 在前，64B inline commit header 在后，每个 queue-depth-32 batch 只请求一个 CQE，不逐对象扫描 checksum。完整 `.pllmex` SHA-256 在实验边界验证。该约束是研究原型的 epoch contract，不是通用并发 object-store consistency。

当前 v2 还把 RDMA READ credits 从 1 提升到协商后的 16，并分别输出 `rdma_seconds`、worker phase 和 process/QP/setup/local-I/O 全部计入的 wall time。正式结论以 wall time 为主；纯 verbs phase 只用于定位瓶颈。reload 仍写本地文件再进入 CUDA/UMA slot，论文仍不声明 GPUDirect RDMA。控制器在 DGX Spark 上用 system `MemTotal` 构造 coherent-UMA capacity，在独显上使用 NVML VRAM total。

## 5. 评估设计

### 5.1 平台与基线

当前先在明确标注的 RTX PRO 6000 + 100Gbps 主机完成可行性验证；最终在 DGX Spark 上复验 Nemotron NVFP4。双机实验使用第二台 Spark 或明确标注的 100Gbps 主机。前台 workload 包括 Blender Benchmark、可重复 graphics trace、NVENC encode 和 NVMe media trace。基线为：

1. vLLM full residency；
2. Level-0 yield；
3. Level-2 drop/reload；
4. kill/restart；
5. LRU/LFU expert cache；
6. MoE-Infinity/ProMoE/SpecMD/ActiveEvict 可复现策略；
7. EER without calibration、without I/O governor、without hibernation boundary；
8. HiberFlow-EER 完整策略。

### 5.2 指标与正确性

报告 detection latency、deadline miss、`MemAvailable` 增量、resident expert bytes、byte hit rate、false-prefetch bytes、blocking I/O、TPOT、前台吞吐/jank、NVMe/RDMA/UMA 带宽、整机功耗、SSD read/write amplification、hibernate barrier、restore time 和 first resumed token。

正确性采用四层检查：

- 每次 MoE 执行的 actual expert ids 与 full-resident baseline 完全一致；
- 每个 expert object weight/scale checksum 与 checkpoint 一致；
- greedy 输出逐 token 相等；
- transaction 在 object write、manifest rename、commit 和 unmap 前后 fault injection，不重复或遗漏已提交 token。

### 5.3 待验证假设

- **H1：存在有用的 elastic 区间。** 在释放 32--56GiB 权重容量时，EER 的前台吞吐优于 full residency，恢复延迟显著低于 Level 2。
- **H2：风险校准优于固定 predictor top-n。** 在相同 resident bytes 下，降低 p95 blocking miss；分布漂移时能够扩大集合或退出 elastic mode。
- **H3：联合控制优于单独 caching。** 仅提高 cache hit 可能伤害前台 NVMe/UMA；加入 I/O governor 和 decode duty cycle 后，前台达到无后台基线的 90%以上。
- **H4：phase boundary 能避免 paging collapse。** 当必要条件不成立时，自动 yield/hibernate 的 p95 用户体验优于持续换页。
- **H5：深度路径保持 stream continuity。** greedy token ledger 与 uninterrupted baseline 完全一致；遗漏 Mamba state 的对照应稳定失败。

这些是假设，不是结果。若真实 expert access 缺乏局部性、NVMe 能耗过高、slot remap 开销抵消收益，H1--H4 都可能失败。

### 5.4 当前实测与算法证据边界

`results/expert_residency_simulation.json` 固定声明 `trace_source=synthetic_no_gpu` 与 `real_route_evidence=false`。它验证了三个不变量：所有 actual Top-22 miss 最终都会精确加载；cache simulator 分开计算 resident hit、useful prefetch、blocking miss、false-prefetch 与 eviction bytes；planner 能在 full、elastic、yield 和 hibernate 间按容量、I/O 与 compute envelope 转换。

该回放没有展示模型效果。相反，它构造的 domain shift 使 263-candidate prediction set 在 32/64/128 slots 下全部 over budget，并触发 hibernation；只有显式标记为 `hypothetical_control_input_not_model_measurement` 的 95% hit 场景产生 256 slots、1.75 token/s 的 elastic plan。这个反例防止系统把 calibration success 当作部署保证。

真实验证已完成 `61 passed`（排除一个依赖当前 `/run/user` 是否存在的旧环境假设）、CMake、完整 expert export、128-slot vLLM、Level 1/2、60GiB CUDA admission、RC RDMA object path 与跨机 volatile pool。vLLM Unix runtime 返回 40 层注册、20,480 objects 与 `data_plane_ready=true`；前端单独显示实际 128 slots 和 planner 建议 256 slots。Level 0 同一 stream 在暂停期间无新 chunk且无需重连，但 independent no-pause requests 也不能 bitwise deterministic，因此 transactional exact resume 仍未建立。

跨机 pool 使用 100Gb/s RoCEv2 链路、4 个不相交 shard/QP 和 32 queue depth。15,858,978,307 bytes 的 PUT wall time 为 2.545s，GET 到 75 本地文件 wall time 为 4.012s；对应 49.85 和 31.62Gb/s。PUT 的 `sum(bytes)/max(worker phase)` 为 90.19Gb/s，但该值没有计入 QP/MR/process setup，不能作为用户可见 offload latency。GET 的本地文件写解释了 wall 与网络 phase 的明显差距。优化后的协议 v2 尚未在 71 重启后跨机复测，因此不报告 v1→v2 speedup。

桌面链路完成了 Blender 5.2 LTS、RTX PRO 6000 OptiX 枚举、程序化 Cycles 场景、GNOME Shell 42 focus bridge 和 PLLM 250ms monitor 的集成；preflight 能返回真实 `blender_blender.desktop` 与 PID。仅运行了 800x450、16-sample 预览，没有执行 full-resident/PLLM/无后台三组正式 render，因此本文不报告 Blender 时间、samples/s 或前台 90% 指标。

## 6. 相关工作与创新边界

MoE-Infinity [3] 使用 batch=1 activation trace 管理 expert cache；ProMoE [4]、ExpertFlow [5]、Fate [6] 和 pre-attention prediction [7] 预测未来 experts；SpecMD [12] 说明 LRU/LFU 的时序假设不可靠并提出 Least-Stale；FlashMoE [13] 面向 SSD cache；ActiveEvict [8] 做动态 budget 与 pre-eviction；OD-MoE [14] 甚至在多节点上按需加载。故本文不声称首次预测、预取、SSD offload 或动态 budget。

与这些工作的差异是问题约束：它们主要在固定设备预算下优化 LLM latency/throughput；HiberFlow-EER 的预算由不可控前台实时收缩，容量层是 UMA 外的 NVMe/remote host，控制器必须同时保护前台 I/O/compute SLO，并在 expert paging 不可行时把 live hybrid request 事务式转入 hibernation。风险校准 prediction set 的价值是让“扩大预取还是降低 token rate”具有可测风险量，而不是提高模型质量。

FastServe [15]、ConServe [16] 和 Sereno [1] 关注细粒度 preemption 或带宽干扰；ServerlessLLM [17]、MAIO [18] 优化模型恢复；Sparse Prefix Caching [19] 研究 recurrent state checkpoint。本文把这些能力放入同一个 time-varying resource envelope，但是否形成足够强的系统贡献取决于真实 EER 和 transactional state 实现，而不是架构图本身。

## 7. 限制与否证条件

1. Conformal coverage 是分布假设下的边际保证，无法保证每个 token；严重 domain shift 必须触发 conservative fallback。
2. expert paging 不减少原始 Top-22 的 FLOPs；让出算力依赖 decode duty cycle。只做 cache shrink 可能让后台更慢却仍干扰前台。
3. NVMe 读取可能显著增加能耗和与创作应用的存储竞争；缓存命中率提高不等于用户体验提高。
4. vLLM fused NVFP4/Marlin slot runtime 已运行，但 128 slots 要求 `max_num_batched_tokens<=5`；解除该上界是性能主问题。版本或 backend 不匹配仍会拒绝启动。
5. 双源恢复共享 UMA，理论链路带宽不能相加后直接作为恢复速度。
6. 首次 transformed-expert export 需要完整模型驻留并额外使用约 routed-weight 规模的 SSD；频繁 miss 还会产生 Python/CUDA 同步和 host-to-device copy。
7. 40 层 destructive resize 尚非原子事务；任一层失败会设置 `faulted=true`、撤销 ready 状态并让服务保持 quiesced。自动 resize 默认关闭。真实模型只证明固定 128-slot profile 可执行，不证明在线 resize 安全或高效。
8. volatile RDMA pool 假设状态机提供 phase-separated epochs；同 slot 并发 PUT/GET 不受支持，71 重启会丢失全部对象。跨机实测来自协议 v1，v2 性能优化尚未跨机复验。
9. Blender demo 目前只证明可重复场景、OptiX backend 和 foreground PID 棴测链路；没有证明策略检测延迟、渲染吞吐、功耗或恢复后 TTFT。

核心假设在以下任一结果下被削弱：不存在比 full residency 和 hibernate 更优的 elastic Pareto 区间；保持 95%以上 byte hit 所需 cache 接近完整 expert 权重；SSD/RDMA traffic 使前台低于无后台基线 90%；slot miss 无法在 deadline 内回退；或 transactional recovery 不能保持 greedy token equality。

## 8. 结论

HiberFlow-EER 把桌面 MoE 资源管理从二元 pause/restart 改写为一个有明确退出条件的连续问题：中等压力下保持原始 Top-22，收缩 routed-expert residency 并精确按需加载；带宽或 deadline 不可行时，提交小型 live state、丢弃已有 canonical copy 的大权重，并从 NVMe/host-staged RDMA 恢复。其潜在创新不在单项预测或 offload，而在前台 SLO 驱动的动态资源包络、风险校准 exact residency 与事务式 hibernation phase boundary。该主张必须由数据面 correctness、Nemotron traces 和 DGX Spark 前台实验验证。

## 参考文献

[1] Xin et al. [Sereno: Inference in the Shadows](https://www.usenix.org/conference/osdi26/presentation/xin). OSDI 2026.  
[2] vLLM. [Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/).  
[3] Xue et al. [MoE-Infinity](https://arxiv.org/abs/2401.14361).  
[4] Song et al. [ProMoE](https://arxiv.org/abs/2410.22134).  
[5] He et al. [ExpertFlow](https://arxiv.org/abs/2410.17954). DAC 2026.  
[6] Fang et al. [Fate](https://arxiv.org/abs/2502.12224).  
[7] Zhu et al. [Pre-Attention Expert Prediction and Prefetching](https://arxiv.org/abs/2511.10676).  
[8] [ActiveEvict: Budget-Aware Pre-Eviction](https://openreview.net/pdf?id=UAMZ4tRFn6). ACL ARR 2026 submission.  
[9] Kyung et al. [SSD Offloading for LLM MoE Weights Considered Harmful in Energy Efficiency](https://arxiv.org/abs/2508.06978). IEEE CAL 2025.  
[10] NVIDIA. [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).  
[11] NVIDIA. [DGX Spark Porting Guide: GPUDirect RDMA](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).  
[12] Hoang et al. [SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching). ICML 2026.  
[13] Kim et al. [FlashMoE](https://arxiv.org/abs/2601.17063).  
[14] Wang et al. [OD-MoE](https://arxiv.org/abs/2512.03927).  
[15] Wu et al. [FastServe](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang). NSDI 2026.  
[16] Wang et al. [ConServe](https://openreview.net/forum?id=eKfWG67mZB). ICML 2026.  
[17] Fu et al. [ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu). OSDI 2024.  
[18] Liu et al. [Programmable Page Cache for Model Loading](https://www.usenix.org/conference/fast26/presentation/liu-yubo). FAST 2026.  
[19] Shirokikh and Nikolenko. [Sparse Prefix Caching for Hybrid and Recurrent LLM Serving](https://arxiv.org/abs/2605.05219).  
[20] Wang et al. Marlin: I/O-Efficient Prefix KV Cache Retrieval for Long-Prefix LLM Serving. DAC 2026.
