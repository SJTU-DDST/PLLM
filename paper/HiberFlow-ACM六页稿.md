# HiberFlow-PhaseEER：前台感知的分相 MoE 驻留与可恢复推理

**HiberFlow-PhaseEER: Foreground-Aware Phase-Constrained MoE Residency for Recoverable Local Inference**

> 六页研究稿，证据更新至 2026-07-19。本文严格区分 RTX PRO/RoCEv2 实测、离线路由重放、scenario UI 与 DGX Spark 待验证结果。当前 GPU 被外部任务占用，新的 decode route/LongBench 数据表留空，不以模拟值代替。

**匿名作者**  
**匿名单位**

## 摘要

桌面 AI 终端需要让后台大模型与游戏、Blender 和视频编码共享容量、内存带宽、算力与功耗。完整驻留会阻止前台资源分配，整模型休眠又让 74.8GiB 模型支付约 40s 冷恢复。MoE 提供了中间点，但已有 expert offloading 多在固定预算下同时处理 prefill 与 decode；我们的真实反例显示，这会把长 prefill 变成灾难性 paging。

本文提出 HiberFlow-PhaseEER。核心算法不是改变 Top-k，也不是再次量化 NVFP4 权重，而是利用两个阶段的工作集不对称：**prefill 永远全驻留；只有 decode 才允许收缩 routed-expert slots**。系统用最近 256 个 decode step 的实际 Top-22 路由构造每层热集，在 384/448/480/496/504 slots 中选择最小可行档。每个档位必须同时满足容量收益、byte hit、I/O debt、实测 3MiB 对象 miss p95 和严格低于 10x 的 TPOT 上界；否则直接 yield/hibernate。预测从不替代原 router，miss 必须等待正确 expert。

PhaseEER 把约 59.063GiB routed experts 与活跃状态分开管理。每个 32K 请求的 Attention KV 与 Mamba conv/SSM 状态估算约 210MiB，两个请求约 421MiB；专家 resize 只复制保留的 GPU expert rows，并用 allocation fingerprint 验证 KV/Mamba storage 指针和字节数不变。新请求进入 prefill 前，PLLM 代理同步扩回 512 slots 并补齐缺失 expert，避免缩容状态污染 prefill。

远端路径在 71 预注册 64GiB volatile MR，容纳全部 20,480 个 runtime expert objects（63,435,912,912B）。模型装载期间后台建立持久 RC QP；decode miss 从 Remote DRAM 读入 registered host staging，再由 Python/Marlin sink 写 GPU slot，本机和 71 均不在 GET 关键路径落盘。跨机实测单专家稳态 p50/p95 为 3.93/7.40ms，22-object batch p50 为 81.90ms；64GiB MR 首连约 0.9s，因此连接必须提前预热。这些数据也说明 RDMA 只能处理高命中率下的尾部 miss，不能支撑大量逐 token 换页。

旧版全阶段 EER-256 在首条 LongBench MQA 的 499s 删失窗口内仍未完成，换入 358.43GiB，延迟下界超过 full-resident 226.09x；全驻留 150 条 QA 的 F1 为 0.3600、输出吞吐 19.04 tok/s。新 decode-only 路由重放与真实 GPU resize 实验尚待 GPU 空闲后完成。因此本文当前证明了算法、数据面和远端暖池已实现，但不预先宣称存在可用的 elastic Pareto 区间。

**关键词：** MoE inference；expert offloading；foreground QoS；unified memory；vLLM；NVMe；RDMA；checkpoint/restart

## 证据与主张阶梯

本文所有结论按三层组织，后文不跨层推断：

| 层级 | 编号 | 可以主张的内容 | 不可以推出的内容 |
| --- | --- | --- | --- |
| **Validated result** | R1 | Level 2 在 0.185s 内回收 44.17GiB，60GiB allocation 从 OOM 变为成功 | Blender FPS、DGX Spark `MemAvailable` 或能耗改善 |
| **Validated result** | R2 | 128-slot route-preserving Marlin 能生成并通过 Level 1/2 恢复 | greedy 输出与 full-resident baseline 等价，或 EER 吞吐可用 |
| **Validated result** | R3 | 64GiB remote pool 容纳完整 20,480-object warm image；v2 stream GET 跨机无两端落盘 | 已测 GPU slot 端到端 TPOT，或 DGX Spark GDR |
| **Validated result** | R4 | full-resident 完成 150 条 QA；EER-256 首条请求发生 paging collapse | EER 的 F1 为 0，或所有 slot/bandwidth 区间均不可行 |
| **Validated mechanism** | R5 | prefill/decode 分相、GPU-row resize、并发 phase 聚合和 state-island allocation guard 通过 89 项测试 | 新 PhaseEER 已通过真实 LongBench/Blender |
| **Mechanism artifact** | A1 | GNOME focus→D-Bus→PLLM 和 Blender OptiX 场景可运行 | 前台 QoS 已改善 |
| **Mechanism artifact** | A2 | live-state bytes、manifest 和 transport 可提交 | NemotronH Mamba/KV/RNG exact resume |
| **Research hypothesis** | H1--H5 | 下文定义了可证伪的 elastic/hibernate phase boundary | 顶会级算法贡献已经由当前数据证明 |

本文当前闭环是**分相数据面 + 完整远端暖池 + 旧方案负面基线**。decode-only Pareto 区间、Blender QoS、深度 exact resume 和 DGX Spark UMA 仍待验证。

## 1. 问题与核心观点

DGX Spark 能在桌面运行 100B 级模型，但桌面并不是独占服务器。前台程序的需求具有突发性：游戏启动需要立即分配大量图形资源，Blender 渲染同时需要计算和内存带宽，视频创作会使用 NVENC/NVDEC、GPU 和 NVMe。仅观察后台 LLM 的吞吐无法反映用户体验；Sereno 在移动 SoC 上展示了前台卡顿与后台 LLM 吞吐下降之间的明显不对称 [1]。

vLLM Sleep Mode 可以冻结 scheduler 或丢弃权重 [2]，但只有完整驻留和完整休眠两档仍然过于粗糙。目标模型是 fine-grained MoE：每个 token 只使用 512 个 routed experts 中的 22 个。若前台只需要 20--40GiB，立即卸载全部 74.8GiB 会支付不必要的恢复成本；若只暂停计算而保持所有权重，又无法满足容量 admission。

本文研究的问题是：

> 给定前台在 deadline `D` 内要求的容量 `R(t)`、在线 decode 路由窗口和已校准的小对象 miss 成本，能否只在 decode 释放一部分 expert memory，同时保证新 prefill、活跃 KV/Mamba 状态和 exact Top-k 语义不受影响；不可行时何时停止换页并整体休眠？

核心观点不是“expert activation 可以预测”，该方向已有大量工作 [3--8]。本文的研究对象是**外部前台 SLO 驱动、受推理 phase 约束的 expert residency，以及弹性稳态与整体休眠之间的可行性边界**。

本文贡献被限定为：

1. **Phase-Constrained Residency。** 将 prefill 的宽工作集视为不可缩约束，只在 decode 使用实际路由窗口选择容量档；并发请求按 `prefill > decode > idle` 聚合，新 prefill 在进入 engine 前强制恢复完整 expert set。
2. **Measured-Latency Guardrail。** 决策不是只看预测准确率，而是联合 checkpoint byte hit、`40x22` miss object 数、跨机 3MiB GET p95、token rate 和前台容量收益；任何估计达到 10x TPOT 的档位在执行前被拒绝。
3. **Weight/State Decoupling。** Routed experts 可以缩放，而 Attention KV/Mamba recurrent allocation 构成约数百 MiB 的 state island。在线 resize 通过 GPU-to-GPU row copy 保留热 expert，不从 SSD 重读 retained rows，并验证 state island allocation 不变。
4. **Remote Warm Expert Image。** 64GiB volatile pool、持久 QP 流式 GET 和 batched source API 使 Remote DRAM 进入 exact miss 数据面；71 与本地 destination 均不落盘。该路径明确是 host staged，不宣称 DGX Spark GPUDirect RDMA。

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
| 384 | 44.297GiB | 60.017GiB | 14.766GiB |
| 448 | 51.680GiB | 67.400GiB | 7.383GiB |
| 480 | 55.371GiB | 71.091GiB | 3.691GiB |
| 496 | 57.217GiB | 72.937GiB | 1.846GiB |
| 504 | 58.140GiB | 73.860GiB | 0.923GiB |
| 512 | 59.063GiB | 74.783GiB | 0GiB |

这些是静态容量算术，不是运行时释放结果。旧 128/256 档位释放大，但真实 prefill 已发生 paging collapse；新算法优先研究 384--504 的 decode-only 小步释放。实际实现还要计入 quantization scale、workspace、KV/Mamba state 和碎片。

### 2.2 换容量会产生带宽债务

一个 token 在 40 个 MoE 层共需要 `40*22=880` 个 expert instances。若全部 miss，按 checkpoint bytes 估算需要约：

$$B_{all-miss}=59.063GiB\times22/512\approx2.538GiB/token.$$

在 10 token/s 下，90%、95% 和 99% 的 byte hit rate 分别仍产生约 2.538、1.269 和 0.254GiB/s miss traffic，错误预取还会增加流量。因此“准确率 99%”不是充分指标；系统必须测量**按 deadline 到达的 byte hit rate、miss object 数、fixed GET latency 和 foreground bandwidth loss**。

小对象延迟不能由链路线速推出。完整 64GiB pool 上，约 3MiB expert 的持久 stream GET 实测 p95 为 7.40ms。对候选档位 `K`，路由窗口给出 `h_K`，则每 token 的对象 miss 与字节 miss 分别为：

$$N_K=40\times22\times(1-h_K),\qquad
V_K=2.538GiB\times(1-h_K).$$

PhaseEER 使用保守的：

$$T_{miss}(K)=\max(V_K/B_{online},N_KL_{p95}),$$

并要求：

$$1+T_{miss}(K)/T_{pot,full}<\tau,\quad \tau<10.$$

默认 `tau=5`，10x 是不可绕过的硬上限。该估计把同层批量 GET 的重叠忽略，因而偏保守；真实路由重放会同时报告 misses/token 与每层 miss 分布，不能只报告整体 hit rate。

稳态弹性驻留的必要条件为：

$$r\cdot E[B_{miss}+B_{false}]\le
\min(B_{nvme}+B_{rdma},B_{stage},B_{uma,slack}/\alpha),$$

其中 `r` 是后台 token rate，`alpha` 表示 NVMe/NIC 写 staging、staging 读和 destination 写对 UMA 的放大。若不等式不成立，继续 paging 只会把容量竞争转换成 I/O 与功耗竞争；正确动作是降低 `r` 或 hibernate。SSD expert offloading 的能耗风险已有定量警告 [9]，PLLM 因而把能耗作为约束而不是只优化 TPOT。

更完整地，给定每层 slots `K`、预测集合策略 `P` 和前台包络
`E_t=(R_t,D_t,B_t,C_t)`，elastic action 的可行性定义为：

$$
F_E(K,P,r)=
\begin{cases}
W_{fixed}+40KS_e+W_{state}+W_{workspace}\le C_{phys}-R_t,\\
r\,E[B_{miss}(K,P)+B_{false}(P)]\le B_t,\\
T_{quiesce}+T_{resize}(K)\le D_t,\\
U_{decode}(r)\le C_t.
\end{cases}
$$

深度休眠 action 的 deadline 可行性为：

$$F_H=[T_{quiesce}+T_{commit}+T_{drop}\le D_t],$$

恢复成本 `T_restore` 进入后续 horizon 的目标函数，但不改变当前 release deadline。HiberFlow 只在 `F_E=true` 且预测的 foreground penalty 低于 hibernate 时选择 elastic；`F_E=false,F_H=true` 是明确的 hibernation 区；两者都为 false 时只能立即 Level-0 yield 并报告 deadline violation。本文的算法候选是**在线估计并跨越该 feasibility frontier**，而不是把预测、cache 和 sleep 简单串联。当前真实实验只观测到一个 `F_H=true` 的 admission 点，尚未测出 `F_E` 的非空区域。

### 2.3 UMA 不是 HBM 加 host DRAM

DGX Spark 的 CPU 与 GPU 共用 128GB LPDDR5x 和 273GB/s 带宽 [10]。把 expert 从 CUDA allocation 复制到普通 CPU pages 可能改变 accounting，却没有增加物理 `MemAvailable`，甚至短暂形成第二份权重。故深层 tier 必须是 NVMe 或远端机器。

Spark 不支持 GPUDirect RDMA、nvidia-peermem、DMA-BUF 或 GDRCopy；NVIDIA 建议 verbs 应用使用 `cudaHostAlloc` 后注册 MR [11]。远端 expert 传输必须走 ConnectX-7 到 host staging，再进入可计算的 UMA slot。PLLM 保留两条不同语义的数据路径：durable object store 用于实验 artifact；高性能 pool 在 71 预注册 volatile host MR，PUT/GET 由计算节点执行 one-sided RDMA，71 CPU 和文件系统不进入数据路径。最新 stream client 通过一个持久子进程/QP 把远端 bytes 直接返回 Python `ExpertPayload` 并写 Marlin slot，本地 destination 也不落盘。它仍有 C++ staging→pipe bytes→Python→CUDA copy，不是零拷贝，GPU slot 端到端 TPOT 尚待真实模型测量。

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

### 3.2 Phase-Constrained Residency 算法

令 `A_{l,t}` 为原始 router 在层 `l`、decode step `t` 选择的真实 Top-22。运行时为每层保留最近 `W=256` step 的 CPU ring；不额外分配持续 GPU trace buffer。热度按 decode frequency、prefill tail frequency 和 recency 的字典序排序，得到 `rank_l(e)`。候选 `K` 的回看命中率为：

$$h_K={\sum_{l,t}|A_{l,t}\cap TopK(rank_l,K)|\over
\sum_{l,t}|A_{l,t}|}.$$

它是 online empirical estimator，不具有 conformal 或逐 token coverage 保证。窗口不足 320 layer-step 时动作固定为 `OBSERVE/FULL`。窗口充分后，控制器按 `K=384,448,480,496,504` 从小到大执行：

```text
if phase != DECODE: return FULL
if observations < 320: return OBSERVE
for K in candidates:
    reclaim = routed_bytes * (512-K)/512
    misses  = 40*22*(1-h[K])
    debt    = max(miss_bytes/B_online, misses*L_p95)
    if h[K] >= h_min and miss_rate <= B_online
       and (TPOT_full + debt)/TPOT_full < tau:
        return DECODE_ELASTIC(K)
return YIELD_OR_HIBERNATE
```

`tau` 默认 5，构造器强制 `tau<10`。算法选择**满足约束的最小 K**，从而在当前证据下最大化可释放容量；若前台要求大于该容量，不能继续减 K 绕过延迟护栏，只能整体休眠。并发时每个代理请求有独立 phase，聚合优先级为 `PREFILL > DECODE > IDLE`，因此任一新 prefill 都会禁止缩容。

### 3.3 精确 expert slot cache

每个 MoE 层维护固定物理 slots 和逻辑到物理映射：

```text
logical expert id -> {slot, generation, source, checksum, ready_event}
```

runtime expert object 包含 Marlin 已转换的 NVFP4 packed weights 与 weight scales。当前 W4A16 Marlin 路径在转换时丢弃 per-expert activation input scales，因此对象格式不虚构这两个 tensors。完整 warm-image 装载前执行 SHA-256；decode hot path 依赖 RC 与 64B commit header 校验 key/slot/size，不重复扫描 payload SHA。执行流程为：

1. actual router 产生原始 `A_{l,t}` 并更新 CPU route window；
2. cache hit 直接把 logical id remap 到 slot；
3. 同层 miss 合并为最多 32-object 的持久 QP RDMA batch；
4. 每个 payload 经 format、fingerprint、tensor layout 和 identity 检查；
5. miss 阻塞该层并写入正确 expert，全部 ready 后发布新 generation；
6. 原始 fused MoE kernel 使用不变的 Top-22 weights 执行。

工作集估计错误只增加 stall、I/O 和 cache pollution，不改变 expert output。缩容在 token-boundary `mode=keep` 下执行；保留 expert rows 以 GPU-to-GPU copy 移入新参数，避免旧实现从 SSD 重读 `Kx40` 对象。扩容回 512 后，控制器在唤醒和转发 prefill 前补齐全部逻辑专家。若 backend 不能接受 logical-to-slot remap，系统必须回退 full residency 或 hibernate。

### 3.4 KV/Mamba 状态小岛

NemotronH 的 live state 与 expert weights 在生命周期上不同。FP8 Attention KV 约为 4KiB/token，32K 时约 128MiB/request；float16 Mamba conv/SSM state 约 82.34MiB/request。因此单请求上界约 210MiB，两个并发请求约 421MiB，只是 59.063GiB routed experts 的 0.7%。PhaseEER 不为在线 resize 复制这些状态，而是保持 vLLM 原 cache arena 原位。

EER patch 在 `GPUModelRunner.initialize_kv_cache` 后绑定 cache tensors，对底层 storage 去重并记录 `(device,data_ptr,nbytes)` fingerprint。每次专家 resize 前后必须满足 fingerprint 和总 allocation bytes 完全相同，否则 runtime 进入 faulted 状态。这个 guard 证明 resize 不触碰 KV/Mamba allocation；它不等同于按 request 提取有效 block，也不证明 Level 2 exact resume。

vLLM 0.25.1 `OffloadingConnector` 已能分别处理 `AttentionSpec` 与 `MambaSpec` cache pages。PLLM 将其作为深度恢复 carrier，但实际 token ledger、sampler RNG 与 connector cache 的端到端 greedy equality 仍待 GPU 故障注入验证。

### 3.5 I/O governor 与算力让渡

expert eviction 释放容量，但 inactive expert 原本不消耗 MoE FLOPs；换出本身不会自动让出算力。PLLM 使用 token bucket 限制 decode iteration，确保降低单次 I/O 后不会因 token/s 上升再次占满 GPU。前台仍活跃时，NVMe、RDMA 和 UMA copy 分别有带宽 token bucket；miss debt 连续超过阈值时依次执行：缩短 prefetch horizon、降低 decode rate、Level-0 yield、事务式 hibernate。

这构成一个可测量的 phase boundary，而不是声称 paging 在所有压力下都优于暂停。

### 3.6 混合状态的事务式深度休眠协议（未完成）

当 elastic mode 不可行时，HiberFlow 的目标是在 token 边界提交：

$$\Sigma=(H,K,M,G,B,L,c,e),$$

其中 `H` 为 token history，`K` 为 Attention KV，`M` 为 Mamba conv/temporal state，`G` 为 RNG/sampler/grammar，`B` 为 block table，`L` 为带序号输出 ledger，`c` 为客户端提交边界，`e` 为 epoch。权重由 model hash 标识，不重复 checkpoint。

Attention KV 是 append-only block，适合持续 shadow flush；Mamba state 每 token 整体更新，系统在复制当前约 82.34MiB recurrent state 与从较旧 checkpoint teacher-force replay suffix 间选择。只有 object manifest、checksum 和 commit record durable 后，系统才应释放 KV、expert slots、dense weights 和 workspace。当前代码完成了 OffloadingConnector 的 Attention/Mamba page carrier、通用 bytes/components transaction 与 manifest-last store，但 token ledger serializer 仍为 `serializer_attached=false`；Level 2 实测依赖 vLLM 原生 drop/reload，不等价于完整提交协议。

恢复若从 `r<=c` 开始，则对 `L[r+1:c]` teacher-force 重放但不再次发送，随后恢复 RNG 并继续生成。该协议目标是 token ledger 的 logical exactly-once；标准 OpenAI SSE 没有 ACK，跨连接保证仅对支持 `(request_id,token_seq)` 的 PLLM 客户端成立。

### 3.7 SparkLoad、多源恢复与时间口径

完整 hibernation 后，SparkLoad 从原始 17 个 safetensors shards 恢复，不生成第二份 74.8GiB checkpoint。PhaseEER 的优先路径不是整模远端 reload，而是把 59GiB routed expert runtime image 常驻远端 DRAM，用于缩容后的尾部 miss 和 expert warm refill：

- local NVMe：`O_DIRECT/io_uring -> cudaHostAlloc staging -> destination`；
- remote：`ConnectX RDMA -> registered host staging -> ExpertPayload -> CUDA/UMA slot`；
- page cache：仅在不会压缩前台 `MemAvailable` 时作为 opportunistic source。

调度器按实测 queue delay、有效带宽和 UMA 放大选择 source。双源并不保证更快；若并发路径争用同一 UMA，planner 必须退化到最佳单源。

对任一 offload/reload，本文统一报告：

$$T_{wall}=T_{process/QP/MR}+T_{source}+T_{RDMA}+T_{sink}+T_{verify}.$$

`T_wall` 是用户可见主指标。`T_RDMA` 只统计 verbs post 到 completion，用于判断网络是否瓶颈；warm-image PUT 的 `T_source` 是本地 `.pllmex` 顺序读，stream GET 的 destination 不落盘，但包含 pipe/Python parse 与后续 H2D。多 worker 的 `sum(bytes)/max(worker phase)` 只能作为 diagnostic，并与 wall throughput 同时报告。

volatile pool 的一致性来自外部调度契约，而不是新 object-store 算法：PLLM 必须先 quiesce，PUT 与 GET epoch 不重叠，shards 不能拥有相同 slot。RC ordering 保证同 QP 上 commit header 排在 payload 之后。当前 wire format 没有持久 epoch ledger；71 重启后匿名页清零，旧 GET 因 header 不匹配失败，所有 clients 必须重连。该 contract 不支持同 slot 并发覆盖或 crash-consistent remote storage。

### 3.8 Marlin 的边界

Marlin 的 query-agnostic token/head selection 与 I/O-compute overlap 仅用于可复用长前缀 KV 的独立实验组，不进入 live expert routing 或正在生成请求的精确状态。Nemotron 权重已经是 NVFP4，本文不再次量化 routed experts。这样可以避免把近似 KV、expert paging 和事务正确性混成一个无法归因的结果。

## 4. 实现状态

第一版只适配 vLLM 0.25.1。仓库实现 Foreground-QoS 控制面、Level-0 `mode=keep`、Sleep API、OpenAI proxy、PySide6/Vue UI、PhaseEER、HiberCache 和 host-staged RDMA。Marlin runtime object 带模型指纹与 tensor layout；完整导出为 20,480 objects、63,435,912,912 bytes。旧 128-slot runtime 已完成功能性推理，新的 512→decode-K 在线路径已完成代码和 mock/CPU 测试，尚待 GPU 验收。

以下表格区分真实模型/硬件验证、synthetic control plane 和未来工作：

| 组件 | 当前状态 | 完成证据 |
| --- | --- | --- |
| expert catalog 与 trace schema | LIVE | 20,480 runtime objects、抽样 checksum |
| decode-only route tracer | IMPLEMENTED / GPU pending | vLLM 版本守卫、prompt-tail skip、NPZ route artifact |
| PhaseEER route window/guardrail | IMPLEMENTED | 384--504 slots、object-p95 + byte-debt、`tau<10` tests |
| resource-envelope planner 与控制 API | LIVE control | 真实 vLLM discovery、hibernate/wake、SSE |
| NVFP4 expert slot manager | LIVE old / new GPU pending | 128 slots 旧基线；GPU-copy resize 与 prefill full guard |
| retained-row resize | IMPLEMENTED / GPU pending | fast path 不读 source，state allocation guard |
| RDMA durable object path | LIVE object path | 20MiB RC PUT/GET、SHA-256；远端落盘 |
| RDMA volatile full pool | LIVE cross-host | 64GiB MR、20,480 objects、stream GET 两端不落盘 |
| RDMA expert source | IMPLEMENTED / GPU pending | persistent QP、batch API、direct ExpertPayload decode |
| live-state SSD/RDMA carrier | LIVE byte carrier | 15MiB snapshot、4MiB chunks、checksum |
| transactional Mamba serializer | 未实现 | 每 token pause 后 greedy token equality |
| full SparkLoad fallback | LIVE local NVMe | Level 2 reload 41.78s；不是论文主优化路径 |

vLLM EER 通过 opt-in `sitecustomize` 在模型构造前保留 512-entry global `expert_map`。新默认从 512 physical rows 启动，避免 prefill batch cap；进入 decode 后才缩到候选 `K`。执行时 `topk_ids` 是唯一权威来源：runtime 阻塞到 actual experts 写入后才发布 mapping 并调用原 kernel。若单批 unique experts 超过 slots、对象缺失或 mapping publish 失败，本次 inference 抛错，不调用 kernel。

动态容量变化要求 `--enforce-eager`。PLLM 先在 token boundary quiesce；每层分配新参数并以 chunked GPU copy 保留 hot rows，交换 parameter 后释放旧 allocation，再重建 Marlin kernel。该操作仍是 40 层 fail-closed、非跨层原子事务。扩回 512 时，缺失 rows 从 Remote DRAM/SSD 补齐后才唤醒并转发 prefill。首次使用前仍需约 59GiB runtime-format export，这是额外 SSD 占用。

远端 tier 不再使用 `ib_write_bw` 代替数据传输。`pllm-rdma-pool` 在 71 注册 64GiB MR，确定性 slot index 覆盖完整 warm image。PUT 与 GET epoch 分离；payload WR 在前，64B inline commit header 在后。在线 stream client 在模型加载期预热一个持久 QP，stdin 发送 key/batch，C++ 将 RDMA bytes 直接 framing 到 stdout，Python 不创建 destination file。完整 SHA 在装池边界验证，hot path 检查 commit header、模型指纹、identity 与 tensor layout。

v2 将 RDMA READ credits 提升到协商后的 16，并分别输出首连、稳态对象 p50/p95 和 wall throughput。64GiB MR 首连约 0.9s，故 runtime 必须提前连接；稳态约 3MiB GET p95 7.40ms 进入算法 guardrail。论文不声明 GPUDirect RDMA。控制器在 DGX Spark 上用 system `MemTotal` 构造 coherent-UMA capacity，在独显上使用 NVML VRAM total。

## 5. 评估设计

### 5.1 平台与基线

当前先在明确标注的 RTX PRO 6000 + 100Gbps 主机完成可行性验证；最终在 DGX Spark 上复验 Nemotron NVFP4。双机实验使用第二台 Spark 或明确标注的 100Gbps 主机。前台 workload 包括 Blender Benchmark、可重复 graphics trace、NVENC encode 和 NVMe media trace。基线为：

1. vLLM full residency；
2. Level-0 yield；
3. Level-2 drop/reload；
4. kill/restart；
5. 旧 all-phase EER-128/256；
6. decode-only LRU 与 window-LFU；
7. PhaseEER without phase guard、without object-latency term、without hibernation boundary；
8. HiberFlow-PhaseEER 完整策略。

### 5.2 指标与正确性

报告 detection latency、deadline miss、`MemAvailable` 增量、resident expert bytes、byte hit rate、false-prefetch bytes、blocking I/O、TPOT、前台吞吐/jank、NVMe/RDMA/UMA 带宽、整机功耗、SSD read/write amplification、hibernate barrier、restore time 和 first resumed token。

正确性采用四层检查：

- 每次 MoE 执行的 actual expert ids 与 full-resident baseline 完全一致；
- 每个 expert object weight/scale checksum 与 checkpoint 一致；
- greedy 输出逐 token 相等；
- transaction 在 object write、manifest rename、commit 和 unmap 前后 fault injection，不重复或遗漏已提交 token。

### 5.3 待验证假设

- **H1：decode 存在小而有用的 elastic 区间。** 释放 0.9--14.8GiB 时，TPOT 低于 5x 且前台获得可测容量收益。
- **H2：phase guard 是必要条件。** 相同 slots 下，prefill-full/decode-elastic 显著优于 all-phase paging，并保持相同 F1。
- **H3：对象延迟项优于 bandwidth-only planner。** 加入 3MiB GET p95 后能在实际 collapse 前拒绝危险档位。
- **H4：phase boundary 能避免 paging collapse。** 当 `tau` 不成立时，自动 yield/hibernate 的 p95 用户体验优于持续换页。
- **H5：深度路径保持 stream continuity。** greedy token ledger 与 uninterrupted baseline 完全一致；遗漏 Mamba state 的对照应稳定失败。

截至 2026-07-19 证据版本，假设状态为：

| 假设 | 当前状态 | 现有证据 | 仍缺少的决定性实验 |
| --- | --- | --- | --- |
| H1 | **待验证；旧大幅缩容被否证** | 384--504 可释放 0.923--14.766GiB；EER-256 >226.09x | 真实 full-route first-50 replay + 在线 resize TPOT |
| H2 | **机制完成 / GPU pending** | 新请求 prefill 同步 full、并发 phase 聚合与单测 | all-phase vs decode-only LongBench/F1/TPOT |
| H3 | **跨机成本已校准** | 单 expert p95 7.40ms，22-object batch p50 81.90ms | 用真实 route 验证 guardrail 的 accept/reject 精度 |
| H4 | **部分支持 fallback 可行性** | Level 2 使 60GiB allocation 从 OOM 变为成功 | 在相同前台 trace 上比较持续 paging 与跨越 phase boundary |
| H5 | **尚未建立** | Level 0 `keep` 可在原 stream 暂停/继续 | 接入 Mamba/KV/RNG serializer 后的 token equality 与 fault injection |

这些是假设，不是结果。若真实 expert access 缺乏局部性、NVMe 能耗过高、slot remap 开销抵消收益，H1--H4 都可能失败。

### 5.4 当前实测与算法证据边界

当前 PhaseEER 软件回归为 `89 passed`，包括 prefill/decode phase、并发聚合、严格 `<10x` guardrail、真实 route NPY 解码、cache replay、retained-row fast resize、state allocation fingerprint、RDMA stream/batch framing 与不落盘 ExpertPayload。CMake 分别验证 CUDA-host client 和无 CUDA 依赖的 71 server build。它们证明控制/协议不变量，不替代 GPU 性能。

LongBench QA 对照进一步给出了真实 phase-boundary 反例。全驻留原生 vLLM 在
150.718s 内完成 MQA/NQA/TQA 各 50 条，F1 为 0.3600、总吞吐
6,734.30 tok/s。EER-256 以 68.96% byte hit 运行同一首条 MQA，在 499s
删失点仍未完成，累计换入 358.43GiB；全驻留同请求仅需 2.207s。该结果不把
未完成预测记为错误答案，而把 F1 标为 N/A。新 guardrail 会拒绝相同的
68.96% hit 输入，但是否接受 384--504 中的任一真实档位仍待 route 数据。

71 现在以 64GiB anonymous MR 常驻完整 20,480-object image。四个 15.859GB shard 顺序 PUT 共 15.661s，63,435,912,912B 的 source-read+RDMA 有效吞吐 32.40Gb/s；纯 RDMA phase 合计 5.500s、92.27Gb/s。并发 4 QP 注册同一 64GiB MR 会触发 remote invalid request，因此装池使用顺序 QP，这不进入在线 miss。

持久 stream GET 不写 71 或 75 文件。完整 pool 上单 expert 首次 GET 含进程/QP/64GiB MR 注册为 0.895s，稳态 p50/p95 为 3.93/7.40ms。22-object batch 首次为 1.443s，稳态 p50/p95 为 81.90/113.06ms。运行时在模型装载期 prime QP，从 steady cost 开始 decode；该优化已接数据面，但 GPU/H2D 后的实际 TPOT 未测。

桌面链路完成了 Blender 5.2 LTS、RTX PRO 6000 OptiX 枚举、程序化 Cycles 场景、GNOME Shell 42 focus bridge 和 PLLM 250ms monitor 的集成；preflight 能返回真实 `blender_blender.desktop` 与 PID。仅运行了 800x450、16-sample 预览，没有执行 full-resident/PLLM/无后台三组正式 render，因此本文不报告 Blender 时间、samples/s 或前台 90% 指标。

### 5.5 关键路径对齐与可复现协议

下表不把不同对象、sink 和硬件上的带宽直接相除。所有系统结论以调用者观察到的 wall time 为主：

| 路径 | 对象与字节 | wall time | 终点语义 | 可支持结论 |
| --- | --- | ---: | --- | --- |
| vLLM Level 2 release | 44.17GiB resident allocation | 0.185s | 释放 CUDA allocation | 前台 capacity admission，不是 239GiB/s 存储吞吐 |
| vLLM Level 2 restore | 74.8GiB checkpoint + runtime rebuild | 41.39s | 恢复可接受请求的 engine | 完整冷恢复代价 |
| full warm-image PUT v2 | 63,435,912,912B / 20,480 objects | 15.661s sequential 4 shards | 71 64GiB volatile MR | 32.40Gb/s wall；92.27Gb/s RDMA phase |
| stream GET single | 3.097MB expert | steady p50/p95 3.93/7.40ms | Python bytes，无两端落盘 | online object-latency calibration |
| stream GET batch-22 | 68.144MB / 22 experts | steady p50/p95 81.90/113.06ms | 同一持久 QP + registered staging | 大量 miss 仍不可接受 |

因此，本文不声称 RDMA pool 已将 Level 2 restore 从 41.39s 缩短到 15.661s；远端对象仅覆盖 59GiB routed runtime experts，不包含 15.72GiB dense/Mamba/Attention 权重。真实 GPU slot miss 与相同对象的 NVMe baseline 完成后，才能计算 source 与 TPOT speedup。

Blender 因果实验固定使用 `/home/cong/hackathon/blender_demo/project/pllm_foreground_demo.blend`、Cycles OptiX、1920x1080 和 256 samples。先运行 1 次不计分预热，再对下列三组随机交错运行 5 次：

1. **B0：无后台 LLM。** 建立前台 render 上限。
2. **B1：full-resident vLLM，PLLM 不介入。** 衡量背景干扰。
3. **B2：full-resident vLLM + PLLM auto。** 在同一长流式生成中由 focus 事件触发决策。

每次记录 focus-to-detection、detection-to-action、release wall time、peak GPU bytes、render seconds/samples、LLM token 序号、整机功耗、wake time 和 resumed TTFT。主结果是 `render_time(B0)/render_time(B2)` 及 B1→B2 的前台改善；后台代价单独报告，不把释放显存等同于渲染提速。三组正式数据当前均为 pending。

主张与 artifact 的固定映射如下：

| 主张 | 原始 artifact | 复现入口 |
| --- | --- | --- |
| R1 Level 2 admission | `results/nemotron_foreground_admission.json` | Nemotron foreground admission harness |
| R2 128-slot route-preserving generation/recovery | `results/nemotron_eer128_{level1,level2,summary}.json` | 128-slot vLLM + PLLM action API |
| R3 cross-host remote pool | `results/eer-memory-profile-full.tsv`、`rdma_stream_get_full_{probe,single}.json` | full pool PUT + persistent stream GET |
| PhaseEER route experiment | `results/decode_residency/`（pending） | `scripts/benchmark_decode_routes.py` |
| foreground mechanism | `/home/cong/hackathon/blender_demo/README.md` | Blender preflight + GNOME focus bridge |
| exact resume 缺口 | `/api/v1/capabilities` 的 `serializer_attached=false` | capabilities API 与 live-state tests |
| LongBench QA 与 paging collapse | `results/qa_benchmark/` | `scripts/benchmark_longbench_qa.py` |

旧证据快照以 commit `24a7dc6` 为准；PhaseEER 将产生新的 route、resize 与 Blender artifact，不覆写旧 EER 负面结果。

## 6. 相关工作与创新边界

MoE-Infinity [3] 使用 batch=1 activation trace 管理 expert cache；ProMoE [4]、ExpertFlow [5]、Fate [6] 和 pre-attention prediction [7] 预测未来 experts；SpecMD [12] 说明 LRU/LFU 的时序假设不可靠并提出 Least-Stale；FlashMoE [13] 面向 SSD cache；ActiveEvict [8] 做动态 budget 与 pre-eviction；OD-MoE [14] 甚至在多节点上按需加载。故本文不声称首次预测、预取、SSD offload 或动态 budget。

与这些工作的差异不是“首次预测 expert”。PhaseEER 的算法约束来自桌面前台：prefill 禁止缩容，decode 的档位由实时容量收益和 measured object latency 联合决定，任一新 prefill 会同步恢复完整工作集；当最小可行档仍不能满足前台容量时，算法停止 paging 而不是继续减小 cache。Fate 与 OD-MoE 追求固定小预算下的持续 offload，PhaseEER 研究的是时间变化预算下何时**不应 offload**。

FastServe [15]、ConServe [16] 和 Sereno [1] 关注细粒度 preemption 或带宽干扰；ServerlessLLM [17]、MAIO [18] 优化模型恢复；Sparse Prefix Caching [19] 研究 recurrent state checkpoint。本文把这些能力放入同一个 time-varying resource envelope，但是否形成足够强的系统贡献取决于真实 EER 和 transactional state 实现，而不是架构图本身。

## 7. 限制与否证条件

1. 当前 route-window hit rate 是经验回看值，不是未来 token 的概率保证；domain shift 只能由在线 miss debt 和 conservative fallback 处理。
2. expert paging 不减少原始 Top-22 的 FLOPs；让出算力依赖 decode duty cycle。只做 cache shrink 可能让后台更慢却仍干扰前台。
3. NVMe 读取可能显著增加能耗和与创作应用的存储竞争；缓存命中率提高不等于用户体验提高。
4. 旧 128-slot all-phase runtime 要求 `max_num_batched_tokens<=5`；新路径以 512 slots 完成 prefill，并只在 `max_num_seqs<=2` 的 decode 缩容。真实在线转换尚未验证，绕过 PLLM proxy 直接向缩容 vLLM 发 prefill 不受支持。
5. 双源恢复共享 UMA，理论链路带宽不能相加后直接作为恢复速度。
6. 首次 transformed-expert export 需要完整模型驻留并额外使用约 routed-weight 规模的 SSD；频繁 miss 还会产生 Python/CUDA 同步和 host-to-device copy。
7. 40 层 resize 尚非跨层原子事务；任一层失败会设置 `faulted=true` 并保持 quiesced。retained-row GPU copy 降低 I/O，但仍有每层 old/new 短时双份 allocation 和 kernel rebuild；真实 wall time待测。
8. volatile RDMA pool 假设 phase-separated epochs；同 slot 并发 PUT/GET 不受支持，71 重启会丢失全部对象。64GiB MR 多 QP 并发注册失败，当前完整装池只能顺序完成。
9. Blender demo 目前只证明可重复场景、OptiX backend 和 foreground PID 检测链路；没有证明策略检测延迟、渲染吞吐、功耗或恢复后 TTFT。

核心假设在以下任一结果下被削弱：不存在比 full residency 和 hibernate 更优的 elastic Pareto 区间；保持 95%以上 byte hit 所需 cache 接近完整 expert 权重；SSD/RDMA traffic 使前台低于无后台基线 90%；slot miss 无法在 deadline 内回退；或 transactional recovery 不能保持 greedy token equality。

## 8. 结论

HiberFlow-PhaseEER 当前是 measurement-grounded prototype。旧 all-phase EER 的 >226x 反例促成了新的分相算法：prefill full、decode adaptive、状态与权重解耦、object-p95 硬护栏和整体休眠 fallback。完整 20,480-object warm image 已驻留 71 的 64GiB volatile pool，steady stream GET 已进入运行时 source；89 项测试覆盖关键不变量。系统尚未证明 384--504 中存在优于 full/hibernate 的真实 Pareto 区间，也未完成 exact deep resume、Blender 因果对照或 DGX Spark UMA 实验。该论文能否成立，最终取决于待运行的真实 route replay 与在线 GPU resize，而不是实现规模本身。

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
