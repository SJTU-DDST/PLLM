# HiberFlow-PhaseEER：前台感知的分相 MoE 驻留与可恢复推理

**HiberFlow-PhaseEER: Foreground-Aware Phase-Constrained MoE Residency for Recoverable Local Inference**

> 六页研究稿，证据更新至 2026-07-19。本文严格区分 RTX PRO/RoCEv2 实测、离线路由重放、scenario UI 与 DGX Spark 待验证结果。当前 GPU 被外部任务占用，新的 decode route/LongBench 数据表留空，不以模拟值代替。

**匿名作者**  
**匿名单位**

## 摘要

桌面 AI 终端需要让后台大模型与游戏、Blender 和视频编码共享容量、内存带宽、算力与功耗。完整驻留会阻止前台资源分配，整模型休眠又让 74.8GiB 模型支付约 40s 冷恢复。MoE 提供了中间点，但已有 expert offloading 多在固定预算下同时处理 prefill 与 decode；我们的真实反例显示，这会把长 prefill 变成灾难性 paging。

本文提出 HiberFlow-PhaseEER。核心算法不改变 Top-k，也不再次量化 NVFP4 权重，而是利用 phase 与层间工作集不对称：**prefill 永远全驻留；只有 decode 才允许收缩 routed-expert slots**。每层用过去 256 个真实 Top-22 route rows 排序 experts，只在下一完整窗口计分。逐层规划器在 `K_l in {256,320,384,448,480,496,504,512}` 中求满足前台释放目标的组合，并把 held-out miss tail、批量 RDMA 成本、GPU compaction、下一次 prefill expansion、kernel rebuild 和剩余 decode horizon 全部计入 `<5x` TPOT SLO；不可行时 yield/hibernate。预测不替代原 router，actual miss 必须等待正确 expert。

PhaseEER 把约 59.063GiB routed experts 与活跃状态分开管理。每个 32K 请求的 Attention KV 与 Mamba conv/SSM 状态估算约 210MiB，两个请求约 421MiB；只重建被选中的层，收缩时 GPU-copy retained rows，扩容时破坏式重建该层并从暖源补齐，避免 old+full layer 峰值。resize 前后比较 KV/Mamba allocation 与每个 storage 的首/中/尾 sampled-content fingerprint。该 guard 不等于 exact resume。前台 reserve 或已有 elastic decode 时，新 prefill 被延迟而不是无条件扩回 512。

远端路径在 71 预注册 64GiB volatile MR，容纳全部 20,480 个 runtime expert objects（63,435,912,912B）。模型装载期建立持久 RC QP；在线 client 把父进程共享 mmap 直接注册为本地 MR，RDMA READ 后 stdout 只返回 16B descriptor，Python 以 memoryview 解析，双方均不落盘。从完整 index 做 strided sampling 的 100 次实测中，1/8/22/32-object steady p95 为 0.477/2.863/42.710/43.897ms，steady throughput 为 55.5/72.4/27.3/29.0Gb/s；32-object p99 仍达 250.9ms。终点仍是 host memory，H2D 与 kernel stall 待 GPU 测量，本文不宣称 GPUDirect。

旧版全阶段 EER-256 在首条 LongBench MQA 的 499s 删失窗口内仍未完成，换入 358.43GiB，延迟下界超过 full-resident 226.09x；全驻留 150 条 QA 的 F1 为 0.3600、输出吞吐 19.04 tok/s。新 decode-only 路由重放与真实 GPU resize 实验尚待 GPU 空闲后完成。因此本文当前证明了算法、数据面和远端暖池已实现，但不预先宣称存在可用的 elastic Pareto 区间。

**关键词：** MoE inference；expert offloading；foreground QoS；unified memory；vLLM；NVMe；RDMA；checkpoint/restart

## 证据与主张阶梯

本文所有结论按三层组织，后文不跨层推断：

| 层级 | 编号 | 可以主张的内容 | 不可以推出的内容 |
| --- | --- | --- | --- |
| **Validated result** | R1 | Level 2 在 0.185s 内回收 44.17GiB，60GiB allocation 从 OOM 变为成功 | Blender FPS、DGX Spark `MemAvailable` 或能耗改善 |
| **Validated result** | R2 | 128-slot route-preserving Marlin 能生成并通过 Level 1/2 恢复 | greedy 输出与 full-resident baseline 等价，或 EER 吞吐可用 |
| **Validated result** | R3 | 64GiB remote pool 容纳完整 warm image；direct shared-host-MR GET 跨机无两端落盘 | 已测 GPU slot 端到端 TPOT，或 DGX Spark GDR |
| **Validated result** | R4 | full-resident 完成 150 条 QA；EER-256 首条请求发生 paging collapse | EER 的 F1 为 0，或所有 slot/bandwidth 区间均不可行 |
| **Implemented / GPU pending** | R5 | request-local window、逐层 planner、GPU-row resize code、prefill admission 和 sampled state guard 通过软件回归 | 新 PhaseEER 已通过真实 GPU resize/LongBench/Blender |
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

1. **Past-to-Next Phase Residency。** prefill 是不可缩约束；decode 的热集只能由过去窗口产生并在下一窗口验证，避免同窗 coverage 偏差。
2. **Layer/Horizon Capacity Planning。** 在总释放目标下求逐层 `{K_l}`，将 miss batch、只对变化层的 compaction、未来 full-prefill expansion 和剩余 decode horizon共同纳入 `<5x` empirical SLO；不可行即跨越到 yield/hibernate。
3. **Weight/State Decoupling。** Routed experts 可缩放，Attention KV/Mamba allocation 原位保留；在线检查 allocation 与 sampled content，但不把它冒充 exact deep resume。
4. **Direct Shared Host-MR Warm Image。** 64GiB volatile pool、持久 QP 和父子进程共享 MR 使 Remote DRAM 进入 exact miss 数据面，消除 payload pipe 与 Python package copy；路径仍是 host staged。

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

这些是统一 K 的静态容量算术，不是新算法的运行时结果。旧 128/256 all-phase 档发生 paging collapse；逐层规划器把 256--504 仅作为单层 options，可让少数高局部性层收缩、其余层维持 512。实际释放还要计入 scales、workspace、state 与碎片。

### 2.2 换容量会产生带宽债务

一个 token 在 40 个 MoE 层共需要 `40*22=880` 个 expert instances。若全部 miss，按 checkpoint bytes 估算需要约：

$$B_{all-miss}=59.063GiB\times22/512\approx2.538GiB/token.$$

在 10 token/s 下，90%、95% 和 99% 的 byte hit rate 分别仍产生约 2.538、1.269 和 0.254GiB/s miss traffic，错误预取还会增加流量。因此“准确率 99%”不是充分指标；系统必须测量**按 deadline 到达的 byte hit rate、miss object 数、fixed GET latency 和 foreground bandwidth loss**。

小对象延迟不能由链路线速或 `misses x single-object p95` 推出。direct shared-MR 路径对 1/2/4/8/16/22/32 objects 的 100 次 steady p95 为 0.477/0.784/1.513/2.863/26.223/42.710/43.897ms，16 objects 后出现 queue/transport 长尾。定义单调插值后的 source-cost 函数 `g(m)`。在线安全门使用后续 held-out 窗口里每层**最坏观测 miss 数** `m_l^max(K_l)`：

$$T_{risk}^{src}(\mathbf K)=\sum_l g(m_l^{max}(K_l)).$$

字节债务另行约束：

$$r\sum_l E[m_l(K_l)]S_{e,l}\le B_{online}.$$

这不是 token-total p95，也不是未来统计置信界，只是对已观测 miss-count 的保守 source-cost surrogate。离线评估对每个 held-out token 直接计算 `S_t=sum_l g(m_l,t)`，再报告 `p50/p95/p99(S_t)`，不相加逐层 quantile。当前 `g` 的终点是 shared host memory，不包含 H2D 与 Marlin stall；GPU 实验后必须用端到端 `g_l` 替换。默认用户可见 SLO 为总摊销 TPOT `<5x`，而非旧稿的 `<10x`。

稳态弹性驻留的必要条件为：

$$r\cdot E[B_{miss}+B_{false}]\le
\min(B_{nvme}+B_{rdma},B_{stage},B_{uma,slack}/\alpha),$$

其中 `r` 是后台 token rate，`alpha` 表示 NVMe/NIC 写 staging、staging 读和 destination 写对 UMA 的放大。若不等式不成立，继续 paging 只会把容量竞争转换成 I/O 与功耗竞争；正确动作是降低 `r` 或 hibernate。SSD expert offloading 的能耗风险已有定量警告 [9]，PLLM 因而把能耗作为约束而不是只优化 TPOT。

更完整地，给定逐层 slots `K_l`、过去窗口预测集合 `S_l` 和前台包络
`E_t=(R_t,D_t,B_t,C_t)`，elastic action 的可行性定义为：

$$
F_E(\mathbf K,\mathbf S,r)=
\begin{cases}
W_{fixed}+\sum_l K_lS_{e,l}+W_{state}+W_{workspace}\le C_{phys}-R_t,\\
r\,E[B_{miss}(\mathbf K,\mathbf S)]\le B_t,\\
T_{quiesce}+T_{now}(\mathbf K^{cur}\rightarrow\mathbf K)\le D_t,\\
U_{decode}(r)\le C_t.
\end{cases}
$$

深度休眠 action 的 deadline 可行性为：

$$F_H=[T_{quiesce}+T_{commit}+T_{drop}\le D_t],$$

恢复成本进入剩余 decode horizon `H` 的目标函数：

$$
\min_{\mathbf K}\left[T_{risk}^{src}(\mathbf K)+
{T_{now}(\mathbf K^{cur}\rightarrow\mathbf K)+T_{prefill}(\mathbf K\rightarrow512)\over H}\right].
$$

实现以 16MiB reclaim bucket 索引非支配 Pareto frontier，支配条件同时覆盖 actual reclaim、objective、mean miss、miss bytes、risk latency、即时动作时间与总 transition；bucket 内不设 frontier cap，parent pointer 避免复制 40 层路径。在当前离散加性模型内这是安全剪枝，不因 bucket 丢弃状态，但最坏复杂度仍可指数增长。仓库以 1,000 个固定种子异构实例与 exhaustive oracle 对照。真实 40 层求解移至后台线程，且只有容量压力存在时前台才先 Level-0 yield。future 携带 request/route/capacity generation；horizon 是精确 token ledger 向下取整到 128-token 的保守下界，执行 resize 前再次核验 generation 与当前 `H`。没有可证明下界时令 `H=0` 并拒绝 resize。HiberFlow 只在 `F_E=true` 且总摊销 TPOT `<5x` 时选择 elastic；`F_E=false,F_H=true` 是 hibernation 区，两者都 false 时 Level-0 yield。当前真实实验只观测到 `F_H=true` 的点，尚未证明 `F_E` 非空。

### 2.3 UMA 不是 HBM 加 host DRAM

DGX Spark 的 CPU 与 GPU 共用 128GB LPDDR5x 和 273GB/s 带宽 [10]。把 expert 从 CUDA allocation 复制到普通 CPU pages 可能改变 accounting，却没有增加物理 `MemAvailable`，甚至短暂形成第二份权重。故深层 tier 必须是 NVMe 或远端机器。

Spark 不支持 GPUDirect RDMA、nvidia-peermem、DMA-BUF 或 GDRCopy；NVIDIA 建议 verbs 应用使用 host registered memory [11]。远端 expert 传输必须走 ConnectX-7 到 host pages，再进入可计算的 UMA slot。PLLM 的 durable object store 只用于 artifact；高性能 pool 在 71 预注册 volatile host MR。在线 client 由父进程建立匿名共享 mmap，子进程把该 mapping 直接 `ibv_reg_mr`，RDMA READ 落页后只经 stdout 返回 16B status/length，Python 以 memoryview 解析，不再传 payload 或创建 destination file。该路径消除了 C++→pipe→Python payload copy，但仍需 package metadata parse 与 host→GPU/UMA slot copy，故称 **direct shared host MR**，不是 GPUDirect 或 GPU zero-copy。

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

容量包络按硬件 accounting 分开校准，不能把 128GB UMA reserve 原样套到 96GB 独显。当前 RTX 静态投影依次为 idle/full、GPU-pressure/480、creative/448、game/384、memory-pressure/hibernate；它们分别对应 0/3.691/7.383/14.766/74.783GiB checkpoint capacity target，真实 allocator reclaim 仍由 GPU transition 实验验证。

控制器选择动作：

$$a_t=(M_e,h,r,q),$$

其中 `M_e` 为 expert cache budget，`h` 为预取 horizon，`r` 为 token rate，`q` 为 full/elastic/yield/hibernate。目标是最小化后台 TPOT、恢复代价、前台失败风险、能耗与 SSD 写放大，并满足 `MemAvailable`、I/O、compute 和 deadline 约束。

### 3.2 Phase-Constrained Residency 算法

令 `A_{l,t}` 为原始 router 在层 `l`、decode token row `t` 选择的真实 Top-22。fused batch 先保留逐 token rows，再为 actual load 去重；统计不使用 union。每层第 `j` 个完整窗口 `W_{l,j}` 只负责产生排序 `rank_{l,j}`，其命中率与 miss tail 只能在 `W_{l,j+1}` 计算：

$$
h_{l,j\rightarrow j+1}(K)=
{\sum_{t\in W_{j+1}}|A_{l,t}\cap TopK(rank_{l,j},K)|
\over\sum_{t\in W_{j+1}}|A_{l,t}|}.
$$

每个 `(l,K)` 保存最近八个 transition 的最差 hit、mean miss、p95 miss 与 max miss，形成 **empirical held-out envelope**；它不是 conformal 或 `Q_{1-delta}` 保证。request boundary 会丢弃未封口窗口并重置 request-local predictor，绝不把两个短请求拼成虚假的 256-token 连续窗口。至少两个完整窗口前固定 `OBSERVE/FULL`。随后求：

```text
if phase != DECODE: return FULL
if heldout_windows < 1 or horizon unknown: return OBSERVE/YIELD
for each layer l:
    options[l] = validated K_l in {256,...,504,512}
Pareto frontier over 16MiB reclaim buckets:
    cost({K_l}) = sum_l g(max_observed_miss_l)
                + (current_to_target + target_to_full) / H
choose {K_l} meeting capacity, byte-hit, I/O, deadline and TPOT < 5x
if none: return YIELD_OR_HIBERNATE
```

逐层选择避免统一 504 档为释放 0.923GiB 却复制约 58GiB：若只需约 4GiB，可只收缩具有较高 held-out locality 的少数层，其余保持 512。route learning 只允许单个活跃 sequence：EER server 固定 `max-num-seqs=1`，代理序列化请求并拒绝 `n>1`；request boundary 同时清空 decode 与 prefill history。non-full profile 下的新 prefill 返回 503 并保存 replay ID，当前需要用户调用 replay API，尚未实现自动 FIFO。

### 3.3 精确 expert slot cache

每个 MoE 层维护固定物理 slots 和逻辑到物理映射：

```text
logical expert id -> {slot, generation, source, checksum, ready_event}
```

runtime expert object 包含 Marlin 已转换的 NVFP4 packed weights 与 weight scales。当前 W4A16 Marlin 路径在转换时丢弃 per-expert activation input scales，因此对象格式不虚构这两个 tensors。完整 warm-image 装载前执行 SHA-256；decode hot path 依赖 RC 与 64B commit header 校验 key/slot/size，不重复扫描 payload SHA。执行流程为：

1. actual router 产生原始 `A_{l,t}` 并更新 CPU route window；
2. cache hit 直接把 logical id remap 到 slot；
3. 同层 miss 合并为最多 32-object 的持久 QP RDMA batch；逻辑请求超过 32 时通过 iterator 分批，并在复用 shared MR 前完成 package decode 与 H2D；
4. 每个 payload 经 format、fingerprint、tensor layout 和 identity 检查；
5. miss 阻塞该层并写入正确 expert，全部 ready 后发布新 generation；
6. 原始 fused MoE kernel 使用不变的 Top-22 weights 执行。

工作集估计错误只增加 stall、I/O 和 cache pollution，不改变 expert output。缩容在 token-boundary `mode=keep` 下执行，且只重建 `K_l` 变化的层；收缩层以 GPU-to-GPU copy 保留 hot rows。扩容采用 destructive layer rebuild，再从 Remote DRAM/SSD 装入目标 rows，避免 old layer 与 full new layer 同时驻留。全部 mapping ready 后才唤醒。若 backend 不能接受 remap 或任一层失败，runtime `faulted=true` 并保持 quiesced。

### 3.4 KV/Mamba 状态小岛

NemotronH 的 live state 与 expert weights 在生命周期上不同。FP8 Attention KV 约为 4KiB/token，32K 时约 128MiB/request；float16 Mamba conv/SSM state 约 82.34MiB/request。因此单请求上界约 210MiB，两个并发请求约 421MiB，只是 59.063GiB routed experts 的 0.7%。PhaseEER 不为在线 resize 复制这些状态，而是保持 vLLM 原 cache arena 原位。

EER patch 在 `GPUModelRunner.initialize_kv_cache` 后绑定 cache tensors，对底层 storage 去重并记录 `(device,data_ptr,nbytes)` fingerprint。quiesced resize 前后还对每个 storage 的首/中/尾 64B 取样并比较内容 fingerprint；任一差异使 runtime faulted。该 guard 可发现 allocation 变化和部分原位误写，但不覆盖全部内容、block table、token ledger 或 RNG，不证明 Level 2 exact resume。

vLLM 0.25.1 `OffloadingConnector` 已能分别处理 `AttentionSpec` 与 `MambaSpec` cache pages。PLLM 将其作为深度恢复 carrier，但实际 token ledger、sampler RNG 与 connector cache 的端到端 greedy equality 仍待 GPU 故障注入验证。

### 3.5 I/O governor 与算力让渡

expert eviction 释放容量，但 inactive expert 原本不消耗 MoE FLOPs；换出本身不会自动让出算力。当前可执行的算力动作是 vLLM Level-0 `mode=keep` yield；连续 token-rate limiter 尚未实现。planner 输出 compute duty-cycle 与 token-rate 建议，但本文不把建议冒充已执行 governor。正式 Blender 对照必须比较“只缩容量”“只 yield”与二者联合策略；若前台瓶颈是 SM/功耗而非容量，正确动作应是 yield/hibernate。

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

`T_wall` 是用户可见主指标。`T_RDMA` 只统计 verbs post 到 completion，用于判断网络是否瓶颈；warm-image PUT 的 `T_source` 是本地 `.pllmex` 顺序读，direct shared-MR GET 的 destination 不落盘且不传 payload pipe，但仍包含 package parse 与后续 H2D。多 worker 的 `sum(bytes)/max(worker phase)` 只能作为 diagnostic，并与 wall throughput 同时报告。

volatile pool 的一致性来自外部调度契约，而不是新 object-store 算法：PLLM 必须先 quiesce，PUT 与 GET epoch 不重叠，shards 不能拥有相同 slot。RC ordering 保证同 QP 上 commit header 排在 payload 之后。当前 wire format 没有持久 epoch ledger；71 重启后匿名页清零，旧 GET 因 header 不匹配失败，所有 clients 必须重连。该 contract 不支持同 slot 并发覆盖或 crash-consistent remote storage。

### 3.8 Marlin 的边界

Marlin 的 query-agnostic token/head selection 与 I/O-compute overlap 仅用于可复用长前缀 KV 的独立实验组，不进入 live expert routing 或正在生成请求的精确状态。Nemotron 权重已经是 NVFP4，本文不再次量化 routed experts。这样可以避免把近似 KV、expert paging 和事务正确性混成一个无法归因的结果。

## 4. 实现状态

第一版只适配 vLLM 0.25.1。仓库实现 Foreground-QoS 控制面、Level-0 `mode=keep`、Sleep API、OpenAI proxy、PySide6/Vue UI、PhaseEER、HiberCache 和 host-staged RDMA。Marlin runtime object 带模型指纹与 tensor layout；完整导出为 20,480 objects、63,435,912,912 bytes。旧 128-slot runtime 已完成功能性推理；新的 full-prefill→逐层 decode `{K_l}`、destructive expansion 和 shared-MR source 已完成代码、CPU/mock 与跨机数据面测试，在线 GPU 验收 pending。

以下表格区分真实模型/硬件验证、synthetic control plane 和未来工作：

| 组件 | 当前状态 | 完成证据 |
| --- | --- | --- |
| expert catalog 与 trace schema | LIVE | 20,480 runtime objects、抽样 checksum |
| decode-only route tracer | IMPLEMENTED / GPU pending | vLLM 版本守卫、prompt-tail skip、NPZ route artifact |
| PhaseEER route window/planner | IMPLEMENTED / GPU pending | request-local past→next、Pareto frontier/exhaustive test、完整 transition、async fallback |
| resource-envelope planner 与控制 API | LIVE control | 真实 vLLM discovery、hibernate/wake、SSE |
| NVFP4 expert slot manager | LIVE old / new GPU pending | 128 slots 旧基线；GPU-copy resize 与 prefill full guard |
| selective layer resize | IMPLEMENTED / GPU pending | unchanged layers 不搬；shrink GPU copy；expand destructive reload |
| RDMA durable object path | LIVE object path | 20MiB RC PUT/GET、SHA-256；远端落盘 |
| RDMA volatile full pool | LIVE cross-host | 64GiB MR、20,480 objects、stream GET 两端不落盘 |
| RDMA expert source | LIVE host endpoint / GPU pending | direct shared host MR、memoryview package、无 payload pipe |
| live-state SSD/RDMA carrier | LIVE byte carrier | 15MiB snapshot、4MiB chunks、checksum |
| transactional Mamba serializer | 未实现 | 每 token pause 后 greedy token equality |
| full SparkLoad fallback | LIVE local NVMe | Level 2 reload 41.78s；不是论文主优化路径 |

vLLM EER 通过 opt-in `sitecustomize` 在模型构造前保留 512-entry global `expert_map`。默认从 512 physical rows 启动；进入 decode 后才应用逐层 `{K_l}`。执行时 `topk_ids` 是唯一权威来源：逐 token rows 进入 held-out trace，actual experts 仅为 load 去重；runtime 阻塞到正确 rows ready 后才发布 mapping。full-resident hit 不再重复 publish mapping。任一 exact precondition 失败时不调用 kernel。

动态容量变化要求 `--enforce-eager`。PLLM 在 token boundary quiesce，只重建 target 与 current 不同的层。shrink 分配较小参数并 chunked GPU-copy retained rows；expand 先释放旧层，再分配目标参数并从暖源装入全部目标 rows，避免双份 full layer。操作仍是跨层非原子、fail-closed。首次使用前仍需约 59GiB runtime-format export，这是额外 SSD 占用。

远端 tier 不再使用 `ib_write_bw` 代替数据。`pllm-rdma-pool` 在 71 注册 64GiB MR，确定性 slot index 覆盖完整 warm image；PUT 与 GET epoch 分离，payload WR 在前、64B commit header 在后。在线 client 预热持久 QP，父进程创建共享 mmap，子进程直接把它注册为 read destination MR；stdout 只传 descriptor。package decode 使用 memoryview，`verify=false` 不再重复 SHA 扫描；完整 SHA 在装池边界完成，hot path 检查 commit header、模型指纹、identity 与 tensor layout。

RDMA READ credits 为协商后的 16。direct shared-MR 的 100 次 curve 对每个 batch 随 index stride 跨层轮转，最大抽样验证 3,200/20,480 objects；1--32 objects steady p95 进入 planner 的 `g(m)`。逻辑请求超过 32 时使用 chunk iterator，单个 batch 的 views 在下一批覆盖前完成消费。64GiB remote MR/QP 首连约 0.83--1.43s，必须提前 prime。论文不声明 GPUDirect；DGX Spark 用 system `MemTotal`/`MemAvailable` 构造 coherent-UMA capacity，独显使用 NVML VRAM。

## 5. 评估设计

### 5.1 平台与基线

当前先在明确标注的 RTX PRO 6000 + 100Gbps 主机完成可行性验证；最终在 DGX Spark 上复验 Nemotron NVFP4。双机实验使用第二台 Spark 或明确标注的 100Gbps 主机。前台 workload 包括 Blender Benchmark、可重复 graphics trace、NVENC encode 和 NVMe media trace。基线为：

1. vLLM full residency；
2. Level-0 yield；
3. Level-2 drop/reload；
4. kill/restart；
5. 旧 all-phase EER-128/256；
6. decode-only LRU、window-LFU、static past-window 与 offline oracle；
7. uniform K 对逐层 `{K_l}`，以及去掉 phase/horizon/transition/fallback 的消融；
8. HiberFlow-PhaseEER 完整策略。

### 5.2 指标与正确性

报告 detection latency、deadline miss、`MemAvailable` 增量、resident expert bytes、byte hit rate、false-prefetch bytes、blocking I/O、TPOT、前台吞吐/jank、NVMe/RDMA/UMA 带宽、整机功耗、SSD read/write amplification、hibernate barrier、restore time 和 first resumed token。

正确性采用四层检查：

- 每次 MoE 执行的 actual expert ids 与 full-resident baseline 完全一致；
- 每个 expert object weight/scale checksum 与 checkpoint 一致；
- greedy 输出逐 token 相等；
- transaction 在 object write、manifest rename、commit 和 unmap 前后 fault injection，不重复或遗漏已提交 token。

### 5.3 待验证假设

- **H1：decode 存在小而有用的 elastic 区间。** 逐层释放约 1--15GiB 时，总 wall-time 摊销后的 TPOT 低于 5x 且前台获得可测容量收益。
- **H2：phase guard 是必要条件。** 相同 slots 下，prefill-full/decode-elastic 显著优于 all-phase paging，并保持相同 F1。
- **H3：held-out batch tail 优于 bandwidth-only/same-window planner。** 加入逐层 held-out miss 与 shared-MR `g(m)` 后能在 collapse 前拒绝危险组合。
- **H4：phase boundary 能避免 paging collapse。** 当 `tau` 不成立时，自动 yield/hibernate 的 p95 用户体验优于持续换页。
- **H5：深度路径保持 stream continuity。** greedy token ledger 与 uninterrupted baseline 完全一致；遗漏 Mamba state 的对照应稳定失败。

截至 2026-07-19 证据版本，假设状态为：

| 假设 | 当前状态 | 现有证据 | 仍缺少的决定性实验 |
| --- | --- | --- | --- |
| H1 | **待验证；旧大幅缩容被否证** | 逐层静态算术可释放 0.9--14.8GiB；EER-256 >226.09x | 真实 full-route first-50 + online `{K_l}` resize/restore |
| H2 | **机制完成 / GPU pending** | 新请求 prefill 同步 full、并发 phase 聚合与单测 | all-phase vs decode-only LongBench/F1/TPOT |
| H3 | **host 端成本已校准** | direct shared-MR 1/22-object p95 0.477/42.710ms；32 p99 250.9ms | GPU H2D/kernel `g_l` 与真实 route accept/reject |
| H4 | **部分支持 fallback 可行性** | Level 2 使 60GiB allocation 从 OOM 变为成功 | 在相同前台 trace 上比较持续 paging 与跨越 phase boundary |
| H5 | **尚未建立** | Level 0 `keep` 可在原 stream 暂停/继续 | 接入 Mamba/KV/RNG serializer 后的 token equality 与 fault injection |

这些是假设，不是结果。若真实 expert access 缺乏局部性、NVMe 能耗过高、slot remap 开销抵消收益，H1--H4 都可能失败。

### 5.4 当前实测与算法证据边界

当前 PhaseEER 软件回归覆盖 past→next-window、Pareto frontier/exhaustive oracle、逐层 horizon planning、prefill admission、精确 token horizon、route layout、cache replay、retained/destructive resize、state sampled fingerprint、chunked shared-MR framing 与 memoryview package。CMake 验证 CUDA-host client 和无 CUDA 的 71 server build。40 层、7 个非 full options 的 synthetic stress 中，4,775 个 peak states 的同步求解 p50 为 2,294.9ms，不能放在 250ms monitor 关键路径；后台 dispatch 为 1.53ms，求解期间先 Level-0 yield。该 artifact 只测 CPU planner，不是 route 或 GPU 证据。

LongBench QA 对照进一步给出了真实 phase-boundary 反例。全驻留原生 vLLM 在
150.718s 内完成 MQA/NQA/TQA 各 50 条，F1 为 0.3600、总吞吐
6,734.30 tok/s。EER-256 以 68.96% byte hit 运行同一首条 MQA，在 499s
删失点仍未完成，累计换入 358.43GiB；全驻留同请求仅需 2.207s。该结果不把
未完成预测记为错误答案，而把 F1 标为 N/A。新 guardrail 会拒绝相同的
68.96% hit 输入，但是否存在任一真实逐层组合仍待 route 数据。

71 现在以 64GiB anonymous MR 常驻完整 20,480-object image。四个 15.859GB shard 顺序 PUT 共 15.661s，63,435,912,912B 的 source-read+RDMA 有效吞吐 32.40Gb/s；纯 RDMA phase 合计 5.500s、92.27Gb/s。并发 4 QP 注册同一 64GiB MR 会触发 remote invalid request，因此装池使用顺序 QP，这不进入在线 miss。

持久 GET 不写 71 或 75 文件。旧 pipe 路径单 expert p50/p95 3.89/8.08ms，22-object 87.06/146.92ms。direct shared-MR 让 RDMA 直接落到父进程可见页；100 次 strided 测量的 1/2/4/8/16/22/32-object p50 为 0.447/0.745/1.386/2.716/14.028/15.807/21.288ms，p95 为 0.477/0.784/1.513/2.863/26.223/42.710/43.897ms，steady throughput 26.8--72.4Gb/s。32-object p99 250.925ms 表明长尾仍不可忽略。GPU H2D/Marlin TPOT 未测。

桌面链路完成了 Blender 5.2 LTS、RTX PRO 6000 OptiX 枚举、程序化 Cycles 场景、GNOME Shell 42 focus bridge 和 PLLM 250ms monitor 的集成；preflight 能返回真实 `blender_blender.desktop` 与 PID。仅运行了 800x450、16-sample 预览，没有执行 full-resident/PLLM/无后台三组正式 render，因此本文不报告 Blender 时间、samples/s 或前台 90% 指标。

### 5.5 关键路径对齐与可复现协议

下表不把不同对象、sink 和硬件上的带宽直接相除。所有系统结论以调用者观察到的 wall time 为主：

| 路径 | 对象与字节 | wall time | 终点语义 | 可支持结论 |
| --- | --- | ---: | --- | --- |
| vLLM Level 2 release | 44.17GiB resident allocation | 0.185s | 释放 CUDA allocation | 前台 capacity admission，不是 239GiB/s 存储吞吐 |
| vLLM Level 2 restore | 74.8GiB checkpoint + runtime rebuild | 41.39s | 恢复可接受请求的 engine | 完整冷恢复代价 |
| full warm-image PUT v2 | 63,435,912,912B / 20,480 objects | 15.661s sequential 4 shards | 71 64GiB volatile MR | 32.40Gb/s wall；92.27Gb/s RDMA phase |
| pipe GET single / batch-22 | 3.1 / 68.1MB | p50/p95 3.89/8.08ms；87.06/146.92ms | Python bytes | 旧 payload-pipe 基线 |
| shared-MR GET single | 3.1MB | p50/p95/p99 0.447/0.477/0.656ms | Python memoryview | host source latency，不含 H2D |
| shared-MR GET batch-22 | 68.1MB | p50/p95/p99 15.807/42.710/59.398ms | persistent QP + shared registered pages | p95 planner calibration |
| shared-MR GET batch-32 | 99.1MB | p50/p95/p99 21.288/43.897/250.925ms | 同上 | 证明 tail fallback 必要 |

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
| R3 cross-host remote pool | `results/eer-memory-profile-full.tsv`、`results/rdma_stream_curve/direct-shared-*.json` | full pool PUT + direct shared-MR GET |
| PhaseEER route experiment | `results/decode_residency/`（pending） | `scripts/benchmark_decode_routes.py` |
| foreground mechanism | `/home/cong/hackathon/blender_demo/README.md` | Blender preflight + GNOME focus bridge |
| exact resume 缺口 | `/api/v1/capabilities` 的 `serializer_attached=false` | capabilities API 与 live-state tests |
| LongBench QA 与 paging collapse | `results/qa_benchmark/` | `scripts/benchmark_longbench_qa.py` |

旧证据快照以 commit `24a7dc6` 为准；PhaseEER 将产生新的 route、resize 与 Blender artifact，不覆写旧 EER 负面结果。

## 6. 相关工作与创新边界

MoE-Infinity [3] 使用 batch=1 activation trace 管理 expert cache；ProMoE [4]、ExpertFlow [5]、Fate [6] 和 pre-attention prediction [7] 预测未来 experts；SpecMD [12] 说明 LRU/LFU 的时序假设不可靠并提出 Least-Stale；FlashMoE [13] 面向 SSD cache；ActiveEvict [8] 做动态 budget 与 pre-eviction；OD-MoE [14] 甚至在多节点上按需加载。故本文不声称首次预测、预取、SSD offload 或动态 budget。

与这些工作的差异不是“首次预测 expert”。PhaseEER 的候选贡献是：前台突发容量 SLO 下，prefill 不可缩；decode 只能用 past→next held-out envelope，并联合求逐层 `{K_l}`、transition/horizon 与 hibernation boundary。当容量目标只能靠超 SLO paging 达成时，算法选择**不 offload**。这是否超过 A+B+C 取决于真实 Pareto 与对照，本文尚不预判。

FastServe [15]、ConServe [16] 和 Sereno [1] 关注细粒度 preemption 或带宽干扰；ServerlessLLM [17]、MAIO [18] 优化模型恢复；Sparse Prefix Caching [19] 研究 recurrent state checkpoint。本文把这些能力放入同一个 time-varying resource envelope，但是否形成足够强的系统贡献取决于真实 EER 和 transactional state 实现，而不是架构图本身。

## 7. 限制与否证条件

1. 当前风险项是最多八个 past→next transitions 的最差经验值，不是统计 `Q_{1-delta}`。runtime 累计 remote GET、package parse、H2D 与 mapping wall debt，超过 resize 时下发的 per-token budget 后由 controller yield/hibernate；该机制尚待 GPU fault-injection 验证。
2. expert paging 不减少原始 Top-22 的 FLOPs；让出算力依赖 decode duty cycle。只做 cache shrink 可能让后台更慢却仍干扰前台。
3. NVMe 读取可能显著增加能耗和与创作应用的存储竞争；缓存命中率提高不等于用户体验提高。
4. 旧 128-slot all-phase runtime 要求 `max_num_batched_tokens<=5`；新路径以 512 slots 完成 prefill，并只在 `max_num_seqs<=2` 的 decode 缩容。真实在线转换尚未验证，绕过 PLLM proxy 直接向缩容 vLLM 发 prefill 不受支持。
5. 双源恢复共享 UMA，理论链路带宽不能相加后直接作为恢复速度。
6. 首次 transformed-expert export 需要完整模型驻留并额外使用约 routed-weight 规模 SSD；shared-MR 消除了 payload pipe，但仍有 route `.cpu().tolist()`、metadata parse 和 H2D。
7. 逐层 resize 尚非跨层原子事务；shrink 有单层 old/new 短时双份 allocation，expand 使用 destructive reload 避免 full 双份，但失败会保持 quiesced。真实 wall time/释放量待测。
8. volatile RDMA pool 假设 phase-separated epochs；同 slot 并发 PUT/GET 不受支持，71 重启会丢失全部对象。64GiB MR 多 QP 并发注册失败，当前完整装池只能顺序完成。
9. Blender demo 目前只证明可重复场景、OptiX backend 和 foreground PID 检测链路；没有证明策略检测延迟、渲染吞吐、功耗或恢复后 TTFT。

核心假设在以下任一结果下被削弱：不存在比 full residency 和 hibernate 更优的 elastic Pareto 区间；保持 95%以上 byte hit 所需 cache 接近完整 expert 权重；SSD/RDMA traffic 使前台低于无后台基线 90%；slot miss 无法在 deadline 内回退；或 transactional recovery 不能保持 greedy token equality。

## 8. 结论

HiberFlow-PhaseEER 当前是 measurement-grounded、可证伪的 prototype。旧 all-phase EER 的 >226x 反例促成 past→next-window、逐层 `{K_l}`、transition/horizon 与 hibernation boundary；direct shared host MR 把 remote warm image 的 steady source 提升到 26.8--72.4Gb/s，同时暴露 250.9ms p99 tail。系统尚未证明任何 `{K_l<512}` 优于 full/hibernate，也未完成 exact deep resume、Blender 因果对照或 DGX Spark UMA。论文能否成立取决于真实 route、online resize/restore 与 foreground QoS，而不是代码规模。

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
