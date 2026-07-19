# Iteration 5 Round 2 审稿意见

**结论：Reject / Major Revision**  
**评分：4/10；审稿信心：5/5**  
**创新性：3/5；正确性：2/5；实验完整性：2/5；证据表达：4/5**

## 总评

当前版本已经从固定统一 K 的专家换页原型，推进到 past-to-next route estimation、逐层 `{K_l}`、horizon-aware planning、destructive expansion 和 direct shared host-MR。方向合理，软件回归与 C++ 构建均通过，RDMA host endpoint 也有真实跨机数据。

但中心算法仍存在可复现的 DP 错误，horizon 和 tail latency 的定义不能支撑 `<5x` SLO，planner fallback 尚未闭环执行。direct shared host-MR 的实验成立于 host memory 终点和单批场景，不能外推为 GPU slot 或 full expansion 的 no-copy。当前仍不具备顶会接收所需的算法正确性和中心实验。

## 已解决问题

1. **past→next 基本成立。** 上一窗口生成排名，下一完整窗口才记录 hit/miss；逐 token Top-22 rows 不再被 fused-batch union 替代。

2. **逐层容量选择已经实现。** planner 可以让部分层保持512、部分层缩至256--504，比统一 K 更符合层间路由差异。

3. **transition进入目标函数。** shrink copy、future expansion、kernel rebuild和remaining horizon均已有代码表示。

4. **destructive expansion存在。** 扩容路径先删除旧层参数并回收，再分配目标参数和重载source，设计上避免old+expanded layer峰值。

5. **prefill安全门存在。** non-full profile下，存在前台压力或活跃请求时不会无条件扩回512。

6. **shared-MR数据面真实成立。** RDMA READ直接写入父子进程共享并注册的mmap，单批时Python得到memoryview，payload不再经过stdout。

## Critical Issues

### C1. Bucket DP不是正确的约束优化

每个reclaim bucket只保留最低`objective_ms`状态，但hit rate、I/O、deadline和精确reclaim均在DP结束后检查。最低objective并不支配同bucket的其他状态。

可复现反例：

```text
full slots = 4
bucket = 64 bytes
target reclaim = 40 bytes

K=3: reclaim=25, objective=0
K=2: reclaim=50, objective=1
```

两者进入同一个bucket，代码保留K=3，最终因回收不足返回`yield`，尽管K=2是可行解。

**必须修改：**

- 每个bucket保存Pareto frontier，而不是一个状态。
- 支配条件至少包含`reclaim >=`、`objective <=`、`mean_miss <=`、`immediate_time <=`。
- 用小规模exhaustive solver验证随机实例。
- 若仅保留启发式实现，论文不得称其为multiple-choice DP或可行组合求解器。

### C2. Release deadline漏算扩容动作

当前`immediate_seconds`只统计shrink copy和缩小层rebuild。若planner从异构profile出发，同时扩展高miss层、缩小高局部性层，destructive expansion的source reload和rebuild不进入release deadline。

保持现有elastic K的层未来恢复full时，代码计入了完整layer bytes，却没有计入future rebuild。

**必须修改：**

```text
T_now =
  Σ shrink_copy_bytes / B_gpu_copy
+ Σ destructive_expand_bytes / B_source_to_gpu
+ changed_layers * T_rebuild

T_future =
  Σ_{K_l < 512} full_layer_bytes / B_source_to_gpu
+ elastic_layers * T_rebuild
```

当前release-deadline检查可能false accept，属于正确性问题。

### C3. `sum_l g(p95_miss_l)`不是总p95

逐层p95之和不等于token stall的p95，也不是统计上界。32-object direct-MR的p99达到250.9ms，进一步说明p95 curve不能称为“上包络”。

**必须修改：**

在真实逐token route上直接计算：

\[
T_t(\mathbf K)=\sum_l g_l(m_{l,t}(K_l))
\]

然后报告`p50_t/p95_t/p99_t(T_t)`。若需要风险约束，使用经验CVaR、bootstrap upper confidence bound或明确的union-bound quantile。

### C4. Horizon不是剩余token数

当前horizon来自：

```text
max_tokens - SSE content chunk count
```

`max_tokens`只是上限，SSE chunk也不等于token。该估计会高估可摊销时间，使短请求在即将EOS时仍可能触发昂贵resize。

**必须修改：**

- 从engine内部读取generated token count。
- 使用EOS hazard或held-out remaining-length predictor。
- planner使用remaining-horizon lower confidence bound，而不是max-token upper bound。
- 无可靠horizon时保持full或yield，不允许elastic。

### C5. Planner fallback没有控制闭环

decode planner返回`yield`时，`_maybe_auto_resize()`只是返回false。最终是否yield仍由独立`PolicyEngine`决定；若其action为`none`，算法会继续运行。

**必须修改：**

将planner输出变成统一仲裁输入：

```text
if capacity requires release:
    if decode_plan feasible:
        resize
    elif capacity requires deep release:
        hibernate
    else:
        yield
```

论文中的“不可行即yield/hibernate”在完成该闭环前不成立。

## Major Issues

### M1. DP复杂度与控制延迟

生产规模约为：

```text
L = 40 layers
C = 8 options
B ≈ 59.063GiB / 16MiB ≈ 3780 buckets
```

名义复杂度为`O(LBC)`，但当前内层重复计算总bytes并复制完整record tuple，实际接近`O(L²BC)`，内存为`O(BL)`。本机完整规模合成输入约需0.210s，已经接近250ms监控周期。

**修改建议：**

- 预计算最大bucket。
- 使用parent pointer，不复制完整tuple。
- planner放入独立线程，旧plan在新plan完成前保持有效。
- 报告planner p50/p95和最大state count。

### M2. Past-to-next跨请求统计不明确

runtime和offline工具都可能把多个短请求拼入一个window。结果依赖数据集顺序，也混合了within-request locality与cross-request domain shift。

**修改建议：**

分别实验：

- request-local windows；
- chronological cross-request windows；
- 随机请求顺序；
- MQA→代码生成、代码→TQA等domain shift。

论文必须明确rank是否还使用当前prompt的prefill tail。

### M3. Runtime和offline窗口不一致

论文/runtime默认256 rows，`benchmark_decode_routes.py`默认64。二者不能共享结论。

**修改建议：**

统一为256，或将64/128/256作为消融并报告反应延迟、hit rate和planner稳定性。

### M4. Prefill queue仅有安全性，没有liveness

当前实现返回503并将请求标记`queued`，但不会在空闲后自动重试。用户必须手工调用replay API。

**修改建议：**

增加真实FIFO admission queue、取消、超时和单次执行语义；否则文稿应改成“拒绝并保存为可手工重放请求”。

### M5. Shared-MR no-copy仅适用于单批

当`get_many()`超过32 objects时，为避免staging复用覆盖，代码会将memoryview转换为`bytes`。因此：

- full expansion的512 objects不是no-copy；
- 双请求最坏44 misses也会触发copy；
- 当前`g(m>32)`与实际多批copy路径不一致。

**修改建议：**

提供chunk iterator，让每个≤32批次完成decode和H2D后再复用staging；或使用双缓冲/ring MR。分别校准1--44和512-object路径。

### M6. Destructive expansion仍缺GPU证据

CPU/mock测试只能证明调用顺序，不能证明CUDA allocator及时归还旧层，也不能证明无OOM。

必须测量：

- 512→异构`{K_l}`→512；
- 每层和整体wall time；
- peak allocated/reserved bytes；
- source read、package parse、H2D、rebuild；
- 失败后的faulted/quiesced状态。

## Direct Shared Host-MR评价

可以接受的主张：

- 远端是64GiB volatile registered MR。
- steady GET不访问71或75本地文件系统。
- 单批RDMA payload不经过stdout。
- 1/8/22/32-object steady p95分别为0.477/2.863/42.710/43.897ms。
- 当前终点是host memory，不是GPU，不是GPUDirect。

不能接受的主张：

- “跨完整index验证”：单组最多抽样3,200/20,480 objects。
- “消除Python package copy”：仅单批成立。
- “支持快速full-prefill restore”：没有512-object→GPU测试。
- “p95是SLO上包络”：32-object p99为250.9ms。
- “完整warm-image PUT可复现”：当前树缺少支持15.661s、32.40/92.27Gb/s的原始PUT JSON。

## 证据边界

**Validated**

- 20,480 objects、63,435,912,912 bytes的runtime index。
- direct shared host-MR跨机sampled GET。
- sampled byte equality和package validation。
- 旧EER-256 paging collapse。
- Level-2容量释放和前台allocation admission。

**Mechanism Only**

- HorizonAwareLayerPlanner。
- destructive expansion。
- per-layer resize。
- sampled state-island guard。
- prefill rejection。
- shared-MR expert source接入。

**Not Established**

- 任一`K_l<512`的可用Pareto区间。
- RDMA→GPU slot端到端收益。
- Blender/游戏QoS。
- 自动prefill queue。
- exact KV/Mamba resume。
- DGX Spark UMA收益。

## 必须新增实验

1. DP与exhaustive oracle的property testing。
2. 真实route下token-level total-stall p95/p99。
3. 256-token窗口和512--2048-token长生成。
4. 100轮GPU shrink/expand及state/token equality。
5. 1--44 miss和512-object expansion端到端分解。
6. local NVMe、pipe、shared-MR和GPU slot对照。
7. no-LLM、full、Level-0、Level-2、fixed-K、PhaseEER独立Blender组。
8. prefill FIFO、公平性、取消和自动重放。
9. DGX Spark上的`MemAvailable`、PSI、功耗和渲染吞吐。

## 文稿必须修改

- 将DP描述降为“bucket heuristic”，直至剪枝正确性修复。
- 将`T_miss^95`改为per-layer heuristic，不得称总p95或上界。
- 将“跨完整index”改为“从完整index进行strided抽样”。
- 将“自动进入replay queue”改为“503拒绝并保存手工replay记录”。
- 将no-copy限定为单批`<=32`。
- 删除“GPU-row resize已验证”，改为“GPU implementation pending”。
- 对齐runtime 256与offline 64的窗口口径。
- 修正代码状态中的`strictly_below_10x`，使其与论文`<5x`一致。
- 为15.661s full PUT补充原始artifact，否则删除该结果。
- 明确所有RDMA latency均不含package-to-GPU和kernel stall。

## 最终判断

Round 2的方向比上一版明显更接近可发表系统：逐层容量、past→next统计和direct shared host-MR形成了一个可验证的研究对象。但planner尚未满足基本正确性，horizon和tail风险定义不成立，核心GPU/QoS实验为空。

在修复DP、闭环fallback并完成真实route与GPU transition前，该工作仍应拒稿。
