# LongBench QA 开关 PLLM 对照实验

> 实验日期：2026-07-19。GPU 为 RTX PRO 6000 Blackwell 96GB，模型为
> `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`，推理框架为 vLLM 0.25.1。
> 本文只报告已经落入 `results/qa_benchmark/` 的实测数据。

## 1. 问题与口径

实验分别取 `mqa`、`nqa`、`tqa` 的前 50 条，共 150 条。三组都使用原始
prompt、greedy decoding（`temperature=0`、`seed=0`）、并发 2；最大输出长度
依次为 64、128、32 token。正式计时前执行一次不计分 warmup。F1 使用
LongBench 英文 QA 口径：小写、去标点和
冠词后计算 token overlap，并在多个参考答案中取最大值。

“开启 PLLM”不是一个单一开关，本实验拆成两个可辨识因素：

1. **控制面与 HiberCache**：是否运行 PLLM monitor，以及是否接入 KV
   `OffloadingConnector`；
2. **真正释放容量的数据面**：是否启用 vLLM Sleep Mode 和 EER 动态专家驻留。

只有第二项能在前台任务出现时释放大量 GPU 容量。把第一项的低开销结果当作
第二项的性能，是无效对照。

## 2. 完整全驻留基线

配置为 512 个专家全驻留、Marlin MoE、FP8 KV、32K context、
`gpu_memory_utilization=0.85`、`max_num_batched_tokens=8192`，关闭 PLLM、
Sleep Mode、EER 和 HiberCache。150/150 请求全部完成，无错误。

| 数据集 | 样本 | F1 | wall time | total tok/s | output tok/s | latency p50/p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MQA | 50 | 0.5461 | 50.681s | 7,155.88 | 11.68 | 1.722/3.904s |
| NQA | 50 | 0.2625 | 61.813s | 7,111.85 | 10.97 | 2.302/5.024s |
| TQA | 50 | 0.2713 | 38.225s | 5,564.83 | 41.86 | 1.517/1.818s |
| **合计** | **150** | **0.3600** | **150.718s** | **6,734.30** | **19.04** | - |

合计处理 1,012,113 prompt token 和 2,870 completion token。`total tok/s`
主要衡量长上下文 prefill，不能替代 decode throughput，因此同时保留 output
tok/s。基线 GPU 峰值为 85,630MiB，三个数据集的估算能耗合计 12.36Wh。

## 3. 可释放 PLLM 配置

### 3.1 全专家 + Sleep Mode：启动失败

全 512 专家、HiberCache 和 Sleep Mode 同时启用时，vLLM 的 CuMem allocator
在 Marlin 权重重排阶段使初始化峰值超过 96GB。两次配置尝试均在已有
94.91GiB GPU 占用、仅余 3MiB 时无法再分配 20MiB，故无法进入服务状态。

这是容量约束，不是质量回归。该组没有产生预测，因此 F1 和吞吐均为 `N/A`。
证据见 `results/qa_benchmark/pllm_full_sleep_startup_failure.json`。

### 3.2 EER-128：容量大、吞吐不可用且试跑受干扰

128 slots/layer 的活动态约为 45,074--46,068MiB；Level 2 将其降至
860MiB，回收 44,214MiB，休眠 0.161s、恢复 36.769s。由于精确 Top-22
要求 `22B<=128`，该配置只能令 `max_num_batched_tokens=5`。

第一条 MQA 在 431s 内没有完成，且期间被自动策略休眠，属于受干扰的删失
试跑，只能说明端到端配置不稳定，不能用于纯 EER 吞吐估计。

### 3.3 EER-256：可控试跑证实 paging collapse

256 slots/layer 是 PLLM planner 在本机资源包络下的建议配置。为排除 daemon
干扰，Level 2 恢复后停止自动控制器，再单独运行第一条 MQA：

| 指标 | EER-256 实测 |
| --- | ---: |
| 活动态 / 休眠态 GPU | 74,578MiB / 860MiB |
| Level 2 回收 | 73,718MiB |
| 休眠 / 恢复 | 0.210s / 37.375s |
| 首条 MQA 观察窗口 | 499s，未完成 |
| 同一请求全驻留延迟 | 2.207s |
| slowdown lower bound | >226.09x |
| expert hit / miss | 276,173 / 124,287 |
| byte hit rate | 68.96% |
| 累计换入 | 358.43GiB |
| expert load 累计时间 | 501.31s |

一条只有 1,738 prompt token 的请求在完成前已搬运模型 checkpoint 约 4.79 倍的
专家数据。继续跑满 150 条预计需要多日且不会增加统计解释力，因此按预注册的
失败条件停止。该组的 F1 是 `N/A`，不是 0；在 499s 删失窗口内 completed-request
throughput 为 0，延迟下界为全驻留同请求的 226.09 倍。

恢复还暴露了第二个实现缺口：Level 2 reload 对 EER 过滤后的 routed expert
报告加载失败，服务虽然恢复为 ready，但 expert slots 实际为冷状态，首请求必须
重新填充。当前不能把 `data_plane_ready=true` 等同于热工作集已经恢复。

### 3.4 控制面 + HiberCache 控制组：未运行

为隔离 PLLM monitor 与 KV connector 本身的开销，实验尝试启动“全 512 专家、
daemon/HiberCache 开启、Sleep/EER 关闭”的控制组。正确配置进入 engine 初始化
时，另一用户的非 PLLM GPU 作业已占用 77,296MiB，只剩 18.89GiB，无法装入
全驻留模型。该组标记为 `not_run_external_gpu_busy`，不参与任何性能比较。

因此本轮不能给出“仅开启控制面”的开销比例。这个缺口不影响 EER paging
collapse 的同请求下界，但必须在 GPU 获得独占时间后补跑，才能完整分解
monitor、HiberCache、Sleep allocator 和 EER 四项开销。

## 4. 存储空间

| 项目 | bytes | GiB | 说明 |
| --- | ---: | ---: | --- |
| 共享只读模型 | 80,365,697,343 | 74.846 | 所有组共同需要 |
| EER runtime experts | 63,435,913,177 | 59.079 | 20,481 个 manifest/object 文件 |
| EER 部署合计 | 143,801,610,520 | 133.926 | 相对基线增加 78.93% |
| HiberCache 实验前 | 2,320 | 0.000002 | 空 cache/metadata |
| HiberCache 当前逻辑文件 | 2,320 | 0.000002 | 未增长；`du -sb` 含目录项为 10,512B |

HiberCache 是按需增长、20GiB quota 的运行时 cache，不能把 quota 写成已占用量。
EER 的 59.079GiB 是当前 Marlin 随机访问格式的真实额外占用，也是明显的部署
代价；后续应改为 safetensors positional read 或无复制重排格式。

## 5. 结论

本轮结果不支持“开启 PLLM 后 F1 不变且吞吐接近基线”这一强主张。准确结论是：

- 全驻留原生 vLLM 完成 150 条，F1 为 0.3600、总吞吐 6,734.30 tok/s；
- 全驻留 Sleep Mode 在当前 96GB 机器上无法启动；
- EER-256 确实可把活动态 GPU 容量减少约 9.8GiB，并可经 Level 2 快速回收
  73,718MiB，但 68.96% byte hit 远低于可用区间，进入灾难性换页；
- EER-128 释放更多活动态容量，却因 batch cap 和更小工作集进一步恶化吞吐；
- 当前 PLLM 的可靠价值是快速 foreground admission；动态专家驻留尚不是可用的
  长上下文推理模式。

这个负面结果收紧了论文假设：EER 必须在在线 miss-debt 超过 I/O budget 时立刻
退出到 yield/hibernate，并且需要 layer-pipelined gather、热工作集恢复和更高
byte hit，才可能形成有效 Pareto 区间。

## 6. 复现

```bash
conda activate pllm
python scripts/benchmark_longbench_qa.py \
  --mode native_full --limit 50 --concurrency 2 --overwrite

python scripts/compare_longbench_qa.py \
  --baseline results/qa_benchmark/native_full/summary.json \
  --candidate results/qa_benchmark/<candidate>/summary.json \
  --output results/qa_benchmark/comparison.json
```

原始逐样本 prediction、reference、F1、token 数和延迟位于
`results/qa_benchmark/native_full/*.jsonl`。输入数据只记录 SHA-256，不复制进
结果目录；`test_data/` 保持为本地未跟踪数据。
