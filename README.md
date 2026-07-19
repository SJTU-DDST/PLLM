# PLLM HiberFlow-EER

PLLM HiberFlow-EER 是面向 DGX Spark 与 NVIDIA 桌面 AI 工作站的前台感知 vLLM 资源运行时。它在 Blender、游戏、视频编码或系统内存压力出现时，按前台资源包络在完整驻留、精确的弹性专家驻留、微暂停和事务式深度休眠间选择。

第一版只控制 vLLM。PLLM 不暂停或终止训练任务、未知 CUDA 进程，也不会控制未开放 Sleep API 的外部服务。

当前已在 RTX PRO 6000 96GB 上实测 vLLM 0.25.1 ModelOpt NVFP4/Marlin 专家数据面：20,480 个 checksummed runtime objects、128-slot actual Top-22 blocking load、Level 1/2 释放恢复，以及 75→71 的 64GiB remote warm image。在线 GET 由 RDMA 直接写入父进程共享 host MR，不经过远端磁盘、客户端落盘或 payload pipe；H2D/UMA slot 与真实 decode 性能仍待 GPU 空闲后验收。

![PLLM Web control center](results/pllm-dashboard-live-desktop.png)

## 核心贡献

- **Foreground-QoS Agent**：每 250ms 融合 GNOME 焦点、NVML 进程级 SM/NVENC/NVDEC、显存、功耗、PSI、`MemAvailable` 与供电状态；经本机校准的成本函数在 `yield` 和 `hibernate` 间选择，并公开每项代价。
- **Phase-Constrained Elastic Expert Residency**：prefill 强制全驻留；decode 用过去 256-token request-local window 预测并在下一完整 window 验证，逐层选择 `K_l`。规划器联合约束 held-out miss tail、收缩/扩容、kernel rebuild、由 `min_tokens` 和精确 token IDs 证明的剩余 horizon，以及 `<5x` TPOT SLO；预测 miss 必须加载正确 Top-22 expert，不可行时 yield/hibernate。
- **HiberCache**：使用 vLLM `OffloadingConnector + TieringOffloadingSpec`，以 512MiB host staging 和 `/mnt/ssd-storage/pllm-cache` 文件层保存可复用 KV block。vLLM 0.25.1 版本守卫补丁在深度 `mode=keep` 时保留 connector cache；缺失的混合模型状态由 token 重算恢复。
- **SparkLoad**：Level 2 不重复写出约 75GB 不可变权重；直接从共享模型目录使用 `fastsafetensors` 恢复。GB10 选择 unified copier，独显可选择 GDS。
- **DGX Spark RDMA fallback**：能力探测禁止在 GB10 上声明 GPUDirect RDMA。高性能 `pllm-rdma-pool` 在 71 预注册 64GiB volatile host MR；计算节点持久 RC QP 的 RDMA READ 直接落入父进程共享 host MR，Python 以 `memoryview` 消费。该路径不落盘且没有 payload pipe，但仍需 host-to-GPU/UMA copy，因此不宣称 GPUDirect。
- **事务式 live-state carrier**：KV/Mamba/sampler/ledger bytes 按 64MiB 分块，SSD directory 与远端 manifest 均最后提交；当前 serializer 尚未接入 vLLM/NemotronH，状态明确为 carrier implemented / exact resume pending。

## 状态机

```text
FULL_RESIDENT <-> ELASTIC_RESIDENT -> YIELDING -> HIBERNATED -> RESTORING
```

- `YIELDING`：vLLM Level 0、`mode=keep`，scheduler 停在 token 边界，HTTP 流保持连接，GPU cache 保留。
- `HIBERNATED`：Level 1/2、`mode=keep`。独显且主存宽裕时可用 Level 1；DGX Spark UMA、游戏、低电量或内存压力使用 Level 2。
- `RESTORING`：Level 2 依次恢复 weights、执行 `reload_weights`、恢复 KV cache，再开放 scheduler。
- `abort` 只用于错误恢复，不是默认抢占路径。

## 环境

环境固定为 Python 3.12 conda，不使用 uv：

```bash
cd ~/PLLM
bash scripts/setup_conda.sh
conda activate pllm
```

当前验收版本：vLLM 0.25.1、Torch 2.11、fastsafetensors 0.3.3、PySide6 6.11。模型只读复用，不下载、不复制：

```text
/mnt/ssd-storage/shared_models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

为 HiberCache 创建专用目录：

```bash
sudo install -d -m 0750 -o "$USER" -g "$(id -gn)" /mnt/ssd-storage/pllm-cache
install -d -m 0750 /mnt/ssd-storage/$USER/pllm-experts
```

## 无 GPU 演示

当前 GPU 忙碌时使用 mock 验证完整控制面：

```bash
conda activate pllm
python scripts/mock_vllm.py --port 18000
PLLM_CONFIG=$PWD/tests/fixtures/integration-config.toml python -m pllm.daemon
```

打开 [http://127.0.0.1:17861](http://127.0.0.1:17861)。控制中心为 Flask 静态托管的 Vue 3 应用，不含 Node.js 构建链；“评委模式”中的回放数据始终显示为历史回放。

悬浮窗：

```bash
python -m pllm.desktop --api-base http://127.0.0.1:17861
```

## 真实模型

仅在 GPU 空闲后运行：

```bash
bash scripts/run_vllm.sh
bash scripts/run_daemon.sh
bash scripts/run_desktop.sh
```

`run_vllm.sh` 使用 NVFP4 + Marlin、FP8 KV、32K context、HiberCache 与 safetensors loader。原生模式最多 2 个请求；PhaseEER 固定 `max-num-seqs=1`，代理拒绝 `n>1`，避免并发污染 request-local route generation。模型端口只绑定 `127.0.0.1`；vLLM 的开发控制端点不得暴露到局域网。

OpenAI 客户端应指向 PLLM 代理：

```text
http://127.0.0.1:17860/v1
```

### Elastic Expert 数据面

仅在 GPU 空闲后首次导出 Marlin runtime experts：

```bash
bash scripts/run_vllm_export_experts.sh
python scripts/eer_runtime_ctl.py status
```

确认 `runtime-manifest.json` 包含 20,480 objects 后，可先从 full slots 启动：

```bash
PLLM_EER_SLOTS_PER_LAYER=512 bash scripts/run_vllm_eer.sh --enable-return-routed-experts
```

该路径强制 vLLM 0.25.1、ModelOpt NVFP4、Marlin modular backend、lazy safetensors 与 eager execution。当前实测导出 20,480 objects、63,435,912,912 bytes，约 308.85 秒。新策略不再全阶段固定 128 slots：prefill 必须 512；decode 只有积累至少两个完整窗口、逐层规划满足容量和 `<5x` SLO 时才收缩，否则保持 full 或 yield。旧 EER-128/256 结果作为 paging-collapse 反例保留。

## RDMA 数据面

```bash
cmake -S rdma_bridge -B rdma_bridge/build -DCMAKE_BUILD_TYPE=Release
cmake --build rdma_bridge/build -j
python scripts/rdma_benchmark.py --allocator aligned --device mlx5_0
```

双机测试在对端先运行 `python scripts/rdma_benchmark.py --server`，本机再加 `--peer <IP>`。输出将 host staging 与网络 RDMA write 分开写入 `results/rdma_bench.json`。

真实 expert profile 的 volatile pool 使用 `rdma_bridge/build/pllm-rdma-pool` 与 `scripts/run_rdma_memory_shards.py`，完整命令见 `rdma_bridge/README.md`。71 已容纳完整 20,480-object、63,435,912,912B image。从完整 index 做 strided sampling 的 100 次 GET 中，1/8/22/32-object steady p95 为 0.477/2.863/42.710/43.897ms，steady throughput 为 55.5/72.4/27.3/29.0Gb/s；最大组抽样 3,200/20,480 objects，32-object p99 250.9ms。逻辑请求超过 32 时使用 chunk iterator；以上仍不含 H2D 和 Marlin stall。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/v1/status` | 状态机、传感器、成本决策 |
| GET | `/api/v1/capabilities` | UMA、loader、HiberCache、RDMA 能力 |
| GET | `/api/v1/telemetry/stream` | SSE 实时遥测 |
| GET | `/api/v1/vllm` | 服务发现与可控性 |
| GET | `/api/v1/events` | 决策与恢复记录 |
| GET | `/api/v1/replays` | 请求、token 位置与恢复状态 |
| GET | `/api/v1/experiments` | 实验指标 |
| GET | `/api/v1/expert-residency` | Expert catalog、当前投影与证据边界 |
| GET | `/api/v1/expert-dataplane` | vLLM 进程内 slot/SSD/RDMA 实时状态 |
| PUT | `/api/v1/policy` | 更新模式与阈值 |
| POST | `/api/v1/policy/compile` | 自然语言偏好编译为受限规则 |
| POST | `/api/v1/actions` | `yield`、`hibernate`、`wake`、`benchmark` |
| POST | `/api/v1/expert-residency/plan` | 计算资源包络建议，不修改 vLLM 权重 |
| POST | `/api/v1/expert-dataplane/actions` | `resize`、`prefetch`、`evict`、`evict_all` |
| POST | `/v1/chat/completions` | OpenAI 兼容代理 |

## 已验证边界

```bash
pytest
```

- 完整回归为 `108 passed`；shell syntax、CMake 常规构建与无 CUDA 的 71 server 构建通过。
- 真实 Level 2 在 0.131--0.185 秒回收约 43--44GiB，恢复约 39--42 秒；恢复后代理请求 HTTP 200。
- 60GiB CUDA allocation 从模型常驻 OOM 变为 Level 2 后成功；该结果是显存 admission，不是游戏或 Blender 吞吐。
- `mlx5_0` 真实完成 20MiB durable RC RDMA PUT/GET；75→71 的 64GiB warm image 与 direct shared host-MR GET 已跨机实测。终点是 host memory，不是 GDR 或已完成的 GPU slot refill。
- 16MiB `cudaHostAlloc` staging 为 127.338Gb/s host copy、422.718us MR 注册；该值不是网络 RDMA bandwidth。
- 实时桌面 1440×1000、移动 390×844 和 PySide6 370×535 截图通过。
- 20 秒稳态守护进程采样为 0.80% CPU、67MiB RSS；GPU 快采样周期仍为 250ms。
- Level 0 同一 stream 暂停期间无 chunk 且无需重连，但跨独立请求的逐 token 等价尚未证明。

## 文档

- [项目报告](docs/PLLM项目报告.md)
- [最新六页论文设计稿](paper/HiberFlow-ACM六页稿.md)
- [四轮审稿与答辩记录](docs/reviews/四轮审稿迭代记录.md)
- [三轮工程与九轮专项审稿总览](docs/reviews/三轮项目迭代总览.md)
- [暂停恢复调研](docs/主流推理框架暂停恢复调研.md)
- [部署说明](docs/部署说明.md)
- [实验报告](docs/实验报告.md)
- [LongBench QA 开关 PLLM 对照实验](docs/LongBench-QA开关PLLM实验.md)
- [数据面实现与待验收说明](docs/数据面实现与待验收说明.md)
- [数据面三轮审稿与答辩](docs/reviews/dataplane-implementation/round-1-review.md)
- [当前实现三轮论文修订](docs/reviews/iteration-4/README.md)
- [演示视频脚本](docs/演示视频脚本.md)
- [黑客松十日谈](docs/十日谈.md)
