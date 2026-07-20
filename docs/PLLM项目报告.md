# PLLM HiberFlow 项目报告

## 1. 项目概述

PLLM HiberFlow 是面向 DGX Spark 与 NVIDIA Linux 桌面 AI 工作站的前台感知推理运行时。它解决的核心问题是：当大模型推理服务长期驻留 GPU，而用户临时启动 Blender、游戏、视频编码或其他前台任务时，系统如何在不中止未知进程、不改变模型路由正确性的前提下，及时让出算力、显存、统一内存容量和 I/O 带宽，并在前台压力结束后恢复推理服务。

传统云端调度通常假设推理任务拥有稳定的设备配额；桌面工作站的优先级恰好相反，交互用户必须拥有最终资源优先权。PLLM 因此把“前台资源包络”作为一等输入，并把暂停、弹性驻留、状态保存和恢复统一到一个控制面中。

第一版只控制明确接入的 vLLM 服务。训练进程、未知 CUDA 进程，以及未公开 Sleep API 的外部服务不在控制范围内。

## 2. 使用场景

典型演示流程如下：

1. NVIDIA Nemotron 通过 vLLM 在后台提供 OpenAI 兼容推理。
2. PLLM 控制中心展示模型、GPU、内存、功耗和专家驻留状态。
3. 用户切换到 Blender 或启动前台渲染。
4. Foreground-QoS Agent 识别前台应用与资源压力，先尝试微暂停或安全的弹性收缩；若资源包络仍不可满足，则进入深度休眠。
5. 前台压力消退后，PLLM 按本地 NVMe、host backup 或远端 warm source 的可用性选择恢复路径。
6. 请求继续通过 PLLM 代理接入，用户无需手动停止和重启模型服务。

同一机制也适用于游戏、视频编解码、交互式 CUDA 应用、内存压力和低电量场景。

## 3. 设计目标与边界

### 3.1 设计目标

- 前台任务能够在短时间内获得明确的资源让渡。
- 后台推理连接、请求记录和恢复过程由统一控制面管理。
- 每次决策都能解释其输入、代价和选中的动作。
- MoE 专家预测只影响预取与驱逐，不改变模型实际 Top-k 路由。
- 本地 NVMe、host memory 与 RDMA 远端缓存共享同一对象和 manifest 语义。
- 对 DGX Spark 的 UMA、功耗和 RDMA 能力做硬件感知适配。

### 3.2 非目标

- 不抢占或终止任意 CUDA 进程。
- 不把训练作业纳入自动休眠控制。
- 不在默认配置中向局域网公开控制 API。
- 不把 host-staged RDMA 声称为 GPUDirect RDMA。
- 不用预测专家替换真实 router 输出，也不通过动态 Top-k 牺牲质量。
- 在模型内部状态序列化未接通前，不声称所有混合模型状态已经实现逐 token 精确恢复。

## 4. 总体架构

```text
┌────────────────────── 感知层 ──────────────────────┐
│ GNOME focus │ NVML │ NVENC/NVDEC │ PSI │ 内存 │ 电源 │
└────────────────────────┬───────────────────────────┘
                         v
              Foreground-QoS Agent
                         │ workload + resource envelope
                         v
             Policy Engine + Cost Model
                         │
          ┌──────────────┼──────────────────┐
          v              v                  v
     vLLM Sleep     Expert Residency   Storage Planner
     Level 0/1/2      + Data Plane     SSD / host / RDMA
          └──────────────┼──────────────────┘
                         v
                  PLLM Controller
                REST / SSE / OpenAI Proxy
                   │               │
                   v               v
             Vue Dashboard    PySide6 Overlay
```

核心代码按职责拆分：

- `monitor.py`、`foreground.py`：系统、GPU 与桌面输入。
- `policy.py`、`cost_model.py`：资源分类、成本比较和动作建议。
- `controller.py`：状态机、动作互斥、恢复编排和安全边界。
- `vllm.py`、`pause_resume.py`：vLLM 服务发现与 Sleep API 控制。
- `hibercache.py`、`hiberstate.py`：活跃状态的分层保存与事务提交。
- `expert_residency.py`、`decode_residency.py`：专家驻留计划。
- `expert_dataplane.py`、`vllm_eer_runtime.py`：真实物理槽位和模型内运行时。
- `expert_store.py`、`rdma_bridge/`：本地及远端对象存储。
- `api.py`：控制 API、SSE、请求代理与 replay。

## 5. Foreground-QoS Agent

Agent 以 250 ms 为默认周期采样：

- GNOME 活跃窗口、PID、应用 ID 与窗口类；
- NVML 进程级 GPU 使用、显存、功耗与温度；
- NVENC/NVDEC 活动；
- `/proc/pressure/memory` 与 `MemAvailable`；
- 电池容量、是否接入电源和系统电源策略；
- vLLM 当前服务状态与推理阶段。

应用模式把 Blender、DaVinci Resolve、OBS、ffmpeg、Steam/Proton 等识别为 creative 或 game workload。训练、DeepSpeed、torchrun 等默认位于排除列表，避免 PLLM 越权控制。

策略不是简单阈值触发。控制器为保持推理、微暂停、弹性驻留和深度休眠计算代价，结合前台持续时间、恢复时间、内存下限、显存缺口、I/O 预算和功耗压力选择动作。Web API 同时公开传感器快照和决策原因，便于演示与诊断。

## 6. 分级状态机

```text
ACTIVE <-> ELASTIC_RESIDENT -> YIELDING -> QUIESCING
                                      -> HIBERNATED -> RESTORING -> ACTIVE
```

- **ACTIVE**：完整模型可服务。
- **ELASTIC_RESIDENT**：decode 阶段维护受约束的专家工作集。
- **YIELDING**：vLLM Level 0 暂停 scheduler；连接保持，GPU cache 不释放。
- **QUIESCING**：阻止新工作并等待连接器进入可提交边界。
- **HIBERNATED**：Level 1 保留 host 权重备份，或 Level 2 丢弃权重并保留恢复信息。
- **RESTORING**：恢复权重、连接器与调度器，然后开放新请求。

所有手动和自动动作共用同一控制器，避免前端、悬浮窗和后台 Agent 同时发起冲突迁移。`abort` 仅用于错误恢复，不是日常抢占策略。

## 7. Phase-Constrained Elastic Expert Residency

Nemotron 的 routed experts 占据主要权重空间，因此 PLLM 把模型拆成必须常驻的稠密部分和可管理的专家对象。运行时导出已经经过 ModelOpt NVFP4/Marlin 转换的对象，并为每个对象保存 layer、expert、shape、dtype、layout、offset、size 和 checksum。

### 7.1 正确性约束

PLLM 始终执行 router 选出的真实 Top-k 专家。预测集合只决定提前搬运和保留的对象；预测 miss 会触发正确专家加载，而不会改写路由、降低 Top-k 或使用近似专家。

### 7.2 阶段约束

- prefill 强制完整驻留，避免大量并行 token 造成换页风暴。
- decode 使用 request-local 历史窗口形成下一窗口候选集合。
- 规划器逐层检查容量、held-out miss、剩余生成 horizon、转换成本和 TPOT 上限。
- 证据不足或预算不可行时维持完整驻留；持续 miss debt 越界时退出 paging，转入 yield 或 hibernate。

### 7.3 数据面

物理数据面在 vLLM 进程内维护 expert slot、映射表和缓存对象。控制面通过本地 Unix socket 查询状态并发送 `resize`、`prefetch`、`evict` 等命令。自动物理 resize 默认关闭，避免未经目标硬件验收的重建动作进入常规运行。

## 8. HiberCache 与恢复

HiberCache 基于 vLLM `OffloadingConnector + TieringOffloadingSpec`，使用可配置 host staging 和本地文件层保存活跃请求拥有的 KV block。PLLM 的版本守卫补丁为深度 `mode=keep` 提供连接器排空和状态重置，并在不满足补丁条件时回退到 token 重算。

恢复数据分为三类：

1. **不可变权重**：不重复写出 checkpoint；从共享模型目录、expert cache 或远端 warm source 恢复。
2. **可重算缓存**：已有 KV block 直接复用，缺失部分由 token 重算。
3. **不可随意重算的 live state**：事务式 carrier 按固定大小分块，manifest 最后提交，避免半完成状态被当成有效快照。

目前 live-state carrier 已具备 SSD/RDMA 事务语义，但模型内部 Mamba、KV block table、sampler 和 RNG serializer 仍需逐项接入。因此系统会明确显示能力边界，不把“carrier 可用”展示成“exact resume 已完成”。

## 9. SparkLoad 与多级存储

SparkLoad 避免为深度休眠再次写出大型不可变权重：

- 本地恢复直接复用只读 safetensors 模型目录。
- fastsafetensors 负责加载路径。
- 独显环境可选择 host backup 或 GDS 能力。
- DGX Spark/GB10 选择统一内存兼容路径，不把 CPU offload 视为额外容量。
- 专家对象可使用连续 pack 减少大量小文件的元数据与随机 I/O。

RDMA bridge 提供带 token 与路径守卫的对象 store，以及持久 RC QP、预注册共享 host MR 的 volatile pool。Python 侧通过 `memoryview` 消费共享区域，避免 payload pipe；GB10 上终点仍是 host memory，后续 slot refill 需要单独的数据搬运。

## 10. DGX Spark 平台适配

PLLM 对 DGX Spark 的优势利用集中在四个方面：

- **统一内存感知**：同时约束系统可用内存、模型驻留和前台显存需求，避免把同一物理内存重复计算。
- **全栈 NVIDIA 软件**：使用 NVML、vLLM、ModelOpt NVFP4、Marlin、CUDA 能力探测和 NVIDIA Nemotron 开源模型。
- **ConnectX 数据路径**：在能力允许时使用 RDMA 远端 warm source；在 GB10 上明确采用 host-staged fallback。
- **桌面交互闭环**：将 GNOME 前台事件与 GPU、编码器和内存信号组合，使 DGX Spark 不只是推理服务器，也能作为共享的个人 AI 工作站。

能力探测结果通过 `/api/v1/capabilities` 暴露。平台不具备的 GDS、GDR 或 loader 能力不会仅凭配置被标记为可用。

## 11. 多智能体与模型融合

项目包含 Reviewer Agent 与 Author Agent：

- Reviewer 负责新颖性、事实、证据边界、平台适配和可证伪性检查。
- Author 对每条意见选择接受、部分接受、证据反驳、降级主张或删除，并输出完整修订稿。

两者使用隔离的 system prompt，编排器至少运行三轮，逐轮保存 review、rebuttal 和 manuscript snapshot，默认不覆盖原文。接口兼容 OpenAI Chat Completions，因此可以连接本地 NVIDIA Nemotron/vLLM，也可以连接阶跃星辰等兼容模型服务。API key 只从环境变量读取，不写入配置或仓库。

该智能体流程用于文档与方案质量控制，不进入实时资源决策关键路径；前台资源动作仍由可审计的本地确定性 guard 执行。

## 12. 前后端与接口

### 12.1 Web 控制中心

前端使用 Vue 3 静态构建文件，由 Flask 直接托管，不依赖 Node.js 构建链。页面包含总览、状态机、资源传感器、策略编辑、专家驻留、数据面能力、事件和请求 replay。

### 12.2 桌面悬浮窗

PySide6 悬浮窗提供当前状态、资源压力、自动/手动模式和常用动作；GNOME Shell 扩展把活跃窗口元数据写入控制面可读取的位置。

### 12.3 OpenAI 兼容代理

客户端统一连接 PLLM `/v1/chat/completions`。控制器在请求进入 prefill 前检查服务和专家驻留条件；暂停时创建 replay 记录并返回可追踪 ID；恢复后可通过 replay API 重新提交。

### 12.4 管理 API

REST API 提供状态、能力、服务发现、策略更新、手动动作、专家计划和数据面操作；SSE 端点提供实时遥测。默认仅监听回环地址。

## 13. 安全设计

- API 与 vLLM 默认绑定 `127.0.0.1`。
- 不提供终止未知进程的权限。
- 训练相关进程模式默认排除。
- RDMA 服务使用 token、root directory 和对象路径校验。
- 模型目录只读复用，缓存写入独立目录。
- API key 和 RDMA token 不进入版本控制。
- manifest 使用最后提交语义；未完成快照不会被当成有效恢复点。
- 自动专家物理重建默认关闭。

## 14. 项目完整性

仓库提供以下可独立演示和部署的组成部分：

- 使用 mock vLLM 的无 GPU 控制面演示；
- 真实 NVIDIA Nemotron/vLLM 启动脚本；
- Web 控制中心、桌面悬浮窗和 GNOME 扩展；
- 用户级 systemd 服务；
- 本地 NVMe 与可选 RDMA 数据路径；
- OpenAI 兼容推理代理；
- 双智能体文档审阅工具；
- 单元测试、集成配置、部署指南、演示脚本和黑客松开发记录。

快速启动、完整配置、端口和常见故障处理见仓库根目录 `README.md`。

## 15. 演示设计

推荐演示遵循一条清晰主线：

1. 在控制中心确认后台推理正在运行。
2. 发起一个流式请求，展示 OpenAI 兼容接入。
3. 切换到 Blender 并开始渲染，展示前台识别和资源压力变化。
4. 展示状态机从 active 进入 yielding 或 hibernated，以及显存/内存包络的同步变化。
5. 停止前台压力，展示 restoring 与推理服务恢复。
6. 打开能力页面，说明 DGX Spark UMA、NVIDIA 模型/SDK 和 RDMA fallback 的适配逻辑。
7. 简要展示双智能体如何用本地 Nemotron 或阶跃星辰兼容接口审阅项目文档。

详细操作见 `docs/Blender手动演示操作指南.md` 与 `docs/演示视频脚本.md`。

## 16. 后续工作

- 接入 NemotronH Mamba/KV/RNG 的模型级 serializer 与 restore。
- 完成 RDMA shared host-MR 到 GPU/UMA expert slot 的流水化 refill。
- 在 DGX Spark 上校准统一内存、带宽、功耗和前台 QoS 包络。
- 扩展 GNOME 之外的桌面环境输入。
- 为跨连接 token exactly-once 增加客户端 ACK 协议。
- 在更多 MoE 模型和前台应用上验证通用性。

## 参考资料

- [NVIDIA DGX Spark Hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark CUDA Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
- [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- [vLLM KV Offloading Usage Guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- [MoE-Infinity](https://arxiv.org/abs/2401.14361)
- [ProMoE](https://arxiv.org/abs/2410.22134)
- [ExpertFlow](https://arxiv.org/abs/2410.17954)
