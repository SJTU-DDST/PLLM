# PLLM Blender 手动演示指南

所有命令都在 **XRDP 桌面里的 GNOME Terminal** 中执行。需要保持运行的命令，请放在不同的终端标签页中。

## 0. 概览：PLLM 有哪些前端模块

PLLM 有两个用户可见的前端模块：

| 前端模块 | 打开方式 | 用途 |
| --- | --- | --- |
| PySide6 桌面弹窗 | `bash scripts/run_desktop.sh --expanded` | 显示当前状态，提供释放、唤醒和策略按钮 |
| Vue.js Web 控制中心 | 浏览器访问 `http://127.0.0.1:17860` | 显示 GPU 指标、调度决策和历史事件 |

以下组件不是前端：

- GNOME 插件：没有窗口，只负责识别当前前台应用。
- Blender：用于制造前台 GPU 负载。
- vLLM：模型推理服务。
- PLLM daemon：连接前台检测、模型服务和两个前端。

完整启动顺序：

```text
GNOME 插件 -> 加载 vLLM -> PLLM daemon -> 两个前端
-> start_llm_load.sh 启动实际推理 -> Blender 渲染
```

## 1. 运行 GNOME 插件

执行：

```bash
gnome-extensions enable pllm-foreground@local
gnome-extensions info pllm-foreground@local
```

应看到：

```text
State: ENABLED
```

确认插件已经识别当前窗口：

```bash
gdbus call --session \
  --dest org.pllm.Foreground \
  --object-path /org/pllm/Foreground \
  --method org.pllm.Foreground.GetActive
```

命令应返回当前窗口的 PID、应用名称和标题。GNOME 插件本身不会弹出窗口。

## 2. 加载模型服务（此时还没有推理请求）

新建一个终端标签页，执行：

```bash
cd /home/cong/PLLM
conda activate pllm

export HIBERCACHE_DIR=/mnt/ssd-storage/cong/pllm-cache
export PLLM_EER_CACHE_DIR=/mnt/ssd-storage/cong/pllm-experts
export PLLM_EER_SLOTS_PER_LAYER=380
export PLLM_VLLM_ENABLE_SLEEP_MODE=1
export PLLM_VLLM_ENABLE_HIBERCACHE=1
export PLLM_VLLM_GPU_MEMORY_UTILIZATION=0.40
export PLLM_VLLM_MAX_NUM_SEQS=1

bash scripts/run_vllm_eer.sh \
  2>&1 | tee /tmp/pllm-vllm-eer380.log
```

不要添加 `--enable-return-routed-experts`，它与本次演示使用的 HiberCache 不兼容。

等待日志出现：

```text
Application startup complete
```

模型初始化约需两分钟。看到上述日志后保持该终端运行。

这一步只让模型常驻显存并监听 `8000` 端口，不会自动生成 token。实际推理由第 8 步的 `./start_llm_load.sh` 启动。

## 3. 运行 PLLM daemon

新建一个终端标签页，执行：

```bash
cd /home/cong/PLLM
conda activate pllm
bash scripts/run_daemon.sh
```

保持该终端运行。

检查 daemon：

```bash
curl -fsS http://127.0.0.1:17860/api/v1/status
```

能返回 JSON 即表示 daemon 已启动。

## 4. 运行桌面弹窗

确认第 3 步的 daemon 已经启动后，新建一个终端标签页，执行：

```bash
cd /home/cong/PLLM
conda activate pllm
bash scripts/run_desktop.sh --expanded
```

执行后，桌面右上角应出现标题为 `PLLM` 的弹窗，并直接显示 daemon、模型、GPU 和前台应用状态。保持该终端运行。

如果没有看到弹窗：

```bash
pgrep -af 'python.*-m pllm.desktop'
```

有输出时按 `Alt+Tab` 并选择 `PLLM`；无输出时查看启动弹窗的终端中是否有 Qt 报错。

## 5. 打开 Web 控制中心

打开 Firefox，在地址栏输入：

```text
http://127.0.0.1:17860
```

现在两个前端都已打开：桌面右上角的 PySide6 弹窗和 Firefox 中的 Vue.js 控制中心。

## 6. 设置演示策略

新建一个终端标签页，执行：

```bash
curl -fsS -X PUT http://127.0.0.1:17860/api/v1/policy \
  -H 'Content-Type: application/json' \
  -d '{"mode":"foreground_priority","creative_hold_seconds":0.5,"resume_idle_seconds":30}'
```

运行预检：

```bash
cd /home/cong/hackathon/blender_demo
./preflight.sh
```

最后应显示：

```text
Preflight passed.
```

## 7. 运行状态监控

在当前终端执行：

```bash
cd /home/cong/hackathon/blender_demo
./watch_pllm.sh
```

该终端会持续显示当前状态、前台应用、GPU 利用率、显存和调度原因。

## 8. 启动实际推理：运行 `./start_llm_load.sh`

新建一个终端标签页，执行：

```bash
cd /home/cong/hackathon/blender_demo
./start_llm_load.sh
```

这一步是 Blender 演示的必需步骤。等待终端连续输出模型生成内容后，再继续第 9 步打开 Blender。

只验证请求链路时，可以用同一脚本运行 16-token 短测试：

```bash
PLLM_DEMO_MAX_TOKENS=16 ./start_llm_load.sh
```

正式演示不设置该变量，默认持续生成 4096 tokens。

## 9. 运行 Blender

新建一个终端标签页，执行：

```bash
cd /home/cong/hackathon/blender_demo
./open_demo.sh demo
```

Blender 打开后：

1. 点击 Blender 窗口，让它成为前台应用。
2. 确认状态监控中的 `foreground` 显示 Blender。
3. 在 Blender 中按 `F12` 开始 GPU 渲染。
4. 同时观察桌面弹窗、Web 控制中心、状态监控和模型输出。

## 10. 观察释放与恢复

渲染开始后应看到：

1. PLLM 检测到 Blender 前台 GPU 负载。
2. 模型生成暂停。
3. PLLM 依次进入 `YIELDING`、`QUIESCING` 和 `HIBERNATED`。
4. 两个前端显示释放动作、原因和已释放资源。
5. Blender 获得更多 GPU 资源。

渲染完成或按 `Esc` 停止后应看到：

1. PLLM 等待系统稳定空闲。
2. PLLM 进入 `RESTORING`，随后返回 `ACTIVE`。
3. 模型恢复输出。

深度释放后的模型重载可能需要数分钟。恢复过程中不要重复点击唤醒。

## 11. 手动释放与恢复

跳过 Blender 自动触发，直接释放：

```bash
curl -fsS -X POST http://127.0.0.1:17860/api/v1/actions \
  -H 'Content-Type: application/json' \
  -d '{"action":"hibernate","level":2}'
```

手动恢复：

```bash
curl -fsS -X POST http://127.0.0.1:17860/api/v1/actions \
  -H 'Content-Type: application/json' \
  -d '{"action":"wake"}'
```

也可以使用桌面弹窗中的“立即释放”和“唤醒”按钮。

## 12. 停止演示

依次执行：

1. 在 Blender 中按 `Esc`，然后关闭 Blender。
2. 在模型生成终端按 `Ctrl+C`。
3. 在桌面弹窗终端按 `Ctrl+C`。
4. 在 PLLM daemon 终端按 `Ctrl+C`。
5. 在 vLLM 模型终端按 `Ctrl+C`。
