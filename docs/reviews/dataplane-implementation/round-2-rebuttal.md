# Data Plane / Round 2 / Author

**DP.R2.1 接受并修改。** Elastic script 强制 lazy `safetensors`。Runtime 复用 vLLM pre-read expert filter，并扩展为跳过全部 numeric routed expert weights/scales；checkpoint 启动只读取 non-routed 权重，routed rows 来自 runtime SSD/RDMA objects。

**DP.R2.2 接受。** Patched `create_weights` 在过滤前将 physical expert parameters 全部清零。Marlin 先对确定性零 rows 建立同形状 kernel，`process_weights_after_loading` 返回后立即从 checksummed runtime cache 覆盖全部初始 slots，服务尚未开放请求。

**DP.R2.3 接受。** Controller 在 Level 1/2 前发送 fail-closed `suspend`，撤销所有 maps；wake 后 `resume`。若 reload 再次执行 Marlin postprocess，runtime 先 unregister 旧 layer，再建立新 sink、mapping 和 generation。

**结论：** 冷启动 I/O 降低与 Level 2 连续性仍需真实 `iostat` 和 fault injection，不写成结果。
