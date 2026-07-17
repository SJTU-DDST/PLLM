# Iteration 3 / Round 1 / Author

## 逐条回应

**I3.R1.1 接受并修改。** `ExpertResidencyControlPlane.recommend` 现在对独显使用 `gpu_memory_total_gb`，对 DGX Spark/UMA 使用 system `memory_total_gb`；每个 plan 新增 `capacity_scope=discrete_gpu_vram|coherent_uma|manual_envelope`。新增两项单元测试防止回归。修复后，当前 99% 非 vLLM GPU 压力不再得到 full-resident，而是 128 slots/layer 的非执行建议。

**I3.R1.2 接受并修改。** Vue 面板固定显示 `CONTROL PLANE ONLY` 和 `NOT EXECUTABLE`；论文和项目报告将 40 层图明确称为 slot projection，而非 cache heatmap。`data_plane_ready=false` 同时出现在 capability probe 与 plan API。

**I3.R1.3 接受并澄清。** monitor loop只刷新 `RuntimeStatus.expert_residency`。当前 control plane 没有 slot backend 引用，所有 plan 都写入 `executable=false`，也不会调用 `VLLMClient.sleep/reload_weights`。API guardrail 测试验证这些字段；物理权重修改仍列为未实现。

移动端截图发现 `NOT EXECUTABLE` 被压窄后逐字换行，已将 evidence 状态移到独立行。实验报告新增 API 证据样例和截图路径。

## 本轮修改结论

保留“实时控制面接入”工程主张；不把任何投影值解释为实际 expert residency、释放量或吞吐。
