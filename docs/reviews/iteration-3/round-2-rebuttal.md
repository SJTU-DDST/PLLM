# Iteration 3 / Round 2 / Author

## 逐条回应

**I3.R2.1 接受并修改。** 自动 recommendation 固定携带 `hypothetical_control_input_not_model_measurement`，并在 `assumptions` 中逐项公开 byte hit 和 false-prefetch ratio。手工端点使用 `manual_control_input_not_model_measurement`。论文没有把它们称为在线 calibration。

**I3.R2.2 接受并澄清。** `RuntimeStatus.state` 不会因 expert planner 变化；`elastic_resident` 只出现在 `expert_residency.plan.action`。界面把其置于独立的规划区。评委回放可以展示未来状态机，但 evidence 固定为 historical simulation，不能与实时数据混合。

**I3.R2.3 接受。** 论文结论继续使用“潜在创新”和“必须由真实数据面验证”。API/UI 仅列为实现完整度，不作为 H1-H5 的支持证据。

全量无 GPU 测试为 39 项。实验报告分别记录 pytest、QtWebEngine 离屏截图和 API dry-run；没有把页面可渲染等同于系统性能。

## 本轮修改结论

拒绝扩大主张。第三轮交付的是受证据约束的 planning/control surface，不是 dynamic expert residency 完成版。
