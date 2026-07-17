# Iteration 3 / Round 3 / Author

## 逐条回应

**I3.R3.1 接受。** 论文首页已经声明“研究设计与系统原型稿”，摘要不报告 Nemotron 真实性能；实现表逐组件区分已实现 control plane 与未实现 data plane。结论保留“潜在创新”措辞，不将标题当作完成声明。

**I3.R3.2 接受。** 实时 API 的 evidence 由后端生成，UI 只映射显示，不能由用户把 `data_plane_ready=false` 改为 live。评委模式使用独立 `replayStatus`，保留 historical simulation 字段。演示脚本要求当前投影值口头标明为规划值。

**I3.R3.3 接受并调整优先级。** 下一工程里程碑固定为最小 slot data plane，不继续扩展框架适配或复杂策略：先对可控 fixture/单层 kernel 验证 object loader、logical remap、generation/checksum 和 exact miss stall，再决定是否扩展到 40 层 Nemotron。若 backend 不能保持 exact Top-22，自动回退 full/hibernate。

README 已链接论文、实验报告和审稿目录；39 项无 GPU 测试与桌面/移动截图只列入实现证据。

## 最终立场

第三轮接受 Weak Accept for hackathon research prototype，不争辩为完成的顶会系统论文。真正的论文成立条件仍是 physical slot backend、真实 route traces、DGX Spark 前台 QoS/能耗和事务连续性实验。
