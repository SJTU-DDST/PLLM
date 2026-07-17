# Iteration 3 / Round 3 / Reviewer

**总体判断：7/10，Weak Accept for hackathon research prototype。** 证据边界现在清楚，剩余风险集中在论文最核心的数据面尚未实现。

## 可信贡献

1. 三轮实现形成 catalog/trace、prediction/planner、controller/API/UI 的递进链路。
2. exact-route invariant、容量域和 evidence taxonomy 已被测试固定。
3. 项目保留失败的 synthetic domain-shift 结果，并定义了明确退出 elastic 的条件。

## 主要问题

**I3.R3.1 Major：论文标题仍比证据强。** “Elastic Expert Residency” 容易让读者认为物理 slot backend 已完成。摘要、实现表和结论必须在最早位置声明 prototype/design status。

**I3.R3.2 Major：最终 Demo 需要双通道。** GPU 空闲前只能演示 LIVE sensors + MOCK planner；GPU 空闲后才能显示实际 resident bytes。前端必须禁止把 historical replay 切换成实时标签。

**I3.R3.3 Major：下一轮不应继续堆控制面功能。** 最有价值的后续工作是最小 physical data-plane proof：单层或小模型的固定 slot tensor、logical remap、exact checksum、miss stall，再扩展到 Nemotron。

## 次要问题

- 论文应给出当前 39 tests 和截图作为工程证据，但不要占用主要结果位置。
- 三轮审稿记录应能从 README 访问。

## 可证伪条件

若固定-layout NVFP4 kernel 无法支持 logical-to-physical remap，或 remap/loader overhead 消除 elastic 区间，则论文必须退回二元 hibernation 系统。

## 建议结论

**Weak Accept for hackathon prototype; system-paper verdict deferred**
