# Iteration 1 / Round 1 / Author

- `I1.R1.1`：接受并修改。论文所有容量表改用严格 `experts.<id>.*` 的 59.063GiB；non-routed 改为 15.720GiB。
- `I1.R1.2`：接受并提供证据。Parser 验证 40 层、每层 512 objects，共 20,480 个；每个 object 汇总其 weight、input scale 和 weight scales，当前 checkpoint 中均为 3,096,592 bytes。
- `I1.R1.3`：接受。所有合成记录带 `source=synthetic_no_gpu`，结果文件带 `real_route_evidence=false`。论文只把它作为 schema/回放测试。

修订：容量 projection、all-miss bytes、实现状态和实验报告全部重算。
