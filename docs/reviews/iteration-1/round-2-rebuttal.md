# Iteration 1 / Round 2 / Author

- `I1.R2.1`：接受。最新论文明确区分 checkpoint object catalog 与 vLLM slot data plane，后者仍为未实现。
- `I1.R2.2`：接受。`expert_catalog.json` 同时输出 min/max/average；当前模型三者相等，其他模型不作等长假设。
- `I1.R2.3`：接受。Catalog 构建时验证每个发现的 MoE 层都恰有 `n_routed_experts` 个对象，缺失即失败。

反驳范围：catalog 已足以给出当前 checkpoint 的静态容量上界，但不用于运行时 reclaim 声明。
