# Iteration 1 / Round 2 / Reviewer

**总体判断：6/10。** 数字已纠正，但 catalog 仍只证明 checkpoint layout，不证明 vLLM 可以按这些对象装载。

## 主要问题

`I1.R2.1 Critical`：Safetensors tensor 的逻辑边界不等于 vLLM/ModelOpt packed destination 的 slot 边界。论文必须继续把 physical slot loader 标为未实现。

`I1.R2.2 Major`：Projection 使用平均比例，仅当 experts 等长时精确。应记录 min/max object bytes，避免对其他 MoE 泛化。

`I1.R2.3 Major`：Header parser 必须拒绝每层缺失 expert 的 checkpoint，防止静默低估容量。

**建议：Borderline。**
