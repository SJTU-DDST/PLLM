# Iteration 1 / Round 1 / Reviewer

**总体判断：5/10。** Header-only catalog 是必要的证据改进，但它推翻了论文原先的容量分类，说明旧稿的 routed/skeleton 定义不够严格。

## 主要问题

`I1.R1.1 Critical`：旧稿写 `64.313GiB routed`，新 parser 只找到 `59.063GiB experts.<id>`。必须以实际可换出的 physical expert object 为准，不能把 latent projection、gate 或其他 MoE 周边权重算成可独立换页 expert。

`I1.R1.2 Major`：需要证明每层 512 个 expert objects 完整存在，并检查 weight 与 scales 都进入同一个 object。

`I1.R1.3 Major`：合成 trace 不能用于任何 prediction accuracy、cache hit 或路由局部性结论。

**建议：Weak Reject，修改静态事实后重审。**
