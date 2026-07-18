# Iteration 4：基于当前实现的三轮论文修订

> 证据快照：PLLM commit `24a7dc6`。本迭代不生成新实验结果，只使论文主张与已有代码、JSON 和硬件实测一致。

## 迭代方法

每一轮固定为“修改论文→独立 Reviewer 严格评估”。各轮 paper snapshot 保留当时文本，不随终稿回填。

| 轮次 | 论文修订 | Reviewer 结论 |
| --- | --- | --- |
| Round 1 | 合入真实 120B EER、Level 1/2、60GiB admission、75→71 remote pool 和 Blender artifact；区分 v1 跨机与 v2 本机 smoke | 5.5/10 Weak Reject；主张过强，RDMA 口径和事务语义不清 |
| Round 2 | 题目降级为 measurement-grounded prototype；增加 validated/artifact/hypothesis claim ladder；统一 wall-time 口径 | 6.5/10 Borderline；诚信问题解决，但可行域、前台因果与真实 route 缺失 |
| Round 3 | 形式化 `F_E/F_H` frontier；增加 H1--H5 状态、关键路径对齐、Blender B0/B1/B2 协议和 artifact mapping | 6/10 Weak Reject（顶会）；8.5/10 Strong Accept（Hackathon） |

## 文件

- `round-1-paper.md` / `round-1-review.md`
- `round-2-paper.md` / `round-2-review.md`
- `round-3-paper.md` / `round-3-review.md`
- 最新稿：`paper/HiberFlow-ACM四页稿.md`

## 最终证据边界

**已闭环：** 128-slot route-preserving functional execution、Level 2 容量 admission、15.859GB 跨机 volatile profile transport、GNOME/Blender 机制 artifact。

**未闭环：** 真实 route 下的 elastic Pareto 区间、Blender QoS 因果对照、RDMA-to-slot resume、NemotronH exact state resume 和 DGX Spark UMA 实验。

因此下一步不是继续修饰论文，而是按 Round 3 审稿意见运行五类决定性实验。
