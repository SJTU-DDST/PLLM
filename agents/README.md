# PLLM Peer-Review Agents

这里包含两个职责隔离的研究智能体：

- `reviewer.system.md`：从新颖性、正确性、证据和 DGX Spark 适配性审查论文。
- `author.system.md`：逐条接受、反驳、删除或降级主张，并输出完整修订稿。

编排器默认运行 4 轮，每轮保存 review、rebuttal 和 manuscript snapshot：

```bash
conda activate pllm
export PLLM_REVIEW_BASE_URL=http://127.0.0.1:8000
export PLLM_REVIEW_MODEL=/path/to/model
python scripts/run_peer_review.py \
  --manuscript paper/HiberFlow-ACM四页稿.md \
  --context docs/research/近一年相关工作矩阵.md \
  --rounds 4
```

默认输出到 `results/peer-review/`，不会修改原论文。审阅产物后可显式传入 `--apply-final`。本地 GPU 忙碌时不要启动模型；仓库中的 `docs/reviews/四轮审稿迭代记录.md` 是本次由 Codex 按相同角色约束完成的静态审稿记录。
