# PLLM 文档审阅智能体

该目录提供两个职责隔离的智能体角色：

- `reviewer.system.md`：从新颖性、正确性、证据边界和 DGX Spark 适配性审查文档。
- `author.system.md`：逐条接受、反驳、删除或降级主张，并输出完整修订稿。

编排器使用 OpenAI 兼容的 Chat Completions API，可连接本地 NVIDIA Nemotron/vLLM，也可连接阶跃星辰等兼容服务：

```bash
# base URL 是 /v1 之前的部分；密钥只通过环境变量传入
export PLLM_REVIEW_BASE_URL='<openai-compatible-base-url>'
export PLLM_REVIEW_MODEL='<model-id>'
export PLLM_REVIEW_API_KEY='<api-key>'

python scripts/run_peer_review.py --rounds 4
```

默认输入为 `docs/PLLM项目报告.md` 和 `docs/主流推理框架暂停恢复调研.md`，输出到 `results/peer-review/`。每轮都会保存 review、rebuttal 和 manuscript snapshot；默认不修改原文，只有显式传入 `--apply-final` 才会覆盖 `--manuscript` 指向的文件。

自定义输入示例：

```bash
python scripts/run_peer_review.py \
  --manuscript /path/to/manuscript.md \
  --context /path/to/architecture.md \
  --context /path/to/evidence.md \
  --output-dir results/custom-review \
  --rounds 4
```

不要把 API 密钥、私有模型地址或未脱敏材料写入仓库和审阅产物。
