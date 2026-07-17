# Data Plane / Round 1 / Reviewer

**总体判断：5/10，Weak Reject。** 数据面源码覆盖了关键模块，但作者仍需证明“physical residency”不只是 Python cache metadata。

## 可信贡献

1. safetensors tensor slice 已具有 shard 与绝对 file offset，可做精确 `pread`。
2. package 包含 model fingerprint、layout 与 SHA-256，具备 fail-closed 基础。
3. actual Top-22 是 mapping 发布前的权威输入，没有引入近似专家。

## 主要问题

**DP.R1.1 Critical：evict 并不释放 allocation。** 仅删除 logical mapping 会保留 physical tensor。论文必须区分 slot replacement 与 capacity resize；只有重建更小的 parameter 第一维才能声称释放容量。

**DP.R1.2 Critical：checkpoint NVFP4 与 Marlin runtime layout 不同。** 如果 miss 直接把原始 safetensors bytes 写入 postprocessed tensor，结果必错。必须说明 transformed cache 如何生成、版本化和校验。

**DP.R1.3 Major：源码未运行。** 旧 39 tests 不能作为本轮实现证据。所有表格和前端必须保持 untested，直到实际 40 层 registration 完成。

## 可证伪条件

若不同 slot profile 的 `torch.cuda.memory_allocated`/UMA `MemAvailable` 不变化，或 miss 后任一 Marlin row 与 full-resident baseline 不一致，则 physical data-plane 主张失败。

## 建议结论

**Weak Reject**
