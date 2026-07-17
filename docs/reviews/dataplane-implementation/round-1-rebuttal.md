# Data Plane / Round 1 / Author

**DP.R1.1 接受并修改。** 文档现在明确：LRU eviction 只撤销映射并复用固定 slots，不释放容量；容量释放只来自 Level-0 quiesced destructive resize。Resize 删除旧 parameters、清 allocator、创建新的第一维并重建 Marlin kernel。

**DP.R1.2 接受。** Runtime miss 不读取 raw checkpoint package。完整模型首次后处理后逐 expert 导出 `vllm_runtime_nvfp4_marlin_v1`；elastic runtime 只接受同模型 fingerprint、同 tensor shape/dtype 和 SHA-256 的 runtime package。Raw `pread` source 仅作为规范化 checkpoint 数据对象，不直接写 Marlin rows。

**DP.R1.3 接受。** 论文、报告和 README 全部改为 `IMPLEMENTED / UNTESTED`。新增 tests 没有运行，旧 39 tests 明确为修改前结果。Frontend 只有 runtime socket 返回 40 层完整 registration 才显示 live/executable。

**结论：** 保留实现主张，删除任何释放性能或正确性结果主张。
