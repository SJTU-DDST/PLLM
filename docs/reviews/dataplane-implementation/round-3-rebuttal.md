# Data Plane / Round 3 / Author

**DP.R3.1 接受并修改。** PUT 现在采用两阶段完成：client RDMA write completion 后发送 transfer-done；server 对 temporary file 执行 `fsync`、atomic rename、parent-directory `fsync` 后返回 commit ACK，client 收到 ACK 才成功返回。

**DP.R3.2 接受并修改。** Server/client 默认必须提供 `--token-file`，使用常量时间比较；只有显式 `--insecure-no-auth` 才可关闭。Object key 拒绝 absolute path 与 `..`。文档声明 token 控制面未加密，仅限受信任 RDMA fabric，并要求防火墙限制 17900/TCP。Package 在进入 slot 前仍需 SHA-256 与 model fingerprint。

**DP.R3.3 接受并收缩。** Resize 不是跨层原子。任一异常使 runtime 设置 `faulted=true`、对外撤销 `data_plane_ready`，controller 进入 `ERROR` 并保持 Level-0 quiesced，不调用 wake。所有外部 prefetch/evict/resize 也必须先通过 token-boundary `quiesced=true` guard。运维只能重试同一 profile 或重启 vLLM。

**DP.R3.4 接受。** 本轮没有执行 Python、CMake、CUDA 或 RDMA tests。最终材料只称 `implemented / untested`，并列出后续命令，不报告编译成功、显存释放或网络吞吐。

**后续静态审计补充。** live-state carrier 已增加 chunk/component 双层 checksum、manifest-last、本地 transaction-directory rename 和远端 manifest-last replication；但 `serializer_attached=false`，不能替代未实现的 NemotronH Mamba/KV/RNG restore hook。

远端 expert warm store 同样改为 objects-first、manifest-last；elastic runtime 必须先验证远端 manifest 的 format、fingerprint 和 object count，否则 fail closed。

**最终立场：** 数据面代码路径完成，系统论文的 correctness/performance claim 仍 pending GPU and two-host evaluation。
