# Data Plane / Round 3 / Reviewer

**总体判断：6.5/10，Weak Accept for untested implementation artifact。** 数据布局和 lifecycle 较完整，剩余风险集中在 RDMA 安全/提交语义与 resize 原子性。

## 主要问题

**DP.R3.1 Critical：远端 PUT 必须确认 SSD commit。** RDMA completion 只证明 NIC 完成，不能证明远端应用已原子落盘。需要独立 commit ACK。

**DP.R3.2 Critical：RDMA store 不能无认证监听所有地址。** 任意网络客户端覆盖 `.pllmex` 会造成持久化供应链攻击。至少需要预共享认证、目录 traversal 防护与防火墙说明。

**DP.R3.3 Major：40 层 destructive resize 不是事务。** 若第 20 层失败，前 19 层已经改变。系统必须保持 quiesced 并明确重试/重启策略，不能自动 wake。

**DP.R3.4 Major：不运行测试意味着不能判断 C++ 是否可编译。** 最终答复必须直接说这一点，不使用“完成并验证”。

## 可证伪条件

断链、错误 rkey、remote disk-full 或 checksum corruption 任一情况下若 vLLM 仍执行 kernel，fail-closed 主张失败。

## 建议结论

**Weak Accept for implementation artifact; evaluation deferred**
