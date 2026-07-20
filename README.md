# gaiban-lite

DeepSeek-V4-Flash（284B/13B）在 2×8×RTX 4090 上的推理系统。官方推理参考实现在
[`reference/`](reference/)；可行性评估、实施方案与性能预估见
[`docs/feasibility-v4-flash-2x8x4090.md`](docs/feasibility-v4-flash-2x8x4090.md)。
实验方法与 kernel 资产来源：`../gaiban`（DeepSeek-V4-Pro on 64×4090）。
长期目标与无人值守约束见 [`CLAUDE_GOAL.md`](CLAUDE_GOAL.md)。

## 当前状态（2026-07-20）

- Phase 0 环境与权重：**已完成**（双机硬件/网络/软件栈核实与部署、权重三处逐分片校验
  通过，详见可行性文档附录 B）。B1/RDMA 标定复跑已完成（2026-07-20，见
  [`experiments/B1-allreduce-recal`](experiments/B1-allreduce-recal/README.md)、
  [`experiments/B2-ib-recal`](experiments/B2-ib-recal/README.md)，锚点全部复现，
  Flash 口径 TP4 [512,4096] allreduce ~300–342 µs、PP GDR 1.84 ms/16.8MB 跳）。
  剩余前置：Flash 版 convert（mp8 转换进行中）+ reference golden-token oracle。
- 下一阶段：**Phase 1 kernel regear**——Marlin MoE（256 experts, K=4096, inter 2048）/
  shared-expert FP8 / sparse_attn(h=16) / fused indexer 四件套换 Flash 几何并重跑
  A3/A4/C1 级 bench，确立 Flash 单层数字。
- 尚无实验目录；实验从 `experiments/` 起建，编号惯例沿用 gaiban。
