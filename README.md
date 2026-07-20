# gaiban-lite

DeepSeek-V4-Flash（284B/13B）在 2×8×RTX 4090 上的推理系统。官方推理参考实现在
[`reference/`](reference/)；可行性评估、实施方案与性能预估见
[`docs/feasibility-v4-flash-2x8x4090.md`](docs/feasibility-v4-flash-2x8x4090.md)。
实验方法与 kernel 资产来源：`../gaiban`（DeepSeek-V4-Pro on 64×4090）。
长期目标与无人值守约束见 [`CLAUDE_GOAL.md`](CLAUDE_GOAL.md)。

## 当前状态（2026-07-20）

- Phase 0 环境与权重：**全部完成**——双机部署与权重校验（附录 B）、B1/B2 标定复跑
  （[`experiments/B1-allreduce-recal`](experiments/B1-allreduce-recal/README.md)、
  [`experiments/B2-ib-recal`](experiments/B2-ib-recal/README.md)，锚点全部复现）、
  Flash checkpoint 契约核实
  （[`experiments/A0-flash-checkpoint-contract`](experiments/A0-flash-checkpoint-contract/README.md)，
  确认免离线 repack、加载侧直读）、reference golden-token oracle
  （[`experiments/D0-reference-oracle`](experiments/D0-reference-oracle/README.md)，
  MP=8 单机跑通，8 条 prompt golden tokens 冻结）。
- Phase 1 kernel regear：**进行中**——
  [`A3F`](experiments/A3F-marlin-moe-flash/README.md) Marlin MoE 完成：数值 gate 通过，
  整 expert 口径峰值 899 GB/s（Pro 锚点 97%），TP4-local inter=512 口径 687–760 GB/s
  （B=512 时 1.24 ms/层 → 11 层 stage MoE ≈ 13.7 ms，高于 §5.2 的 ~10.8 ms 假设）；
  [`A4F`](experiments/A4F-attention-flash/README.md) attention/HC 计时完成：TP4 h=16
  原生可跑（无需 topk-block 修复），h=64 单 launch 撞 sm89 smem 墙 → DP-attention 须
  head-loop（4×h16）；DP 口径折算 11 层 stage 非 MoE 部分 ~13 ms 量级，高于 §5.2 的
  6–7 ms 假设，decode 预估收敛至 **~15–17k tok/s**（区间下沿），待 C1F 实测修订。
  下一步：**C1F 集成 block bench**（移植 gaiban C1' `--attn-mode dp`，Marlin MoE +
  head-loop attention + HC + NCCL 整层 decode 定数）；随后 fused indexer 接入。
