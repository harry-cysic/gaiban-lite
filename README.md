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
  [`C1F`](experiments/C1F-integrated-block/README.md) 集成整层 bench 完成并给出两组
  causal A/B：**DP-attention 胜 head-shard（B=512 −23%）、intermediate-TP 胜 EP**，
  既定形态全部通过检验；B=512/8K 整层加权 3712 µs → **reference-op 基线组件
  roofline ≈ 12.5k output tok/s**（低于预估带下沿 ~20%，归因：HC fp32 sinkhorn
  7.8 ms/stage、allgather 6.0 ms/stage 未入模型、attention 未用 W8A16）。
- 资产接入 A/B 已完成（见 C1F README 追加节）：**三项 gaiban 资产在 decode 侧
  全部收益甚微或改判**——eager fused MHC −3%、W8A16 投影中性、fused indexer 属
  prefill 杠杆。decode 回收路径修订为：C2g tilelang HC 边界融合（Phase 2 运行时）、
  decode 形状 fused index score（候选新 kernel）、全 stage 单 graph；MTP（Phase 4）
  另计 ~1.5×。其中 **C2g 已单独量化**
  （[`A5F`](experiments/A5F-hc-boundary-fusion/README.md)：decode 形状 B=512 下
  2.92×、省 461 µs/边界，数值 1 ulp bf16）——满接入估计回收 ~10 ms/stage，
  12.5k → ~15–16k，回到预估带内。
- 下一阶段（Phase 2）：**dsv4_direct 移植 Flash 层表**（单机 TP4×PP2 → 双机 PP4），
  契约层已移植并在真实分片上通过（`runtime/dsv4_direct/`，7 层型 PASS + 4 阴性
  对照）；加载层已移植并 smoke 通过（滑窗/ratio-4/ratio-128 × rank，itp 切片 +
  Marlin repack，MoE 常驻 862 MB/层/rank ≈ 9.3 GiB/11 层，与容量模型吻合）。
  前向四竖条全部移植完成并过真实权重 oracle gate：ratio-128（E0ef）、
  ratio-4/indexer topk 512（E0ff，148/148 exact）、TP4 MoE（E0cf，runtime 与
  手工 Marlin 路径位级一致）、滑窗层型 L0/L1（E0wf，Flash 新层类，无-YaRN RoPE +
  环形 KV）；整层装配完成（E0df：DirectDecodeBlock 三层型分派，黑盒前向与分量
  oracle 组合逐位相等，三层型全 PASS）。superstage + stateful CUDA graph 亦完成
  （E0sf：Flash graph family 与 Pro 同构的推导成立；6 层 stage 132 步 graph
  replay 与 eager 逐位相等，含 NCCL-in-graph 与 ratio-4×ratio-128 双边界）。
  **单机 TP4 多层 stage 已具真实权重对拍背书**。下一步：C2g HC 边界融合 A/B
  （语义变更走 oracle gate）→ 单机 TP4×PP2 → 双机 PP4 → E2E golden-token
  对拍 D0 oracle → 性能收敛验证（12.5k→15k+）。
  12.5k 为 reference-op 基线，暂不构成对 15–25k 的证伪，但若 Phase 2 集成后仍
  显著低于 15k，须按目标文档修正容量模型。
