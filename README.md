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
  **单机 TP4 多层 stage 已具真实权重对拍背书**。C2g HC 边界融合已接入为可选 backend
  （默认 eager 不变；per-layer gate PASS，stage 级发散在 1-ulp 路由敏感度包络内；
  成对计时实测 6 层 stage −2.59 ms/步@bl=128、−6.4 ms/步@bl=512，11 层 stage 按
  C1F 工作点回收 ~4.7 ms → decode 预估修正为 **~14k**，fused 放行与否由模型级
  canary 裁决）。单机 TP4×PP2 缩尺管线亦通过（E0pf：8 卡两 stage 出口 264/264 步
  逐位、KV digest 一致、handoff 0.23 ms/32KiB 机内；E1b2z 现役 NCCL 机制 +
  staged D2D unpack）。双机跨机管线亦逐位通过（E0qf：跨机集合实测
  bitwise 决定性，no-GDR/GDR 双配置 264/264 步逐位，handoff 0.24 ms/32KiB）。
  **E2E 收官达成**：43 层满配 PP4×TP4（双机 16 卡）+ embed/head 完整模型
  decode 对 D0 golden tokens 匹配 **468/482 = 97.1%**，全部 14 处分歧均为近平局
  翻转（golden deficit ≤0.94，vs 正常判决余量中位 6.67），语义等价成立；
  **fused HC 边界融合放行**（与 eager 同分率，482 中仅 5 处近平局互异）；
  ratio-4 层低位置路径已补齐（带实权重预门）。报告性时延 B=1 无 graph：
  eager 85 ms/步、fused 55–57 ms/步。满配 graph 化吞吐已量化
  （[`E1F`](experiments/E1F-full-decode-throughput/README.md)：graph vs eager
  132/132 逐位×4 stage；复制口径 bl=192 实测 920 tok/s closed-loop；由实测
  stage 时间换算 DP+满流水 **12.7–13.4k @B_global=512、14.9–15.5k @768**——
  修正后模型 ~14k 获实测支持，原始带 15–25k 下沿在 768 运营点可及）。
  真 DP-attention 已实现并过 gate（E0dpf：runtime 集合序本就是 DP 形态，
  真 DP=喂不同序列+KV/4；graph 132/132 逐位，漂移在复制口径自有 lane 噪声带内；
  **DP 实测 max-stage replay 坐实换算分母**：B_global=512 达成 103–104%、768 达成
  100%；serial closed-loop 实测 3259/3679 tok/s；KV 杠杆兑现，8K 折扣实测 ~11%，
  768+8K 差 222MiB OOM）。交织流水已实现（E1IF：无串扰逐位 gate,瓶颈 stage 94% busy）,
  **吞吐-容量前沿已实测定界并触发容量模型正式修正**（E1F README）：
  8K/bf16-KV decode-only 实测天花板 **~6.4k tok/s**（B_total=640,显存限而非
  算力限）、2K 达 8.6k;原 15–25k 带的两个假设被证伪——t_stage 线性缩放
  （实测强次线性）与管线填充 KV 同驻（×4 漏算）。**回收路径：FP8 KV（×2 行数,
  需 Flash 几何重验）+ MTP（~1.5×）投影 ~15–16k,带下沿在两杠杆齐备时可及**;
  短 ctx 运营另有余量。**FP8 KV 已判活并放行**（A6F：torch 路径罚金 0.99–1.08×,Pro 的 1.4–4.3×
  是 tilelang dequant 特有；集成后质量 gate 全过,E2E 467–470/482 与基线不可
  区分,**8K 前沿 6392→7523 tok/s +17.7%**）;容量瓶颈已转移到 fp32 attention
  工作区 + graph 私有池（新杠杆:工作区瘦身）。**MTP 已接入且协议无损**（oracle+E2E 硬验收全过,
  free-run 输出与 off 完全一致;接受率 **0.86**,B=1 实效 1.30–1.34× eager 实测、
  graph 投影 ~1.4×;chained 形天然填 PP 空泡）。工作区瘦身完成
  （pool 共享 +2.5 GiB、sparse core 逐位精确瘦身 +4.6% 副收益、半精度累加
  数值门 FAIL 弃用）：**8K 前沿 7523 → 8733 tok/s(+16.1%,bl72)**,bl80 墙
  已是权重+KV 结构容量。大 B MTP graph 化完成但 **×1.4 投影被证伪**
  (满流水中无空泡可填,每轮付 2 slot;8K 持平 1.00×(省 11% KV)、
  **2K +13% → 9656 tok/s 新高**;无损性 force-reject 全逐位证据链完整,
  失步臂分歧归因到 Marlin 批组成 ULP 敏感,任何投机形态共有)。
  **当前实测最优:8K 8733、2K 9656 output tok/s**。带下沿 15k 的剩余差距
  ~1.7×@8K,候选:seqlen-2 融合 verify(需先解无损)、MTP head GEMM 优化
  (10ms/轮)、KV 行宽、handoff overlap、短 ctx 运营。
- Prefill(C2F + 重归因):**dense-BF16 MoE 路线证伪**(慢 4–17×;Marlin 在
  prefill 已跑到 135 TFLOPS ≈ 4090 BF16 峰值 82%,C2F 的 "MFU 11.5%" 归因有误),
  且用 C2F 自己的脚本复跑发现**基线被低估 41–45%**:baseline **15.0k**、
  全开(W4A8+fused indexer)**16.6k** input tok/s/stage(差额全部在 MoE 桶,
  归因为 fp32 临时量的分配器抖动,已记为可修缺陷 ≈ +32%)。修正后的 prefill
  归因:**attention 58%**(ratio-4 41% + ratio-128 17%)、MoE 24%、HC 16% ——
  **prefill 由 attention 主导**,而 runtime 至今用 torch masked-einsum 正确性
  实现,reference 的 tilelang sparse kernel 从未接入。单池 T 投影 **1.68k**
  (原 1.2k);够到 3.2–4.2k 带需 P≈40k,路径 = attention 换真 kernel(3× 则
  ~27k)+ prefill HC 融合 + MoE 临时量预分配。
  12.5k 为 reference-op 基线，暂不构成对 15–25k 的证伪，但若 Phase 2 集成后仍
  显著低于 15k，须按目标文档修正容量模型。
- **Prefill 基线与杠杆已实测**
  ([`C2F`](experiments/C2F-prefill/README.md)):单 stage(TP4、11 层、DP4
  口径)chunked prefill 基线 **10.7–11.7k input tok/s**(chunk 1024–8192 近
  平坦);两个 §4 杠杆接入并过数值门——W4A8 Marlin 端到端仅 1.02–1.03×
  (§4.2 "~1.07× 不指望"成立)、D0b fused indexer 30→2.5 ms/层@8192 但端
  到端 1.043×;全开 **11.4k(1.068×)**,E2E 回归两臂均 468/482(fused indexer 逐
  prompt 与基线全同;W4A8 分歧仍全为近平局)。对照 §5.3 模型(30–40k)差 ~3×,归因:Marlin 大 M
  MFU ~11.5%(48% 时间)+ eager torch attention(40%)——**prefill kernel
  竖条(MoE 换 BF16 dense/cutlass 执行形态、稀疏 attention kernel、HC 融合)
  是 P 侧主路径**,roofline P≈65–80k 说明 30–40k 仍可达。单池投影
  (D=8733):当前 P → **T≈1.2k**,带下沿 3.2k 需 P≈40k。
