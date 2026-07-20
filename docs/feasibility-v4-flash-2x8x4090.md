# DeepSeek-V4-Flash 在 2×8×RTX 4090 上的推理可行性评估

- 日期：2026-07-20
- 状态：评估结论（roofline + 校准外推，非实测）
- 参考实现：本仓库 `reference/`（DeepSeek-V4 官方 HF release，含 `inference/` 与 `encoding/`）
- 实验依据：`../gaiban`（DeepSeek-V4-Pro on 8×8×4090 的全部 A/B/C/D/E 系列实验）

## 0. 结论（TL;DR)

**可行，且是 gaiban 资产的"甜点级"落地目标。** V4-Flash（284B 总参 / 13B 激活，checkpoint
实测 148.6 GiB）在 16×24GB = 384 GiB 显存上放得下且余量充足（权重仅占 ~40%）。它与
V4-Pro 共享同一套 V4 架构（CSA/HCA 混合稀疏注意力、mHC、MXFP4 experts、FP8 主体、
hash 路由、MTP），gaiban 里 A1→E1 的绝大多数结论和 kernel 换几何参数即可复用，且
**几乎所有几何变化都朝对 sm89 有利的方向**（heads 减半绕开 smem 墙、hidden 缩小减通信、
专家数减少加快权重摊销、index_topk 减半）。

| 指标（8K ctx 为主） | 预估值 | 置信度 |
|---|---|---|
| decode-only roofline（16 卡合计） | **~15–25k output tok/s**（中心 ~20k） | 中，roofline 级 |
| prefill 吞吐 | **~30–40k input tok/s** | 中 |
| 8K/1K 混合 aggregate（共享 16 卡） | **~3.2–4.2k output tok/s** | 中低 |
| 4K/1K 混合 aggregate | **~5.5–7k output tok/s** | 中低 |
| 单用户 decode 速度 | ~20–30 tok/s（MTP 后有望 ~1.5×） | 低 |
| 8K bf16 KV 并发 | ~2,000–2,300 条活跃序列（FP8 KV ~2×） | 高，字节级可验 |

对照：**Flash 在 2 台机器上的 8K/1K aggregate 与 Pro 在 8 台机器上的容量模型
（~3.6–3.7k）基本同级** —— 硬件是 1/4，即单位成本产出约 4×，与激活参数比
（49B/13B ≈ 3.8）一致。逻辑自洽：decode 均为权重带宽瓶颈，吞吐 ∝ 总带宽 / 激活字节。

## 1. 模型与官方实现要点

### 1.1 V4-Flash 几何（`reference/config.json` + `reference/inference/model.py`）

- 43 层主干 + 1 MTP 层；hidden 4096；vocab 129280；1M 上下文（YaRN, 双 rope_theta）。
- MoE：256 routed experts（MXFP4, inter 2048）+ 1 shared expert（FP8）×每层，top-6，
  `sqrtsoftplus` 打分 + noaux_tc；**前 3 层 hash 路由**（`gate.tid2eid` 按 token id 查表，静态路由）。
- Attention：MQA 形态单 KV（`wkv: 4096→512`），64 heads × head_dim 512（含 rope 64）；
  Q 低秩（1024）；O 分组低秩（o_groups 8 × o_lora 1024，wo_a 为 bf16）；attn_sink；
  滑动窗口 128 + 压缩 KV 稀疏注意力：
  - `compress_ratios`：21 层 ratio-4（带 indexer，top-512 稀疏）+ 20 层 ratio-128
    （重压缩，全量注意）+ **L0/L1 与 MTP 为纯滑窗层（ratio 0）**。
  - Indexer：64 头 × 128 维，对全部 ratio-4 压缩位打分取 top-512；自带 Hadamard 旋转 +
    FP4 QAT 模拟的压缩 KV。
  - Compressor：ratio-4 为 overlap 门控池化，fp32 状态机；ratio-128 无 overlap。
- mHC：hc_mult=4（残差 4 份副本，**PP payload 为 4×dim**），Sinkhorn 20 轮。
- 量化：experts MXFP4（4bit + e8m0/32），其余 FP8 E4M3 128×128 block-scale（ue8m0），
  **KV 存储做过 FP8（nope 维 64-group）/ FP4（indexer）QAT 模拟** —— FP8/FP4 KV
  是训练认可的存储格式（model.py 内有明确注释）。

### 1.2 与 V4-Pro 的差异及 sm89 影响

| 参数 | V4-Pro（gaiban 目标） | V4-Flash | 对 sm89 的影响 |
|---|---|---|---|
| 层数 | 61 + 1 MTP | 43 + 1 MTP | — |
| hidden | 7168 | 4096 | AR/PP payload ×0.57，利好 |
| n_heads | 128 | **64** | TP4 后每卡 h=16，**绕开 101376B optin smem 墙**（Pro h=32 需 sparse_attn32 的 block=32 补丁），利好 |
| routed experts | 384 × inter 3072 | 256 × inter 2048 | 每层 expert 权重 12.9 → **~3.2 GiB**；B≈128 即近全专家命中，权重摊销更快，利好 |
| index_topk | 1024 | 512 | 稀疏 gather 减半，利好 |
| q_lora / o_groups | 1536 / 16 | 1024 / 8 | 小改 |
| compress_ratios | 30×r4 + 31×r128 | 21×r4 + 20×r128 + 2×r0 | 纯滑窗层是新 layer 类型但最简单（128-token ring buffer） |
| hash 层 | 3 | 3 | D5 已实现，直接复用 |

### 1.3 官方 kernel（`reference/inference/kernel.py`）定位

全部 tilelang（`tilelang==0.1.8`），显式关闭 TMA / warp-specialization（可在 sm89 编译）。
`fp4_gemm` 走"FP4 软解码 → FP8 MMA"：A1 已证明 sm89 上可编译、数值对
（rel_fro 1.66e-3）、但吞吐仅 ~17–25 GB/s（~2% 峰值，block_K=32/2-stage 的实现问题）。
`model.py` 的 MoE 是 per-expert Python 循环（A3 实测比 grouped Marlin 慢 38×）。

**定位：reference 实现 = 正确性 oracle（golden-token 对拍源），生产路径走 gaiban kernel。**

## 2. 显存与容量测算

### 2.1 权重（据 `model.safetensors.index.json`，total_size = 159,609,485,896 B = 148.6 GiB）

| 分类 | 体积 | 说明 |
|---|---:|---|
| routed experts MXFP4 | ~140.2 GiB | 44 层 × 256 × 3×4096×2048，283.5B 参数 |
| shared experts FP8 | ~1.0 GiB | |
| attention FP8/BF16 | ~4.4 GiB | wo_a 为 bf16 |
| 其余（gate/HC/compressor/embed/head 等） | ~3–4 GiB | compressor wkv/wgate 为 fp32 |

16 卡平摊 **~9.3–9.5 GiB/卡**；尾 stage（+MTP、+fp32 head）~10.5 GiB。
对比 Pro 的 12.9–13 GiB/卡更宽松。

### 2.2 KV cache（C3 字节公式换 Flash 几何）

- 每 token 增量：ratio-4 层 320 B（latent 512/4 + indexer 128/4，bf16）、ratio-128 层 8 B
  → 全模型 **6.72 KiB/token（bf16）/ ~3.4 KiB（FP8）**。
- 每序列常量：r4 层 ~213 KB、r128 层 ~655 KB（fp32 compressor 状态占大头，可降 bf16）。
- 每序列全模型：8K ≈ 54 MiB、32K ≈ 215 MiB、128K ≈ 860 MiB、1M ≈ 6.9 GiB（bf16）。
- 每卡 KV 预算 ≈ 24 − 10（权重）− ~2.5（graph/NCCL/workspace/激活）≈ **11–12 GiB**；
  每卡每 stage（11 层）@8K ≈ 19.5 MB/seq → **~500–580 seq/卡**，
  集群活跃并发（×4 DP）≈ **2,000–2,300 条 @8K bf16**，支撑全局 microbatch B≈512、
  in-flight 4 个 microbatch；FP8 KV（+compressor 状态 bf16 化）约翻倍。
- **1M 上下文单序列显存成立**（每卡每 stage ~1.9 GB），吞吐塌至个位数 tok/s
  （indexer O(ctx) 线性扫主导，与 Pro 的 C3/D0a 结论一致）——能力项而非运营点。

## 3. 部署形态

**推荐：PP4 × TP4（每 socket 一个 TP4 super-stage）+ DP-attention，与 gaiban T4 形态同构。**

```
node0 socket0: stage0 = embed + L0–L10          (TP4, ~9.5 GiB/卡)
node0 socket1: stage1 = L11–L21                 ← xGMI 一跳
node1 socket0: stage2 = L22–L32                 ← 100G IB 一跳（全管线唯一大 IB 跳）
node1 socket1: stage3 = L33–L42 + MTP + TP4 vocab head
                                                ← xGMI 一跳；采样 token id 回流 stage0（极小）
```

- expert 摆放沿用 A3b/A3c：**per-expert intermediate-TP**（每 rank 持全部 256 专家的
  inter/4=512 分片）；shared expert 同样 inter/4 切分，与 routed partial 合并进同一次
  reduce-scatter。
- PP 边界 payload（mHC 4×dim）：B=512 时 512×4×4096×2B ≈ 16.8 MB ≥ 1 MiB
  → 按 B2/B4 校准走 **GDR（~9 GB/s，~1.9 ms/跳）**，no-GDR（~4.2 GB/s，~4.2 ms）保底，
  与计算 overlap。
- 放弃项：
  - **TP16（reference 官方多机方式）**：44 层 × 每层多次 [B,4096] AR 走 IB
    （busbw 5.3–11.7 GB/s），仅 AR 即 >50 ms/step，不成立；
  - **TP8×PP2**：B1 实测 TP8 全口径 tax ~38%（TP4 仅 ~17%，Flash hidden 减半后预计
    12–15%）；
  - **TP2×PP8**：7 个 stage 边界 × E1 实测 ~2.7 ms/跳 handoff 罚金，流水更深。

## 4. gaiban 资产复用清单

### 4.1 直接复用（改配置/几何即可）

| 资产 | 路径（`../gaiban/`） | 说明 |
|---|---|---|
| P2P 驱动补丁 + 检查 | `experiments/B1-tp-allreduce/patch_p2p_driver.sh` 等 | 无补丁机内 AR 掉到 ~4 GB/s（1/4） |
| IB/NCCL 资格测试 | `experiments/rdma-nccl-qualification/`, `experiments/ib-latency/` | 双机组网 runbook |
| 直连 runtime 骨架 | `experiments/E0-direct-runtime/dsv4_direct/` | physical/fractional stage、stateful CUDA graph、pipeline overlap、static KV、checkpoint 加载、deterministic MoE align、distributed control —— 最大一块资产 |
| fused MHC | `experiments/C2-prefill/c2f_fused_hc.py` | hc_mult=4 相同，bitwise 等价 |
| fused Triton indexer | D0b 产物 | 索引头几何 64×128 完全相同，仅 topk 1024→512 |
| hash-gate（tid2eid） | D5 路径 | 相同 |
| weight-only FP8 attention 投影 | E1b2q 路径（`attention_w8a16_projection.py` 等） | 已过质量门的 1.74× 路径 |
| DP-prefill 结构 | C2 MVP3 / C2g | pre-expand gather + HC 边界融合 |
| 聊天模板编解码 | 本仓库 `reference/encoding/` | 服务化直接用 |

### 4.2 需 re-tune（同 kernel 族，换 shape）

- **grouped Marlin MXFP4 MoE**：384×(3072,7168) → 256×(2048,4096)；重跑 A3 bench 矩阵，
  预期仍 ~900+ GB/s（Pro 实测 916–929 GB/s ≈ 92% 峰值）；decode W4A16 / prefill
  W4A8-FP8 分工不变（W4A8 对 prefill 端到端仅 ~1.07×，不指望）。
- **A3c-v4 shared-expert FP8 kernel**（E8M0 128×128 block-scale 格式相同）。
- **sparse attention**：h=16/卡（TP4 后）、d=512、idx = 128 窗口 + 512 topk；比 Pro 的
  h=32 宽松，`sparse_attn32` 甚至 block=64 原版即可。
- **checkpoint 转换 + Marlin repack**（A3 `common.py` + `reference/inference/convert.py`）。

### 4.3 Flash 特有新工作（均小）

L0/L1 纯滑窗层；o_groups=8 的 wo_a 分组 einsum；MTP block 接入尾 stage
（D6.4 拓扑本就为本地 MTP 预留）。

### 4.4 明确不做/降级

- attention activation 原生 FP8（E1b2j–p 质量门全灭：pooled RMS / top-6 route 全部超门，
  且已归因到 activation 量化本身）——保持 BF16 计算 + weight-only FP8；
- FlashInfer sparse MLA（拒 sm89，SM100+/SM120+ only，结论不变）；
- tilelang fp4_gemm 调优（A1 Gate C 已判死，Marlin 替代）。

## 5. 性能预估与校准依据

### 5.1 校准锚点（全部为 gaiban 实测）

| 锚点 | 数值 | 出处 |
|---|---|---|
| grouped Marlin MoE decode 有效带宽 | 916–929 GB/s（~92% 峰值） | A3 |
| 单 stage 集成路径可兑现 roofline | E1a27：22.205 ms@B240 = 10,808 tok/s（目标 10,800） | E1a |
| 全流水 TP4×PP2 BF16 全口径 | E1b2w：23.03 ms/token（≈600 GB/s/卡 all-in） | E1b |
| 机内 AR（TP4, [B,7168] bf16） | B128 161 µs / B256 266 µs；D2D 25.2 / 22.7 GB/s | B1 |
| 机间 IB | no-GDR 大包 ~4.0–4.2 GB/s；GDR ~8.8–9.3 GB/s | B2/B4, rdma-nccl-qualification |
| PP handoff 罚金（未闭环） | ~2.65–2.88 ms/跳 | E1b2z/E1b3 系列 |
| Pro decode/prefill roofline | 10.8k(8K)/17k(4K) out；44–46k in（64 卡） | D0a/D0b/D4 |

### 5.2 Flash 单 stage 分解（11 层，TP4，B_micro=512 global，8K ctx）

| 分项 | 估计 | 依据 |
|---|---:|---|
| MoE 权重流（~9.4 GiB/卡 @ ~900 GB/s） | ~10.8 ms | A3 带宽 |
| KV/compressor 读写（B_local=128 × ~7.3 MB/seq/stage） | ~1.7 ms | C3 字节 + 几何 |
| DP gather/reduce-scatter（11 层 × [512,4096]） | ~4.4 ms | B1 换算 |
| attention + indexer + HC + 固定开销 | ~6–7 ms | A4/C1 折算（h 减半） |
| PP handoff 摊销 | ~2 ms | E1b 罚金 + overlap |
| **t_stage 合计** | **~24–27 ms** | |

→ decode ≈ 512 / t_stage ≈ **19–21k tok/s**；区间 **15–25k** 覆盖 B∈[384,768] 与
handoff 兑现程度。聚合带宽利用率 ≈ 40%（对比 Marlin 裸 92%，差额在通信/attention/固定
开销，与 Pro 的 E1 全口径一致）。

### 5.3 其他工作点

- **prefill**：Pro 实测 ~700 in-tok/s/卡 × 激活比 3.8 × 注意力折扣（h 减半、topk 减半、
  sm89 占用率墙缓解）→ **~30–40k input tok/s**（16 卡）。
- **8K/1K 单池混合**（1/T = 1/D + 8/P）：D=16–20k、P=35–40k → **~3.2–4.2k output tok/s**；
  4K/1K → **~5.5–7k**。
- **单用户**：B=1 时每层仅读 7 个小专家（~93 MB/层，TP4 后 ~23 MB/卡），固定开销主导
  → ~33 ms/token ≈ **~30 tok/s**（Pro 仅 3.8，Flash 专家小是本质改善）；MTP 后 ~1.5×。
- **长上下文**：128K 并发 ~180 条、~2–3k tok/s；1M 单序列 ~10 tok/s（能力项）。

### 5.4 不确定性声明

1. 以上为 roofline + 校准外推，**非端到端实测**；gaiban 在 Pro 上也尚未跑通 closed-loop
   全模型吞吐。
2. E1 的 PP handoff 罚金在 Pro 上仍未解决（E1b3j interleaved placement 修复中）；
   Flash PP4 有 3 跳，若 overlap 不掉会吃 ~10–25% 吞吐。
3. 11 层/stage 的 CUDA graph 比 Pro 4 层/stage 长 ~2.75×，固定开销估计 ±30%。
4. prefill/decode 混跑干扰未建模（保守再打 9 折）。
5. FP8 KV 是容量杠杆而非速度杠杆：D0a 实测 sm89 attention 侧 FP8 KV 慢 1.4–4.3×，
   需在 Flash h=16 几何上重测。

### 5.5 实测修订（2026-07-20）

**分解表修订**（对照 §5.2 的预估，依据 experiments/C1F-integrated-block、runtime E0hf/E1F 实测）：

| 分项 | §5.2 预估 | 实测/修订 | 出处 |
|---|---:|---:|---|
| MoE 权重流（11 层/stage） | ~10.8 ms | ~12.1 ms（inter=512 切片实效 687–760 GB/s） | A3F/C1F |
| DP 通信 | ~4.4 ms | allgather 6.0 + AR 3.2 = 9.2 ms（原表漏 allgather） | C1F |
| attention+indexer+HC | ~6–7 ms | attention ~10.3 ms + HC eager ~7.8 ms；fused HC 边界（C2g）后 HC 回收 ~4.7 ms/stage | C1F/E0hf |
| PP handoff | ~2 ms | 机内/跨机 0.9–6.9 ms 发送侧（bl 依赖），no-GDR 非瓶颈 | E0qf/E1F |

**满配实测锚点**（E1F，graph 化、fused HC、serial closed-loop、复制口径）：四 stage replay
均衡（<15% 差）；bl=128 时 max-stage 38.2 ms → DP+满流水换算 12.7–13.4k tok/s
@B_global=512（修正后预估 ~14k，带内）；bl=192 → 14.9–15.5k @768（§5.2 原始带 15–25k
下沿可及）；ctx 8K 较 2K 慢 ~4.6%；复制口径容量上限 bl=192。单用户 B=1 实测
27.5 tok/s（预估 ~30，MTP 前）。

**语义收官**：direct runtime 满配 E2E 对 D0 golden tokens 468/482=97.1%，全部 14 处
分歧为近平局翻转（golden deficit ≤0.94 vs 判决余量中位 6.67），语义等价成立；
fused HC 边界融合放行（与 eager 同分率）。

**表述纪律**：E1F 的 12.7–15.5k 为"由实测 stage 时间做的 DP+满流水换算"，实测
closed-loop（serial 单 microbatch）最高 920 tok/s@bl=192；真 DP-attention 序列切分与
≥4 microbatch 交织实现后方可作为 E2E 实测吞吐引用。

## 6. 风险清单

| 风险 | 等级 | 缓解 |
|---|---|---|
| PP handoff 罚金未闭环（E1 遗留） | 中 | Flash 仅 3 跳且 t_stage 更长（罚金占比≈Pro）；E1b3j 修复直接移植 |
| 长 stage（11 层）固定开销放大 | 中 | 全 stage CUDA graph（含 NCCL collective，C1 已验证）；decode 是 launch-bound，graph 收益 4–6× |
| 数值质量（MXFP4+FP8+fp32 compressor 链路） | 中 | D5-canary 逐层对拍 + E1 冻结质量门方法论；attention 保持 BF16 计算 |
| 2 节点无 P/D 分离空间，prefill 挤占 decode | 低中 | chunked prefill 交错（C2g 已验证）；必要时按 4K/1K 运营 |
| sm89 prefill sparse attention 天花板（D3/D0c） | 低 | Flash h=64、topk=512 使占用率墙明显缓解；prefill 非首要瓶颈 |

## 7. 实施路线图（约 6–9 周至可基准测试）

1. **Phase 0 – 环境与转换**：✅ 环境与权重部分已于 2026-07-20 完成（见附录 B）；
   剩余：B1/RDMA 双机实测标定复跑、Flash 版 convert + Marlin repack、用 reference
   实现在 sm89 上建 golden-token oracle（慢但数值对）。
2. **Phase 1 – kernel regear**（~1–2 周）：Marlin MoE / shared FP8 / sparse_attn(h=16) /
   fused indexer 四件套换几何，重跑 A3/A4/C1 级 bench，确立 Flash 单层数字。
3. **Phase 2 – 单机 TP4×PP2**（~1–2 周）：`dsv4_direct` 移植 Flash 层表（22 层/node），
   D5 式逐层 canary，CUDA graph，对齐 t_stage 模型。
4. **Phase 3 – 双机 PP4**（~1 周）：IB 接入（先 no-GDR 后 GDR），攻 handoff overlap。
5. **Phase 4 – 服务化与调优**（~2–4 周）：continuous batching + chunked prefill 交错 →
   固定序列基准（InferenceX 口径）→ FP8 KV 容量开关 → MTP 投机解码（decode 预期再 ~1.5×）。

## 附录 A：关键几何/字节速查

- 每层 routed expert 权重：256 × 3 × 4096 × 2048 × (0.5 + 1/32) B ≈ **3.2 GiB**（MXFP4）。
- 每卡每 stage 权重（11 层 TP4）：≈ 9.4 GiB；checkpoint 总量 148.6 GiB / 16 卡 = 9.29 GiB。
- KV 每 token：21×320 B + 20×8 B = **6.88 KB（bf16）**；每序列常量 ~9–14 MB（fp32
  compressor 状态可降 bf16 减半）。
- PP payload：4 × 4096 × 2 B = **32 KB/行（bf16）**；B=512 → 16.8 MB/跳。
- 专家命中（coupon-collector, 256 experts top-6）：B=64→~200、B=128→~243、B=256→~255。
- decode 每 token 全模型读量（B=512, 8K）≈ 权重摊销 ~300 MB + KV ~30 MB。

## 附录 B：实验环境清点（2026-07-20 核实并部署完成）

### B.1 机器与网络

| 项 | titan064（10.234.1.64） | titan065（10.234.1.65） |
|---|---|---|
| GPU | 8× RTX 4090 24GB | 8× RTX 4090 24GB |
| 驱动 / CUDA | 590.48.01 / CUDA 13.2 toolkit | 同左 |
| CPU / NUMA | 2× EPYC 7773X，2 NUMA node，4+4 卡/socket（GPU0-3 socket0，GPU4-7 socket1） | 同左 |
| 内存 / 磁盘 | 1TB RAM；NVMe 余 ~530G（传完权重后） | 同左 |
| **P2P 补丁** | ✅ 已生效（`nvidia-smi topo -p2p w` 全对 OK，含跨 socket）| ✅ 同左 |
| GDR 路径 | home 下有 `libcuda-onebyte-patch`（DMA-BUF opt-in）；无 OFED / nvidia_peermem（gaiban 路线不需要） | 同左 |
| IB | CX-5（MT4119）100G，口 Active，NIC 挂 NUMA1（GPU6 PHB） | 同左 |
| memlock | ~126 GiB（RDMA 足够） | 同左 |

与 gaiban 的 dsv4exp（titan052）完全同构；`nvidia-smi topo` 确认 NIC 亲和 socket1，
PP stage 落位时 IB 边界 stage 优先排 socket1。

网络事实：
- 双机同一 IB fabric（SM lid 均为 1，Base lid 18/22）；内网无防火墙，双机 TCP/ssh
  互通已验证（注：titan 系统未装 `ping`/iputils-ping，用 ping 测连通会误报）；
  IPoIB netdev DOWN（verbs 不需要）。
- **earth**（权重仓库机）：与 titan 同内网 `10.234.1.151:22`，10G NIC；
  titan→earth 实测拉取 ~110 MB/s/机。titan 的 `~/.ssh/id_ed25519`
  （comment `dsv4-flash-transfer-*`）已加入 earth 的 authorized_keys。
  （外网入口 earth.s.cysic.work:2226 仅供外网的开发工作站使用。）
- 机器均在中国大陆：**pip 走 huaweicloud 镜像**（已写入两台
  `~/.config/pip/pip.conf`）；GitHub 下载经本地工作站中转（FHT 构建脚本模式）。
- 开发工作站（外网）的 ssh 访问 titan 均经 **ProxyJump earth** 中转；已配置别名
  `titan064`/`titan065`（与 IP 形式等价、共享 ControlMaster 复用连接，
  复用后单命令延迟 ~0.27s vs 首连 ~2.7s）。

### B.2 软件栈（venv `~/Workspace/venvs/sglang`，两台一致，导入全套验证通过）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python / torch | 3.12 / **2.11.0+cu130**（识别 8 卡，NCCL 2.28.9） | |
| vllm | **0.22.1** | 仅作 Marlin MXFP4 kernel 库（`_custom_ops`、`marlin_utils_fp4`）；与 dsv4exp 验证组合一致 |
| tilelang | **0.1.8** | reference/sglang 钉住版本 |
| flashinfer | 0.6.12（python+cubin） | sglang 钉住版本 |
| fast_hadamard_transform | **1.1.0 sm89 自编译** | gaiban C1 脚本构建，GPU fp16/bf16/fp32 自测通过 |
| sglang | dev（editable, `~/Workspace/sglang`） | gaiban Pro 时代部署，保持可用（正确性 oracle 用途） |
| tokenspeed-mla / llguidance | 0.1.6 / 0.7.30 | sglang 钉住版本（vllm 安装曾改动，已恢复） |

**tilelang JIT 环境要求（2026-07-20 实证）**：必须 `export CUDA_HOME=/usr/local/cuda-13.2`
并把 `$CUDA_HOME/bin`、`$CUDA_HOME/lib64` 置于 PATH/LD_LIBRARY_PATH 前，否则 tilelang
取 venv 内 pip 的 `nvidia/cu13/bin/nvcc`（同为 13.2.78 但头文件路径不兼容）报
"CUDA compiler and CUDA toolkit headers are incompatible"。gaiban 各 `*_titan.sh`
亦同此做法。

**已知无害警告**：pip 会报 vllm 0.22.1 声明依赖（tilelang==0.1.9、
flashinfer==0.6.11.post2、tokenspeed-mla==0.1.2、llguidance>=1.7）与现装不符——
刻意保持 sglang 钉住版本优先；vllm 仅 kernel 库用途，其 serving 栈不运行。
若未来需要跑 vllm serving，另建独立 venv（dsv4exp 的 `dsv4-experiments/.venv` 模式）。

两台 `~/Workspace/` 还留有 gaiban Pro 时代资产（`dsv4-runtime-package`、
`dsv4-16stage-pp-skeleton`、`dsv4-checkpoint-stages` 104G/67G 等）——**均须保留，
不得删除**（Pro 实验仍在进行）；磁盘余量（~530G）对 Flash 工作足够，无需腾盘。

### B.3 权重（两台一致，逐分片校验通过）

- 路径：`~/Workspace/DeepSeek-V4-Flash/`（源：earth `/big/harry/llm/DeepSeek-V4-Flash/`）。
- 校验（2026-07-20，earth 源与两台副本三处一致）：46/46 分片；每文件
  `大小 = 8 + header + data` 严格相等；张量总字节 = **159,609,485,896** 与 index
  total_size 精确相等；69,187 个张量无缺无多；config/tokenizer md5 与本仓库
  `reference/` 一致（config `7f34d4…`，tokenizer `3f75db…`）；确认为 V4-Flash
  FP4 版（hidden 4096 / 43 层 / 256 experts / expert_dtype=fp4）。
- 该校验为结构完整性（头部+字节数），未与上游做全量哈希比对；bit 级完整性由
  Phase 0 的 golden-token 对拍兜底。
