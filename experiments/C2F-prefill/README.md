# C2F — direct runtime prefill 基线 + 两个 prefill 专属杠杆(W4A8 Marlin / D0b fused indexer)

第十九竖条(2026-07-21,titan064 实测)。decode 8K 前沿已实测 8733 tok/s
(E1F + 工作区瘦身);第二指标 8K/1K 单池 3.2–4.2k 按 `1/T = 1/D + 8/P` 需要
**P ≈ 35–40k input tok/s**——prefill 是当前主战场。本实验建立 direct runtime
的单 stage prefill 基线,并接入可行性 §4.1/§4.2 点名的两个 prefill 杠杆
(W4A8-FP8 Marlin MoE;D0b fused Triton indexer,`_FUSE_MIN_SEQLEN=1024`)。

## 结论速览

- **基线(单 stage TP4 11 层,DP4 口径)≈ 10.7–11.7k input tok/s**,对 chunk
  1024→8192 基本平坦(O(s²) 项在 8K 内尚不主导)。
- **W4A8 Marlin**:MoE 分量 1.05–1.11×,端到端仅 **1.02–1.03×**——可行性
  §4.2"仅 ~1.07×,不指望"在 Flash 几何 + runtime 内成立。数值门 PASS
  (层级 rel_fro 3.79e-2,与 A3F 的 reference-tilelang 量级一致)。
- **D0b fused indexer**:kernel 级 30.1 ms → 2.45 ms/层 @8192(12.3×),
  端到端 **1.043× @8192**(短 chunk 中性)。数值门 PASS(全部 chunk×层
  masked top-k **逐行集合完全一致**,score 最大差 ≤ 4.8e-7)。
- **全开 @8192:2.870 s → 11,417 input tok/s(1.068×)**,两杠杆近似可加。
- **对照 §5.3 模型(30–40k):实测约为其 1/3**,折合 713 in-tok/s/卡——
  恰好回到 Pro 水平,§5.3 期望的 3.8× 激活比收益没有兑现。归因:当前执行体
  是 eager torch 正确性路径 + decode 形 Marlin,不是 prefill 级 kernel
  (量化见"归因"节);roofline 显示 30–40k 仍可达,但需要 prefill kernel
  竖条,而不是本次这两个"接线级"杠杆。
- **单池投影(D=8733)**:P=11.4k → **T ≈ 1.23k**,远低于 3.2–4.2k 带;
  带下沿 3.2k 在 D=8733 下需要 **P ≈ 40k**(当前 P 的 ~3.5×)。
- E2E 硬门(e0ef2e 短 prompt 回归,改动臂):**两臂均 PASS**——fused
  indexer 臂 468/482,逐 prompt 与冻结基线完全一致;W4A8 臂(全 MoE 调用
  W4A8,比部署形态更严)468/482,总分持平,分歧仍全为近平局翻转。

## 口径(所有数字共用)

- **chunk = 单次 start_pos=0 整段 prefill 的序列长**(B=1/lane)。等价于 L 长
  序列 chunked prefill 的**末段** chunk 成本(O(s²) 项取到端点),与 gaiban
  D0b 的 "chunk" 定义一致;首段更便宜,故按本口径换算的全序列 prefill 吞吐
  是**保守下界**。
- **DP 形态**:4 条 TP lane 各喂一条**不同**的 B=1 合成序列(per-rank seed),
  一次 stage pass 处理 4×chunk 输入 token;attention 每 lane 全 64 头本地算,
  MoE all-gather 4×chunk 行 → 本 rank inter=512 切片 → reduce-scatter
  (runtime 原生 DP-attention + intermediate-TP 集合序,未改)。
- **input tok/s/stage = 4×chunk / stage_pass_wall**(host wall 夹
  `torch.cuda.synchronize` + `dist.barrier`,warmup 2 + timed 5 取 p50);
  eager、无 CUDA graph、无 PP handoff、无 decode 混跑——**open-loop 单 stage**。
- **满配 16 卡投影 P ≈ 4×chunk / t_stage**:PP4 chunk 流水稳态下整机吞吐 =
  单 stage 吞吐(4 个 stage 并行各处理不同 chunk,即相对"4 卡跑 43 层"的
  ×4 换算)。前提:4 个 stage 时长相近(实测对象 stage 1 = L11–L21,
  ratio-128×6 + ratio-4×5,全 learned 路由;stage 0 的 2 个滑窗层更便宜、
  stage 3 是 10 层 + head,近似成立),且未计 PP 边界 handoff 与
  prefill/decode 混跑干扰(§5.4 项 4 的 9 折未打)。
- 输入为合成 BF16 残差(scale 0.02;gate 输入经 rms_norm 归一)。吞吐口径;
  语义门单列(下文)。

## 方法

- bench:[`runtime/c2f_prefill_stage_bench.py`](../../runtime/c2f_prefill_stage_bench.py)
  (torchrun --standalone 4 ranks,titan064 GPU0–3),每个 (chunk, moe,
  indexer) 组合独立进程:load → 位级单元检查 → warmup 2 + timed 5 →
  分量 instrumented pass(单独一遍,不入 headline)→ 可选数值门。launcher
  [`runtime/run_c2f_prefill_titan.sh`](../../runtime/run_c2f_prefill_titan.sh)
  (前后 nvidia-smi 快照入 log)。
- **ratio-4 prefill >128 token**:补齐 reference 的 ring wrap-on-prefill 分支
  (model.py:518–523;`ratio4_fullpos.py` 改 index_copy_ 形式的 ring 写入),
  in-bench 与 reference 公式在 5 组 seqlen 上逐位对拍 PASS。prefill attention
  本身不读 ring(读全序列 raw + compressed),wrap 只影响后续 decode 步。
- 为让 8192 chunk 装进 24 GB,对既有语义路径做了三个**位级等价**的内存变换
  (均带 in-bench 位级单元检查,16/16 run 全 PASS):
  1. prefill 稀疏核 query 行分块(ratio-4 `sparse_row_block=1024`;ratio-128
     经 `DSV4_PREFILL_SPARSE_ROW_BLOCK` 环境变量,默认关闭)——行独立
     (逐行 softmax),分块与整段逐位相等;否则 FP32 gather 工作区
     [1,8192,640,512] 单层即 10.7 GB;
  2. ref indexer 打分链 relu/mul 原位化 + s 行分块(`_ref_index_scores`,
     [b,s,h,t] fp32 临时量 ≤ 1 GiB;否则 8192 时 4.3 GB ×3);
  3. **11 层共享一套 Marlin per-shape 缓冲**(`TP4MoE(buffer_donor=...)`,
     层间 MoE 严格串行;32768 行时 cache13 1.6 GiB/层 ×11 → ×1,不共享则
     加载即 OOM)。
- **W4A8**:加载侧 `ops/marlin_moe.py::_prepare_one_mxfp4` 参数化
  (`is_a_8bit` repack + `mxfp4_marlin_process_scales(input_dtype=FP8)`,
  镜像 A3F `common.py::_prep_one`);运行侧 `TP4MoE(marlin_input_dtype=FP8)`
  打开 `_fused_marlin_moe(input_dtype=...)` 并预置 per-token FP8 激活量化
  (A3F `preset_w4a8_quant` 同款)。W4A16/W4A8 重排布局不兼容且驻留字节相同
  (861,931,008,容量中性),同一权重只驻一份 → 本形态是 **prefill 专用
  引擎的加载期开关**(P/D 分离或按相位专机),不是同进程动态切换。
- **fused indexer**:[`dsv4_direct/ops/indexer_fused.py`](../../runtime/dsv4_direct/ops/indexer_fused.py)
  (D0b kernel 原样移植)接进 `Ratio4FullPositionAttention(
  index_score_mode="fused", fuse_min_seqlen=1024)` 的 prefill 打分路径;
  decode(seqlen=1)与短 chunk 一律走 ref。`paired_gate` 模式在同一输入上
  双算 ref/fused 并记录计时 + score 差 + masked top-k 逐行集合一致性。

## 结果(titan064 GPU0–3;torch 2.11.0+cu130,vllm 0.22.1,triton 见 JSON)

主表(wall = stage pass p50,5 次;tok/s = 4×chunk/wall;T 投影见下节):

| chunk | MoE | indexer | wall p50 (s) | input tok/s(DP4/stage = 16 卡投影) | vs 基线 | 峰值显存/卡 |
|---:|---|---|---:|---:|---:|---:|
| 1024 | W4A16 | ref | 0.350 | 11,717 | 1.000× | 18.0 GB* |
| 1024 | W4A16 | fused | 0.350 | 11,703 | 0.999× | 14.9 GB |
| 1024 | W4A8 | ref | 0.340 | 12,048 | 1.028× | 14.9 GB |
| 1024 | W4A8 | fused | 0.336 | 12,180 | 1.039× | 14.9 GB |
| 2048 | W4A16 | ref | 0.755 | 10,848 | 1.000× | 16.4 GB |
| 2048 | W4A16 | fused | 0.747 | 10,961 | 1.010× | 16.4 GB |
| 2048 | W4A8 | ref | 0.739 | 11,088 | 1.022× | 16.4 GB |
| 2048 | W4A8 | fused | 0.725 | 11,302 | 1.042× | 16.4 GB |
| 4096 | W4A16 | ref | 1.507 | 10,872 | 1.000× | 17.6 GB |
| 4096 | W4A16 | fused | 1.470 | 11,142 | 1.025× | 17.6 GB |
| 4096 | W4A8 | ref | 1.472 | 11,134 | 1.024× | 17.6 GB |
| 4096 | W4A8 | fused | 1.436 | 11,406 | 1.049× | 17.6 GB |
| 8192 | W4A16 | ref | 3.066 | 10,688 | 1.000× | 20.4 GB |
| 8192 | W4A16 | fused | 2.938 | 11,152 | 1.043× | 20.4 GB |
| 8192 | W4A8 | ref | 3.007 | 10,898 | 1.020× | 20.4 GB |
| **8192** | **W4A8** | **fused** | **2.870** | **11,417** | **1.068×** | 20.4 GB |

\* 该 run 额外带 --gate-indexer/--gate-w4a8(门自身占显存);同臂无门 run 14.9 GB。

分量墙钟(instrumented pass,@8192,单位 s;插桩自身 +~9%,只作占比参考,
headline 一律取无插桩 pass):

| 分量 | 基线 (W4A16/ref) | 全开 (W4A8/fused) | 每层 |
|---|---:|---:|---|
| MoE(11 层,32768 全局行) | 1.439 | 1.368 | 131 → 124 ms |
| attention ratio-4(5 层) | 0.955 | 0.819 | 191 → 164 ms |
| attention ratio-128(6 层) | 0.328 | 0.329 | 55 ms |
| HC(hc_pre/hc_post ×22) | 0.313 | 0.313 | 28 ms/层 |
| rms_norm | 0.032 | 0.032 | — |
| 合计(instrumented) | 3.067 | 2.862 | |

杠杆分量归因:fused indexer 省 128 ms/pass(30.1→2.45 ms/层 ×5 ratio-4 层,
kernel 级 12.3×;gaiban D0b 在 Pro 几何/bf16 链上为 ~53 ms/层,本 runtime
ref 是 fp32 分块链故为 30 ms/层),W4A8 省 59–71 ms/pass;两者可加
(196 ms 实测 ≈ 199 ms 分量和)。W4A8 的 MoE 级增益随行数递减:
1.11×@4096 全局行 → 1.066×@8192 → 1.05×@16384+。

## 数值门

| 门 | 结果 |
|---|---|
| ring wrap-on-prefill vs reference 公式(5 组 seqlen,含 >128 wrap) | **PASS(逐位)**,16/16 run |
| prefill 稀疏核行分块 vs 整段 | **PASS(逐位)**,16/16 run |
| ref 打分行分块 vs 整段 | **PASS(逐位)**,16/16 run |
| fused indexer vs ref(paired_gate,1024/2048/4096/8192 × 5 层,同输入) | **PASS**:masked top-k **逐行集合完全一致**(8192 时 8192/8192 行);score 最大绝对差 ≤ 4.8e-7(|score|max ≈ 3–5) |
| W4A8 vs W4A16 层级 A/B(L11,同 hidden,1024/2048 行/rank) | rel_fro **3.79e-2**、max|Δ| 0.021、有限性 OK——与 A3F 数值门量级一致(A3F:W4A8 vs fp32 oracle 4.6e-2,即 reference tilelang 量级);路由同构(gate 权重与输入相同) |
| **e0ef2e E2E 硬门:fused indexer 臂**(fuse_min_seqlen=8,短 prompt prefill 强制走 fused) | **PASS:468/482,逐 prompt 匹配数与冻结 eager 基线完全一致**(2/27/127/124/11/22/32/123);mismatch 仍为 14 处近平局(deficit max 0.936);lane 标志与基线相同 |
| **e0ef2e E2E 硬门:W4A8 臂**(全部 MoE 调用走 W4A8——比"仅 prefill W4A8"的部署形态更严) | **PASS:468/482,总分与基线持平**;仍是 14 处近平局翻转(deficit 中位 0.27、最大 1.24,较基线包络 0.94 略高但仍比正常判决余量中位 6.67 低一个量级);逐 prompt 构成小幅移动(±2,如 27→29、123→121),符合真量化语义变更的预期形态 |

fused indexer 的 top-k 逐行全等强于预期:indexer 值域是 FP4 量化后的离散值,
fp32 求和序差异几乎处处不改变 top-512 集合。

W4A8 臂说明:同一权重只能驻一种重排,故该臂 decode 步也走 W4A8,是比部署
形态(prefill W4A8 / decode W4A16)**更严格**的门;判读标准沿用 E2E 语义
等价准则(分歧须为近平局翻转,幅度在既有 14 处的包络内)。

## 与 §5.3 模型(30–40k)的差距归因

实测全开 11.4k ≈ 模型带下沿的 **0.36×**;折合 **713 in-tok/s/卡**,即回到
Pro 的 ~700/卡,§5.3 所乘的 3.8× 激活比收益没有兑现。差距不在物理,在执行体
形态——分项量化(@8192 全开,2.862 s):

1. **MoE 1.368 s(48%)——Marlin 在 prefill M 下 MFU 仅 ~11.5%**。
   32768 行 × topk6 的 routed FLOPs = 2.47 TFLOP/层,BF16 dense roofline
   ~15 ms/层,实测 131 ms(W4A8 124 ms)。Marlin 是为 decode 权重流
   (memory-bound 小 M)优化的;prefill 大 M 是 compute-bound,W4A8 也只
   救回 5%。**prefill MoE 的正确杠杆是换执行形态**(dequant→BF16 dense /
   cutlass grouped GEMM),潜在 ~5×,远大于本次两个杠杆之和。
2. **attention 1.148 s(40%)——eager torch 正确性路径**:FP32 gather 稀疏核
   (行分块后仍是纯 DRAM 流量)、torch 逐元素模拟的 fp4/fp8 QAT、fp32
   compressor 链。kernel 级(D0b/A4F 类)同工作量估计 ~0.1–0.2 s。
3. **HC 0.313 s(11%)**:fp32 sinkhorn ×22 次 × 8192 token;C2g/A5F 边界
   融合是现成回收路径,本次未接(其量化基于 decode 形状,prefill 形状需
   另 tune)。
4. 汇总:kernel 级 roofline 合计 ~0.4–0.5 s/stage pass → P ≈ 65–80k,
   §5.3 的 30–40k 仍然可达,但需要 prefill kernel 竖条(MoE 执行形态 +
   稀疏 attention kernel + HC 融合),而非接线级杠杆。

O(s²) 项在 8K 内不主导(基线对 chunk 平坦;ref indexer 30 ms 只占 ratio-4
层 191 ms 的 16%),这也解释了 fused indexer 端到端只有 1.04×。

## 单池 T 投影(1/T = 1/D + 8/P,D = 8733 实测)

| P(16 卡投影) | T(8K/1K) |
|---:|---:|
| 10,688(基线 @8192) | **1,159** |
| 11,417(全开 @8192) | **1,227** |
| 30,000(§5.3 带下) | 2,623 |
| 40,000 | 3,180 |

- 带下沿 **T=3.2k 在 D=8733 下需要 P ≈ 40.4k**(当前的 ~3.5×);T=4.2k 需
  P ≈ 65k,或 D 同步升到 ~20k。§5.3 的 3.2–4.2k 原本假设 D=16–20k 且
  P=35–40k,**两个前提当前都不成立**:D 差 ~2×(MTP/短 ctx 运营是 D 侧
  路径),P 差 ~3.5×(prefill kernel 竖条是 P 侧路径)。
- 按当前实测 (D, P) = (8733, 11.4k),8K/1K 单池上限 **~1.2k output tok/s**。

## 意外发现(原样记录)

1. **Marlin 大 M MFU ~11.5%** 是 prefill 最大单项;且 W4A8 增益随 M 递减
   (1.11×@4096 行 → 1.05×@16384+),与"compute-bound 下 FP8 双倍吞吐"的
   朴素预期相反——Marlin 的 FP8 输入路径同样没进入高 MFU 区。
2. prefill 吞吐对 chunk 基本平坦(11.7k@1024 → 10.7k@8192 基线):8K 内
   per-token 固定成本主导,O(s²) 项还没到场。
3. fused indexer 的 masked top-512 与 ref **逐行集合全等**(全部 26 组 gate
   记录),语义风险比预期低一个量级;E2E 硬门逐 prompt 与基线全同。
4. TP4MoE 逐层各持一套 per-shape 缓冲在 prefill 行数下**加载即 OOM**
   (cache13 1.6 GiB/层);新增 `buffer_donor` 跨层共享(`moe_runtime.py`,
   默认不启用,decode 路径不变)是任何 prefill 引擎形态的必要 runtime 特性。
5. W4A8 重排驻留字节与 W4A16 完全相同(861,931,008)——切换容量中性。
6. ratio-128 层 prefill 的 FP32 gather 工作区(3.2 GB @8192)会在 selected
   上 `.float()` 两次(score 与 output 两个 einsum 各一次)——行分块绕开了
   OOM,但 kernel 化时应一次物化。
7. **W4A8 在 decode 形状(M=4)会被 Marlin kernel 拒绝**:vLLM 对 1 字节
   激活路径强制 `block_size_m ≥ 16`(marlin_moe.py:329),runtime 冻结的
   block-size 镜像缺这条规则 → 首次 W4A8 E2E 臂在各 stage 不同时刻抛
   "Invalid thread config, MKN=[4,4096,1024]" → 集合序错开,16 卡 NCCL
   100% util 空转假死(~95 W 判据)。已修
   (`moe_runtime._marlin_block_size_m(input_dtype=...)`),
   `c2f_w4a8_repro.py` 覆盖 M∈{4,48,88} × hash/learned 全 PASS。
   对 prefill 行数(块早已 64)无影响,W4A16 路径不变。

## Artifacts

- `results/c2f-chunk*-{w4a16,w4a8}-{ref,fused}.json`:全部 16 个 bench JSON
  (per-rank timings、位级单元检查、paired 门记录、W4A8 门、峰值显存)。
- `runtime/c2f-chunk*-titan064.log`:逐 run 原始日志(含前后 nvidia-smi,
  收尾均回到 1 MiB/卡)。
- E2E 回归:`runtime/out-e0e2e-c2f-{fusedidx,w4a8}/`(rank*.json /
  result.json);本目录 `results/e2e-*-rank0.json` 存 rank0 摘要副本。
- 运行入口:`runtime/run_c2f_prefill_titan.sh`、
  `runtime/run_e0e2e_c2f_arm.sh`;汇总:`summarize.py`;
  W4A8 小 M 诊断:`runtime/c2f_w4a8_repro.py`。
- runtime 改动(均默认关、decode 语义路径不变):
  `dsv4_direct/ops/indexer_fused.py`(新)、`dsv4_direct/ratio4_fullpos.py`
  (>128 prefill + index_score_mode + sparse_row_block)、
  `dsv4_direct/attention.py`(env 门控 prefill 行分块)、
  `dsv4_direct/ops/marlin_moe.py` + `dsv4_direct/moe_runtime.py` +
  `dsv4_direct/physical_stage.py`(W4A8 加载/运行开关、buffer_donor)、
  `e0ef2e_golden_gate.py`(回归臂旗标)。
