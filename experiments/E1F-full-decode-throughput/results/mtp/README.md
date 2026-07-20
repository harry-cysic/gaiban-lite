# MTP(multi-token prediction)投机解码接入 — 第十六竖条结果

日期 2026-07-21。双机 16 卡(titan064 s0/s1 + titan065 s2/s3),43 层 + embed/head +
mtp.0(尾 stage,titan065 s1,与 head/embed 同处)。torch 2.11.0+cu130,
权重根 `titan064:~/Workspace/DeepSeek-V4-Flash`(各 gate 的 checkpoint_id 见其
result/summary JSON;id 按被 inspect 的分片集合计算,故三个 gate 数值不同、根相同)。
代码:`runtime/dsv4_direct/{mtp_block,verify2}.py` + `block_weights/window_attention/
head_stage` 扩展;gate 脚本 `runtime/e0mf_mtp_block_oracle.py`、
`runtime/e0mtp2e_golden_gate.py`、`runtime/e1mtpf_verify_bench.py`。

## MTP 语义要点(reference model.py 逐行核实)

- **输入流**(MTPBlock.forward :757-763):`x = 主模型 pre-head HC 残差 [b,s,hc,d]`
  (位置 p 的 hidden)+ **已定下一 token** 的 embedding:
  `e = enorm(embed(ids)); x = hnorm(x); x = e_proj(e).unsqueeze(2) + h_proj(x)`,
  e_proj 广播到 4 条 HC 流;MTP 输出预测再下一个 token(p+2 位置的输入)。
- **block 核**(:764 → Block.forward):layer_id 43 → compress_ratios[43]=0,
  纯滑窗 attention(无 compressor/indexer,**no-YaRN base rope_theta=10000**,
  model.py:477-479)+ learned noaux_tc 路由 MoE(43 ≥ num_hash_layers)。
- **head 归属**(:765 + :750-752 + :793):logits 用**共享主 head.weight**,
  但经 MTP **自有** 的 sigmoid hc_head(hc_head_fn [4,16384]/base/scale)与
  **自有** norm;last-position fp32(ParallelHead.get_logits :716)。
  embedding 同为共享表(:792)。
- **协议**(reference generate.py 不用 MTP,按标准 1-draft speculative decoding
  自定):KV 已提交至位置 P、待送 token x_{P+1}、draft z≈x_{P+2}(来自 MTP 吃
  (h_P, x_{P+1}));verify 步一次送 [x_{P+1}, z] 两个位置 → y_{P+1}, y_{P+2};
  贪心下 **z == y_{P+1} 即接受**(emit 2 token,提交 P+2),否则 emit y_{P+1}、
  回滚第二位置状态、提交 P+1;MTP 只吃已提交 (hidden, next-token) 对,自身无需回滚。
  回滚要点:窗口 ring/ratio-128 compressor 为纯赋值可由 refeed 自愈,但 ratio-4
  overlap compressor 的边界移位(model.py:353-354)是破坏性的 → 采用
  post-first-token 状态快照/恢复(`verify2.snapshot/restore_decode_state`)。

## (a) MTP block 前向 oracle 对拍(E0mf,titan064 TP4,mtp.0 真实权重)

E0df 框架改造:candidate = `MTPLane`(bf16 控制路径,marlin MoE),按 stage
teacher-forced 对独立 fp32 oracle(bridge fp32 重算、E0wf 窗口 attention lane、
fp32 HC、mtp.0.ffn fp32 routed(MXFP4)+shared(FP8) oracle、fp32 hc_head/norm/共享
head)。prefill 96 + decode 96..98;**bf16-KV 与 fp8-KV 两臂均 PASS**:

| stage(最差 rms_rel,bf16 臂) | 值 | 限 |
|---|---:|---:|
| bridge(enorm/hnorm/bridged) | 0.0031 | 0.012 |
| attn_branch / state.raw | 0.0160 / 0.0102 | 0.04 / 0.02 |
| moe_local(fp32 oracle) | 0.0080 | 0.03 |
| route_weights(ids 逐位相等) | 0 | 2e-5 |
| logits | 0.0024 | 0.012 |

全部 exact checks(twin 逐位、route ids、HC post/comb、state 位置)通过;
所有 phase logits argmax 与 fp32 oracle 一致。fp8-KV 臂 attn_branch 0.0314、
state.raw 0.0130,仍在原限内。Artifacts:`e0mf-bf16/`、`e0mf-fp8/`。

## (b) E2E golden 分歧率 + 输出一致性(E0mtp2e,16 卡,B=1,fp8 KV + fused HC)

五臂,8 条 D0 golden prompts(482 token):

- **off_teacher(基线重现)= 467/482** —— 与 E0e2e fused+fp8 臂逐 prompt 完全
  相同(2/28/128/124/12/22/31/120)。
- **mtp_teacher = 467/482,预测流与 off_teacher 逐位相同**(MTP 只旁挂,
  主路径不动)→ 分歧率精确等于基线,MTP on 不劣化。teacher 口径接受率
  **406/474 = 0.857**(draft 与模型自身预测比;draft 直接命中 golden 404/474)。
- **off_free vs mtp_free(chained verify,协议精确形)**:8/8 prompt 输出 token
  序列**完全一致**(硬验收 PASS)。部署口径接受率 **221/257 轮 = 0.860**。
- **mtp_fused(融合 seqlen-2 verify)**:输出与 off_free **不完全一致**
  (5/8 prompt 分歧,共 110 个位置;接受率 0.835 略低)。归因:verify 步中
  hidden 侧 GEMM 形状 [2,d] vs [1,d] 的 bf16 数值差在 near-tie 处翻转 argmax
  (与 fused-vs-eager HC、bf16-vs-mp8 golden 分歧同类;free-run 下逐步放大)。
  → **B=1 语义路径采用 chained 形;fused 形是大 B 的性能形态,质量上属
  语义变更臂,需按冻结质量门单独验收后才可默认开启。**

逐 prompt 接受率(chained / fused):1.0/1.0, 0.81/0.61, 1.0/1.0, 0.90/0.87,
0.63/0.63, 0.91/0.91, 0.94/0.94, 0.72/0.71。Artifacts:`e0mtp2e-fp8-fused/`
(result.json + rank0/rank12 全记录含逐轮数据)。

## (c) 实测实效吞吐(eager 全位置口径)与 graph 口径投影

**注意口径**:E2E/bench 走的是 eager 全位置控制路径(E0e2e 形),B=1 闭环
串行时延 ~208-213 ms/步(= 4 stage 串行 + head + token 回环广播;E1F 的
36.3 ms/步是 stateful graph 路径,两者不可混比)。

E0mtp2e 实测(真实 prompt,短 ctx):off_free 213.1 ms/token;
mtp_free(chained)加权 **159.1 ms/有效 token(1.34x)**;mtp_fused 170.2(1.25x)。

E1MTPF 实测(ctx 2048 种子态,B=1,settle 16 + 3x100 轮,p50 漂移 <0.5%):

| phase | p50 (ms) |
|---|---:|
| baseline 单 token 步 | 208.0 |
| chained verify 轮(accept / reject) | 298.8 / 288.0 |
| fused verify 轮(accept / reject) | 316.8 / 307.1 |

实效 ms/有效 token(实测轮时 + 实测接受率):@α=0.86:chained **159.8(1.30x)**、
fused 169.6(1.23x);@α=0.66:chained 177.8(1.17x)、fused 188.8(1.10x)。

**关键结构发现:chained 双 pass 一轮 ≈ 单步 + 1 个 stage 时间 + MTP 开销**
(298.8 ≈ 208 + ~52 + ~38),因为 draft 在轮首已知,第二 pass 尾随第一 pass
流水交叠——B=1 串行 decode 的 PP4 管线本来 3/4 空泡,投机第二 token 恰好填泡。
fused 形在该 eager 口径反而更慢(逐位置核翻倍 + 每层 in-pass 快照 + host 同步),
其 GEMM 融合收益要到权重读摊薄(大 B)或 graph 化后才兑现。

**graph 口径投影(未实测,须 MTP block graph 化后验证)**:chained 形不需要
seqlen-2 graph——round ≈ t_step(36.3) + t_tail_stage_replay(~8) + t_mtp(~2-4)
≈ 47-49 ms → @α=0.86 ≈ **25-27 ms/有效 token(~1.4x)**;@α=0.66 ≈ 28-30 ms,
与预估 ~24ms@0.66 方向一致、略保守(预估未计回环/尾 stage 附加项)。按 eager
实测比值外推(298.8/208=1.44)则为 36.3x1.44/1.86 ≈ 28 ms/token。

## 大 B graph 化设计说明

见 `DESIGN-largeB-graph.md`(定长 2-token verify graph + per-sequence position
向量 + 掩码化 boundary + 影子提交回滚;拒绝不同步的三种替代方案证否)。

## 意外发现

1. **fused seqlen-2 verify 不保输出逐位一致**(上文 (b)):协议本身无损
   (chained 臂 8/8 一致证明),分歧完全来自 [2,d] GEMM 形状数值——把
   "2-token 融合"当作独立语义变更臂对待。
2. **chained 投机天然填 PP 空泡**:B=1 下 draft-verify 的第二 pass 与第一 pass
   管线交叠,一轮增量只有 ~1 stage 时间;这意味着 B=1 图化路径不需要
   seqlen-2 capture 就能拿到大部分 MTP 收益(简化了落地顺序:先 MTP block
   graph 化 + chained 轮,再做大 B 的定长 verify graph)。
3. E0e2e 基线的 "decode 56 ms/step" 是 teacher-forced 管线化吞吐口径
   (输入已知,步间交叠);真实闭环 B=1 eager 时延是 ~208-213 ms/步。两口径
   此前未区分,本竖条 off_free 臂首次实测闭环口径。
4. mtp.0 checkpoint 契约与 A0 核实一致(33 tensors,146,068,332 B/rank 复制部,
   完整 256-expert FFN 由 marlin key_prefix="mtp.0.ffn" 装载,861,931,008 B/rank)。

## Artifacts

- `e0mf-bf16/`、`e0mf-fp8/`:MTP block oracle gate(summary + rank0)。
- `e0mtp2e-fp8-fused/`:E2E golden 五臂(result + rank0/rank12 含逐轮记录)。
- `e1mtpf-fp8-fused/`:ctx-2048 轮时 bench(result + rank0/rank12)。
- `logs/`:各 run 的 node 日志(含前后 nvidia-smi;收尾两机 16 卡均 1 MiB)。
- 驱动脚本:`runtime/run_e0mf_titan.sh`、`runtime/run_e0mtp2e_dual.sh`、
  `runtime/run_e1mtpf_dual.sh`。
