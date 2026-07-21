# 大 B chained MTP 投机解码 × 交织流水(graph 化)— 第十八竖条结果

日期 2026-07-21。双机 16 卡(titan064 s0/s1 + titan065 s2/s3),43 层 + embed/head +
mtp.0(尾 stage,与 head/embed 同 GPU 组)。torch 2.11.0+cu130,checkpoint
33fd3df002d0b44c…(rank JSON 内全量)。代码:`runtime/dsv4_direct/specdec.py`
(per-row position 机制,新增文件)+ `runtime/e1mtp_largeb_bench.py` +
`runtime/specdec_unit_gate.py` + `runtime/specdec_stage_probe.py` +
`runtime/moe_composition_probe.py`;驱动 `runtime/run_e1mtp_largeb_dual.sh`,
汇总 `runtime/aggregate_mtp_largeb.py`。**改动纯增量:未修改任何既有模块**
(B=1 golden 路径与既有 E1IF 路径结构上不受影响)。

## 形态取舍:chained 双 pass(选定)vs 定长 2-token verify graph

按设计文档优先评估 chained 形,选定 chained,理由:

1. **无损性**:E0mtp2e 已证 fused seqlen-2 verify 因 [2,d] GEMM 形状数值
   在 near-tie 处翻转 argmax、不保逐位一致(5/8 prompt 分歧);chained 每
   pass 与标准 1-token 步同算子链,可逐位对齐基线 —— 本竖条的硬验收
   (同组成下 ON≡OFF 逐位)只有 chained 形能满足。
2. **实现风险**:两种形态都必须落地 per-row position 向量 + 掩码 boundary
   (批内 accept 不同步是共通的);chained 免去 seqlen-2 capture 与逐位置
   核翻倍,pass 图 = 既有 1-token 算子链的 row-pos 版。
3. **管线经济学**:mb2 lanes × 2 pass = 4 slot 恰好填满 PP4(与 E1IF mb4
   同稳态);batch 加倍摊销(bl 减半 lane 数减半 → 每 slot 行数翻倍,
   replay 次线性)承担了 fused 形"GEMM 融合摊销"的角色。

实现要点(`dsv4_direct/specdec.py`):positions[B] 向量驱动全部位置派生量
(RoPE 逐行 gather、ring/compressor slot 逐行 scatter、ratio-128 padded 索引
逐行构建、ratio-4 indexer 可见性逐行掩码);三 family 合并为单一均匀图
(compressor pooling 每步恒算、边界按 `phase==ratio-1` 掩码提交,ratio-4
破坏性 shift 掩码执行);拒绝回滚 = pass-A 图尾影子快照 + 下轮图头按
accept 掩码恢复(仅 ratio-4 overlap 状态;ring/ratio-128/compressed 行由
refeed 自愈);MTP 每轮定长 2 个 1-token 图(pass-1 吃 (h_A,t1) 恒提交,
pass-2 吃 (h_B,t2) 接受掩码经 ring 自愈),draft/pending 全程 device 端
torch.where 选择,loopback 携带 (pending, draft, accept)。

## (a) 正确性 gate(mb2, bl8, ctx2048, fp8+idxfp8, fused HC)

**单卡单元门(`specdec_unit_gate.py`,合成权重,bf16/fp8 双臂全 PASS)**:
T1 均匀位置 132 步逐位 vs family 路径(含两类边界);T2 污染不变性
(chained 轮×136,拒绝 pass-B 内容任意变化,提交输出/终态逐位不变);
T3 图捕获 vs eager 逐位;T4 失步位置(行间 offset 0..3 混相位)逐位。

**16 卡硬验收(`out-e1mtplb-gate-...-force_reject/`,R=132,全 PASS)**:
同组成臂(全轮拒绝 → 批内位置对齐,ON 每步组成与 OFF 完全一致):
每行 132-token 流与 OFF eager-family 基线**逐位前缀相等**(8 行×2 lane×
4 TP,全 16 rank),graph-vs-eager twin 264/264 slot 逐位(含 MTP 图),
solo vs interleaved 逐位(无串扰),positions 全 16 rank 一致,teardown 干净。

**协议臂(`out-e1mtplb-gate-...(normal)/`,R=132)**:twin 逐位、无串扰逐位、
positions 一致同样全 PASS;但 ON-vs-OFF 逐位前缀在批内失步开始后数 token 内
分歧(实测 α≈0.92-0.99 的合成态下每行前缀 2-7+)。**归因已定位并证实
(`specdec_stage_probe.py` T5)**:失步改变批组成 → Marlin 分组 GEMM 对
批组成存在 1-2 ULP 级逐行敏感(T5:MoE 输入行逐位相等、输出行 max_abs
0.5-2e-3 ≈ bf16 1-2 ULP;`moe_composition_probe.py`:行内容组成不敏感、
行序/组内偏移敏感 1.5e-5)。种子合成态 logits 近平坦 → 翻转率被放大;
该敏感度与已放行的 fused-HC 1-ulp/bf16 near-tie 包络同类,**对任何大 B
失步投机形态(chained 或 fused)同等存在,非本实现缺陷**(stub-MoE 后
T5 全逐位;force_reject 全逐位)。协议本身无损的证据链 = force_reject
全逐位 + T2 污染不变性 + B=1 chained 8/8(E0mtp2e)。

## (b) 吞吐(fp8 KV + idxfp8,global pool,fused HC,3 轮×300 chained 轮)

每轮 = 每 lane pass-A(verify)+ pass-B(draft)各一次 1-token 图;
实效 tok/s = B_total×(1+α_batch)×轮数/wall(accepts 由 rank12-15 JSON 复核)。
α_batch 为种子合成态实测(退化流,MTP 命中率异常高,故另列 α=0.86
(E0mtp2e 真实 prompt 部署口径)归一值)。

| 配置 | B_total(KV 行/卡) | 轮 wall (ms) | α_batch 实测 | 实效 tok/s(3 轮) | @α=0.86 | 基线对照 |
|---|---|---:|---:|---|---:|---|
| 8K mb2-bl128 | 1024 (256) | 233.8-234.6 | 0.983-0.997 | 8667 / **8734** / 8716 | 8145 | bl72-mb4 基线 8733(288 行/卡):**1.00×**(@0.86:0.93×) |
| 8K mb2-bl112 | 896 (224) | 216.0-217.5 | 0.983-0.995 | 8224 / **8276** / 8219 | 7717 | bl56-mb4 基线 7872(同 224 行/卡):**1.051×**(@0.86:0.98×) |
| 2K mb2-bl128 | 1024 (256) | 211.5-212.9 | 0.982-0.996 | 9563 / **9656** / 9600 | 9007 | bl64-mb4 基线 8515(同 256 行/卡):**1.134×**(@0.86:1.058×) |

**主结论:大 B 满流水下 chained MTP 增益远低于 B=1 投影(~1.4×)——
8K 同容量点仅 +5%(α≈0.99)/约平(α=0.86),2K +13%。** 结构原因:
B=1 的"第二 pass 填 PP 空泡"在 93%+ busy 的交织流水中不存在,每轮付
2 个整 slot;收益只剩 (1+α)/2 × 批加倍摊销(T_bl / T_bl/2)⁻¹×2。实测
bl56→112 slot 比 1.89(含 s3 MTP 附加),摊销不足以在 8K 兑现 (1+α);
2K slot 更短、权重读占比更高 → 摊销更强,净增益转正。轮内分解
(bl128 s3 slot p50):replay 38.3 + MTP 10.3 + recv 2.5 + head 0.4 ms
—— MTP 图(块+共享 head fp32 GEMM)占轮 ~9%,是首要优化点
(bf16 head/与下 slot overlap 可回收)。

**容量意外**:MTP 常驻(embed 1.06 GiB + mtp MoE 0.86 GiB + block)压低
尾 stage 容量墙:mb2-bl144(=bl72-mb4 的 288 行/卡)OOM 于 s3 lane 构建,
**8K 可达上限 = 256 行/卡(bl128)**;为容下 bl128 已做:seed payload 逐层
流式化、timed 模式快照逐 lane 瞬态化(17 竖条同款教训复犯后修正)、
timed 弃 eager MoE slot(slots=2×mb)。收尾显存检查两机 16 卡均 1 MiB。

## 意外发现

1. **Marlin MoE 分组 GEMM 的批组成 1-2 ULP 逐行敏感**(上文归因;三个
   递进 probe + stage 级 T5 证据链)。它给"大 B 投机 = 与基线逐位相等"
   设了硬上限;后续若要求严格逐位,需组成不变的 MoE 内核(每行独立
   k-序)或接受 near-tie 包络口径。
2. **chained 大 B 的经济学与 B=1 完全不同**:E0mtp2e 的"chained 轮 ≈
   步 + 1 stage"仅在管线空载时成立;满流水下应按 (1+α)/2×摊销系数
   评估 —— 8733×1.4≈12.2k 的投影对 chained 满流水形不适用,须转向
   verify 图内融合(seqlen-2)或 draft-pass 降价(小 draft 模型/部分层)
   才有兑现空间。
3. 种子合成态上 MTP 接受率 0.98-0.997(退化重复流极易预测),不能当
   部署口径;α=0.86(E0mtp2e 实测)归一列为准。
4. bl112/128 首跑 OOM 三连的根因均为宿主侧驻留(全 lane payload、全 lane
   快照、eager slot),不是模型容量;修正后 bl128@8K 反而比基线 bl72 少用
   11% KV 行数打平吞吐。

## Artifacts(均含 16×rank JSON + result.json;logs/ 含两节点日志)

- `out-e1mtplb-gate-fp8-idxfp8-mb2-bl8-ctx2048-force_reject/`:硬逐位 gate(PASS)。
- `out-e1mtplb-gate-fp8-idxfp8-mb2-bl8-ctx2048/`:协议臂 gate(R=132,含逐行
  token 流 dump;twin 264/264 与无串扰逐位 PASS,α 0.916/0.943,前缀分歧
  归因 MoE ULP,故 result.json accepted=False 为前缀硬判所致、非机制缺陷)。
- `out-e1mtplb-gate-...-split/`、`-split-eager/`:归因二分臂(强制半批拒绝,
  含 per-stage digest trace:stage-0 输出在首个失步轮即分歧、输入仍逐位)。
- `out-e1mtplb-timed-fp8-idxfp8-mb2-bl{112,128}-ctx8192/`、`-mb2-bl128-ctx2048/`:吞吐。
- `unit-gates/`:单元门 result.json(T1-T4,18+24 checks 全 PASS)与
  MoE 组成敏感性 probe(out-moe-probe{,3})。
