# C1F-integrated-block:Flash 集成整层 decode bench(TP4 单机)

日期 2026-07-20。Phase 1 收官实验:真 Marlin MoE + 真 model.py attention/HC +
NCCL,4×4090(titan064 GPU0-3),CUDA graph 口径。脚本
`c1f_tp_block_bench.py`(gaiban C1' 移植,几何读 reference config.json)。
两组 A/B:`--attn-mode dp/head-shard`,`--moe-mode itp/ep`。

## 主结果(dp+itp,ctx 8K,graph µs/整层;完整含 4K/ep/head-shard 见 results/)

| B | L2(ratio-4) | L3(ratio-128) | L0(滑窗) | 43 层加权平均* |
|---:|---:|---:|---:|---:|
| 128 | 2211 | 1922 | 1878 | ~2064 |
| 256 | 2748 | 2413 | 2343 | ~2573 |
| 512 | 4043 | 3404 | 3312 | ~3712 |

*21×ratio-4 + 20×ratio-128 + 2×滑窗。

B=512 分解(DP-BD,µs):attn_local(bl=128)= 1239(L2)/ 632(L3)/ 554(L0);
allgather [512,4×4096] = 546;moe_total = 2226–2233,其中 routed(marlin,
inter512)≈ 1100(与 A3F 独立实测 1241 同量级)、ffn 侧 HC ≈ 708、
AR [512,4096] ≈ 295(与 B1 实测一致)、shared ≈ 82。
eager 5.2 ms vs graph 4.0 ms(B=512)——全层 graph 刚需再确认。

## A/B 结论(causal,同条件)

1. **DP-attention 胜 head-shard**:B=512 L2 整层 4043 vs 5284 µs(−23%);
   head-shard 的 attention 半层随全 B 扫描(3068 µs)且带 wo_b AR(736 µs)。
   B=128 时两者打平(2211 vs 2286)。**decode DP-attention 形态成立**。
2. **intermediate-TP 胜 EP**:B=512 moe_total 2226 vs 2697 µs,routed 1100 vs
   1940 µs。EP 受 expert_map 稀疏调用与负载不均拖累;A3F 的"整 expert 带宽
   更高"没有兑现成 EP 端到端优势。**itp 摆放通过 A/B 检验,维持既定形态**。
3. ctx 4K ≈ 8K(B=512 L2:3942 vs 4043)——此 B 范围 decode 对 ctx 不敏感,
   KV 读占比小。

## 对容量模型的修正(§5.2)

组件实测 roofline(B=512,8K,不含 PP handoff):
11 层 × 3712 µs ≈ **40.8 ms/stage** → decode ≈ 512/0.0408/…= **~12.5k output
tok/s(16 卡聚合)**,即 ~785 tok/s/GPU。低于可行性预估带 15–25k 的下沿约 20%。

差异归因(vs §5.2 分解表):
- MoE 权重流:routed 12.1 ms/stage ≈ 模型 10.8 ✓;
- DP 通信:allgather 6.0 + AR 3.2 = 9.2 ms/stage,模型只记了 4.4(漏 allgather);
- **HC:ffn 侧 708 µs/层 → 7.8 ms/stage(attn 侧另计入 attn_local)——模型
  把 attention+indexer+HC 合并 6–7 ms,严重低估了 B=512 的 fp32 sinkhorn HC**;
- attention:attn_local 加权 ~933 µs/层 → 10.3 ms/stage,模型 ~6–7。

**尚未接入的已验证复用资产(Phase 2 的回收路径)**:
- fused MHC(gaiban c2f_fused_hc.py,hc_mult=4 相同、bitwise 等价)——
  HC 是最大单项差额;
- fused Triton indexer(D0b,几何相同仅 topk 1024→512);
- attention 投影 W8A16(E1b2q,Pro 实测 1.74×,本 bench 用的是 reference
  act-quant FP8 路径)。
结论:**12.5k 是 reference-op 未优化基线,不构成对 15–25k 区间的证伪**;
预估带能否回收取决于上述三项在 Flash 上的兑现,列为 Phase 2 首批工作。

## 资产接入 A/B(2026-07-20 追加)

1. **fused MHC(C2f eager 级,`--hc-mode fused`)**:数值等价确认
   (hc_pre 精确相等,hc_post 差 1 ulp bf16)。B=512 整层 −100 µs 左右
   (L2 4043→3937,L3 3404→3309,L0 3312→3211,约 −2.6~3.0%)。
   **结论:eager 级融合在 decode 收益很小**——C2f 的大头在 prefill 大 chunk
   的巨型临时张量;decode 下 HC 的 ~700 µs 是 sinkhorn kernel + fp32
   [512,16384] 传输本身。要真正压缩需 C2g 的 tilelang 边界融合
   (vllm `mhc_fused_post_pre_tilelang`,hc_post+hc_pre+norm 单 kernel),
   属 dsv4_direct 运行时结构改动,归入 Phase 2 移植。
2. **fused Triton indexer(D0b)重新定位**:该 kernel 的收益是 prefill 的
   O(s²) score 物化(`_FUSE_MIN_SEQLEN=1024` 才启用);decode(s=1)不在
   收益面。从 decode 回收清单移除,归入 prefill(C2 级)工作。

## 容量注记

DP bl=128 @8K 每卡 KV ≈ 6.9 GB + 权重 ~9.4 GiB → B=512 已近 24 GB 上限,
B~640 是 8K 的实际上限(与模型的 B∈[384,768] 假设相容)。

Artifacts:`results/log_{dp_itp_8k,hs_itp_8k,dp_ep_8k,dp_itp_4k}.txt`。
现场:titan064:~/c1f/。
