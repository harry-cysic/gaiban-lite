# A5F-hc-boundary-fusion:C2g tilelang HC 边界融合(Flash decode 形状)

日期 2026-07-20。单卡 titan065。动机:C1F 显示 HC 是 decode 最大非 MoE 单项
(ffn 侧 ~708 µs/层 @B=512),eager 级 C2f 融合仅 −3%;本实验在 Flash decode
形状(s=1)上量化 gaiban C2g 路径(vllm `mhc_fused_post_pre_tilelang`,
hc_post + hc_pre 单 kernel + 独立 RMSNorm)的收益与数值行为。

脚本 `a5f_hc_boundary.py`:真 Flash Block 的 HC 参数,边界 op =
attn 侧 hc_post → ffn 侧 hc_pre → ffn_norm,CUDA graph 口径 ref vs fused。
沿用 C2 的结论:`with_norm` 分支在 sm89 ≥128 token 不等价,norm 留独立 kernel。

## 结果

| B | ref (µs) | fused (µs) | speedup |
|---:|---:|---:|---:|
| 128 | 130.8 | 78.7 | 1.66× |
| 256 | 277.3 | 130.8 | 2.12× |
| 512 | 701.6 | 240.3 | **2.92×** |

数值:post/comb 与 reference 基本精确(≤9e-6);residual/h_norm max 差
1.56e-2 = bf16 在 [2,4) 幅值的**恰好 1 ulp**(与 B 无关),即 bf16 舍入级
差异,无系统性偏差。按冻结质量门方法论,最终判定在模型级对拍(D5 canary /
D0 golden tokens)做。

## 结论

1. **C2g 边界融合在 decode 是实杠杆**:B=512 每边界省 461 µs。每层两个
   HC 边界(attn 侧、ffn 侧),满接入估计 ~0.9 ms/层 → 11 层 stage 省
   ~10 ms,把 C1F 的 40.8 ms/stage 基线推向 ~31 ms、decode 12.5k →
   **~15–16k tok/s**(回到预估带内)。这是 Phase 2 dsv4_direct 移植中
   必须结构化接入的路径(跨 attn/ffn 边界融合,不能仅 monkeypatch)。
2. 收益随 B 增长(128:1.66× → 512:2.92×),与 fp32 [B,16384] 传输被
   kernel 内消化一致。

Artifacts:本目录脚本;原始输出见 git log(数值小,直接录入本 README)。
