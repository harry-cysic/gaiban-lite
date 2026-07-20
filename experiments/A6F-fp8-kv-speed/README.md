# A6F-fp8-kv-speed:FP8 KV 容量杠杆的 decode 速度可行性(Flash 几何)

日期 2026-07-20。titan065 GPU0(RTX 4090, sm89),torch 2.11.0+cu130,单卡
microbench。脚本 `a6f_fp8_kv_bench.py`,原始数据 `results/`。

**结论:判"活"。** torch 路径下 attention 侧 FP8 KV(e4m3 存储 + 读时
`.to(bf16)`)只慢 **1.2–8.4%**(主力 ratio-4 层 bl=64/8K 仅 +3.1%),不是 Pro
tilelang kernel 的 1.4–4.3×;显存减半逐字节确认。换算:t_stage 罚 +0.4–1.9%,
换 KV 行数 ×1.5–2 → 8K decode 6.4k → 投影 ~8–9.6k tok/s,收益 >> 损失。

## 口径

- 被测对象:runtime 三条 decode sparse core 的**逐算子镜像**(gather →
  dequant → fp32 einsum QK → sink softmax → fp32 einsum PV),即
  `dsv4_direct/attention.py::_torch_sparse_decode_padded_prevalidated`
  (ratio-128/ratio-4 带 mask)与
  `window_attention.py::_window_sparse_decode_prevalidated`(window 无 mask)。
  FP8 KV 只改变这一段(cache 读侧),投影/compressor/indexer 不动。
- Flash decode 几何:h=64(DP,每 rank 全 head)、d=512、seqlen=1;
  window K=128;ratio-128 K=128+ctx/128;ratio-4 K=128+512(index_topk=512,
  cache N=128+ctx/4,topk 随机指向压缩区)。
- 三变体:
  - `bf16`:latent `[bl,N,512]` bf16(现行 runtime);
  - `fp8_cast`:latent float8_e4m3fn,读时 `.to(bfloat16)`(常量 scale 口径);
  - `fp8_scale`:fp8 + per-token fp32 scale `[bl,N,1]`,gather 后 cast×scale
    (per-token scale 上界口径)。
- 计时:CUDA graph(50 次/capture,3 轮 replay,CUDA event,p50/次);
  轮间漂移 <2.5%。质量不在本竖条(后续 oracle gate)。

## Bench 数据(p50 µs/层/步;ratio = vs bf16)

| family | ctx | bl | bf16 | fp8_cast | fp8_scale |
|---|---:|---:|---:|---:|---:|
| window | — | 32 | 99.4 | 106.1 (1.068×) | 105.0 (1.057×) |
| window | — | 64 | 132.5 | 143.7 (1.084×) | 139.7 (1.054×) |
| window | — | 128 | 259.4 | 257.0 (0.991×) | 230.9 (0.890×) |
| ratio128 | 2048 | 32 | 112.5 | 119.1 (1.058×) | 120.7 (1.073×) |
| ratio128 | 2048 | 64 | 160.6 | 167.2 (1.041×) | 168.3 (1.048×) |
| ratio128 | 2048 | 128 | 351.5 | 357.9 (1.018×) | 359.3 (1.022×) |
| ratio128 | 8192 | 32 | 125.2 | 133.4 (1.065×) | 135.1 (1.079×) |
| ratio128 | 8192 | 64 | 206.9 | 209.3 (1.012×) | 216.6 (1.047×) |
| ratio128 | 8192 | 128 | 434.9 | 447.9 (1.030×) | 480.5 (1.105×) |
| ratio4 | 2048 | 32 | 336.4 | 357.3 (1.062×) | 361.0 (1.073×) |
| ratio4 | 2048 | 64 | 736.0 | 748.8 (1.017×) | 959.4 (1.304×) |
| ratio4 | 2048 | 128 | 1794.0 | 1911.6 (1.066×) | 2015.0 (1.123×) |
| ratio4 | 8192 | 32 | 340.3 | 345.9 (1.016×) | 369.7 (1.086×) |
| ratio4 | 8192 | 64 | 728.4 | 751.1 (1.031×) | 960.0 (1.318×) |
| ratio4 | 8192 | 128 | 1796.3 | 1913.0 (1.065×) | 2014.1 (1.121×) |

window 与 ctx 无关(只测 2048 几何)。`fp8_cast` 全表 0.99–1.08×;
`fp8_scale` 的 `[bl,1,K,1]` 广播乘在部分形状踩到差 kernel(ratio-4 bl=64
1.32×)——per-token scale 若最终需要,应在写入侧折掉或预物化 bf16 scale,
速度可行的形态是 `fp8_cast` 式读路径。

## 显存确认(分配口径)

latent cache 实测 `torch.cuda` 分配字节(ratio-4, bl=64, ctx=8192):

| 变体 | bytes | vs bf16 |
|---|---:|---:|
| bf16 | 142,606,336 | 1.000 |
| fp8_cast | 71,303,168 | **0.500** |
| fp8_scale | 71,860,224 | 0.504 |

逐配置的解析字节数(表中 `cache_bytes`)与分配一致:**FP8 使 latent KV
字节精确减半**。

注意"×2 行数"的完整账(8K,每卡每 stage 11 层 ≈ 5.5 ratio-4 + 5 ratio-128 +
0.5 window,每序列):latent 合计 ~13.3 MB(69%),ratio-4 indexer_kv bf16
~2.9 MB,ratio-128 compressor fp32 状态(kv_state+score_state
`[128,512]`×2)~2.6 MB,杂项 ~0.5 MB。只把 latent 转 FP8 → 行数 ×~1.53;
再加 indexer_kv fp8 + compressor 状态 bf16 化(可行性 §4.4 括注)→ ×~1.94。

## 判定:活(速度侧放行)

罚金侧(fp8_cast,8K,per-stage 11 层加权 5.5/5/0.5):

| bl | Δattention/stage | t_stage(E1F replay) | 罚金 |
|---:|---:|---:|---:|
| 32 | ~75 µs | ~20.3 ms | +0.37% |
| 64 | ~142 µs | ~26.2 ms | +0.54% |
| 128 | ~706 µs | ~36.6 ms | +1.9% |

收益侧(E1IF 实测锚点:8K bf16 KV 上限 bl_mb=40 → 6392 tok/s,容量限):

- 仅 latent FP8(×1.53 行):bl 40→61,t_stage ~26.0 ms(含罚金)→
  **~8.1k tok/s(+27%)**;
- 全套减肥(latent+indexer fp8、compressor 状态 bf16,×~1.94 行):bl 40→80,
  t_stage ~28.9 ms → **~9.6k tok/s(+50%)**,与可行性投影 10–11k 同向
  (投影用的乐观外推,本估计用 E1F 次线性曲线)。

收益(+27–50% 吞吐)>> 损失(t_stage +0.4–1.9%),**FP8 KV 在 Flash torch
路径上是活的容量杠杆**。判据不依赖 fp8_scale 变体(其 1.3× 形状可规避)。

### 归因:为什么与 Pro 结论(1.4–4.3× 慢)不同

Pro D0a 的两条死路都是 tilelang kernel 特有的:(1) in-kernel **标量**
fp8→bf16 dequant 循环(4.3×,软解码吞掉字节收益);(2) fp8×fp8 MMA 的
per-tile p-requant + 4 gemm 重构(1.4–1.9×,occupancy-bound 下省字节无收益)。
当前 runtime 是 torch masked-einsum 路径:KV dtype 只影响 gather 源字节
(减半)+ 追加一次向量化 elementwise cast(`[bl,1,K,512]` fp8→bf16),
主导成本(fp32 einsum×2 + softmax 链)完全不变,所以罚金只有个位数百分比。

## 进 runtime 的路径

1. `static_kv.py` / `static_ratio4_kv.py` / `static_window_kv.py`:`latent`
   dtype 改 `float8_e4m3fn`;写侧(`_write_decode_*`、prefill_write、
   compress finalizer)落缓存前 `.to(torch.float8_e4m3fn)`。
2. 三个 sparse core:gather 后加一行
   `selected = selected.to(torch.bfloat16)`(本 bench 的 fp8_cast 形态),
   mask/einsum 链不动。
3. 质量门(后续竖条,oracle gate):checkpoint QAT 是 nope 维 64-group
   幂次 scale e4m3(`fp8_quant_dequant`),latent 经 RMS-norm 后 O(1) 幅值,
   常量 scale 的直接 e4m3 存储在 e4m3 动态范围内,但与 QAT 口径不逐位一致,
   且 rope 64 维也被存成 fp8(Pro 设计稿保留 bf16)——golden-token 走
   D0-reference-oracle 判。若质量门要求 per-group scale,须先解决
   fp8_scale 变体的广播乘形状问题(写侧折 scale 或融合 dequant)。
4. 容量兑现的配套(同优先级):indexer_kv fp8、ratio-128 compressor
   fp32 状态 bf16 化;然后重扫 E1IF OOM 前沿(bl_mb 上限)。

## Artifacts

- `a6f_fp8_kv_bench.py` — microbench(单卡,CUDA graph 口径)
- `results/results-titan065.json` — 全量数据(逐轮)
- `results/run-titan065.log` — 运行摘要
