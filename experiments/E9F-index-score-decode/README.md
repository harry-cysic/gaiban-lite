# E9F — index-score 链的 decode 尾巴折叠（判死：不能复用 prefill 融合核）

**状态**：micro 判死。**结论：既有 prefill 融合核在 decode 形状上更慢，
折 `index_topk_done` 尾巴需要一个 decode 专调核，不是 E4F 那样接现成核。**

## 动机

E2F §5b 把 `index_topk_done`（**44.5 µs/层、roofline 0 = 纯 launch-bound elementwise**）
列为 E4F/E5F 之后**最大的待折尾巴**。其核心是 decode 里的 eager 链：

```
scores = einsum("bshd,btd->bsht", q, kv)                      # [b,1,h,t]
scores = scores.relu_().mul_(w[...,None]).sum(dim=2).float()  # [b,1,t]
```

而 `ops/indexer_fused.fused_index_score` **已经把这条链融成一个核**
（`sum_h relu(q_h @ kv^T) * w_h`）——**但只有 prefill 路径（ratio4_fullpos）接了它**，
decode（ratio4_attention）仍走 eager。**假设**：像 E4F 那样把这个现成融合核接到
decode，能折掉 44.5 µs 的尾巴。

## 判死（micro-benchmark，单卡 titan065）

`runtime/e9f_index_score_micro.py`——decode 形状（b=1, s=1, h=64, d=128）、
成对交替计时（§9.1）、扫 candidate_width。artifact：`results/micro-result.json`。

| candidate_width | eager | fused | speedup | topk 一致 | score max Δ |
|---:|---:|---:|---:|---:|---:|
| 512 | 99.5 µs | 118.6 µs | **0.84×**（慢 19.0） | True | 6.9e-4 |
| 1024 | 98.9 µs | 108.6 µs | **0.91×**（慢 9.8） | **False** | 5.8e-4 |
| 2080 | 98.8 µs | 108.7 µs | **0.91×**（慢 9.9） | **False** | 9.7e-4 |

**两条判死理由**：
1. **更慢**：融合核在 decode s=1 上比 eager **慢 10–19 µs/次**——它是**为大 s prefill 调的**
   （融 h 轴省 O(s²) DRAM），s=1 时那点 DRAM 省不出来，反被核自身开销盖过。
2. **非逐位**：fp32 求和序变化让 **topk 选择在 t≥1024 时翻**（topk_equal=False）——
   即便它快，也是 §9.6 软门（会改 decode 输出）。

**⟹ E4F 模式（接现成融合核）对 index-score 链不成立。** micro 先判死，
省下了把一个**回归**接进 frozen decode 路径（会拖慢刚建好的 serving 路径）。

## 尾巴折叠的下一步（redirect）

- **index_topk_done 要 decode 专调核**：只融 elementwise 尾（relu+mul+sum+mask，
  E2F 的 44.5 µs 是**减掉 einsum 之后**的非 GEMV 部分），保持逐位（保 h 求和序）——
  这是**新核 authoring**（比接现成核大），量级同 C4F 的融合核。
- **或先挑 `raw_kv_done`（30.1 µs，rms_norm/rope）**：可能比 index-score 链更易融，
  且 rms_norm/rope 的 elementwise 融合更标准。E2F §5b 排序：index_topk_done(44.5) >
  sparse_done(32.2) > raw_kv_done(30.1)。
- ⚠️ 沿用 E4F/E5F 纪律：micro 判活/判死 → 层内 A/B 实测收益 → D0L 门（逐位则硬门，
  改求和序则软门不越包络）→ 闭环。**引用收益前必层内 A/B。**

## 与目标的关系

尾巴折叠是**单路性能目标（≥150 框架 ≈ 187.5 裸）的必要条件之一**（E2F：分片 + 尾巴折叠
缺一不可，分片已放行 E6F）。它同时下拉 E8F 刚测的 serving decode（24.8 ms/tok）。
本判死不改变该结论，只说明 index-score 链要走 decode 专调核这条更贵的路。
