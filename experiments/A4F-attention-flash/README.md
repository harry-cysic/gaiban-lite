# A4F-attention-flash:Flash 每层非 MoE 时间(attention + indexer + compressor + HC)

日期 2026-07-20。Phase 1 kernel regear。单卡 titan064(GPU0),真 Flash
`model.py` Block(8 专家小 MoE 不调用),几何直接读 `reference/config.json`。
脚本:`a4f_attn_timing.py`(gaiban A4 `a4_attn_timing.py` 换 Flash 几何:
dim 4096 / 64 heads / q_lora 1024 / o_groups 8 / topk 512 / window 128;
新增 L0 纯滑窗层型;默认 `--world 4` → n_local_heads=16)。

## 结果(8K ctx,graphed µs/层;完整 4K/8K/16K/64K 见 results/a4f_timing.txt)

world=4(head-shard TP4,h=16/卡),fixed_graph = attention 半层 + 两次 HC:

| B | L0(滑窗, ratio 0) | L2(ratio 4, indexer) | L3(ratio 128) |
|---:|---:|---:|---:|
| 1 | 215 | 376 | 235 |
| 64 | 350 | 604 | 412 |
| 128 | 452 | 833 | 533 |
| 256 | 814 | 1483 | 926 |
| 512 | 1813 | 3070 | 2004 |

eager 口径 ~2.2–3.9 ms/层(B 几乎不敏感)→ graph 后 4–10×,再次确认
decode 全 stage CUDA graph 是刚需(与 Pro A4 结论一致)。

## 关键发现

1. **TP4 h=16 原生可跑**:sparse_attn(topk 512)在 h=16 单 launch 无 smem 问题,
   graph capture 全配置成功。Pro 需要的 topk-block 64→32 修复在 Flash TP4 下不需要。
2. **h=64 单 launch 撞 sm89 smem 墙**:`--world 1` 实测
   `Failed to set the allowed dynamic shared memory size to 141312`(上限 101376)。
   即 **DP-attention(每卡全 64 heads)必须 head-loop 子 launch(4×h16)**,
   与 gaiban C1' 对 Pro(4×32)的做法同构。reference kernel 单 launch 全 heads,
   不能直接用于 DP 口径计时。
3. **DP 每卡成本待 C1F 定数**:DP(64h×128seq/卡)与已测 head-shard
   (16h×512seq)head-seq 积相同、KV 总字节相近(head-loop 4 次重读 128 序列
   ≈ 单次读 512 序列),故 L2 型 ~2.4–3.1 ms/层是 DP 口径的**上界代理**;
   按 B=128 分量折算(indexer/compressor/HC 只处理 128 序列)估计
   ratio-4 层 ~1.3–1.4 ms、ratio-128 层 ~1.0–1.1 ms → 11 层 stage
   **~13 ms 量级**,显著高于可行性 §5.2 的 "attention+indexer+HC ~6–7 ms"。
   该行需在 C1F(head-loop DP 集成 bench)实测后修订——若坐实,
   t_stage ~30–35 ms、decode 收敛到 **~15–17k tok/s**(区间下沿)。
4. indexer+compressor 增量(L2−L3):B=128 时 ~300 µs、B=512 时 ~1.07 ms,
   与 ctx 弱相关(8K);topk 减半+h 减半使 Flash 的 indexer 占比低于 Pro。

## 后续

- **C1F**:移植 gaiban `C1-integrated-block/c1_tp_block_bench.py`
  (自带 `--attn-mode dp` + `make_headloop_sparse_attn`)换 Flash 几何,
  实测 DP-attention 每卡整层(Marlin MoE + attention + HC + NCCL)decode 数字,
  修订 §5.2 分解表。
- fused indexer(D0b 产物,64×128 几何相同、topk 1024→512)接入。

Artifacts:`results/a4f_timing.txt`(world=4 全量)。world=1 失败现场见本 README
(预期内,不重跑)。
