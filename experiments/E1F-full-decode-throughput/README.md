# E1F-full-decode-throughput:满配 PP4×TP4 stateful graph 化与 decode 吞吐

日期 2026-07-20。双机 16 卡(titan064 s0/s1 + titan065 s2/s3),43 层 + embed/head,
每 stage 11 层 stateful CUDA graph(三 family 惰性 capture,E0sf 机制)、fused
tilelang HC 边界默认开(e0ef2e 已放行)、跨机 serial handoff(E0qf,no-GDR)。
checkpoint f8ae78fd…,torch 2.11.0+cu130。脚本 `runtime/e1f_full_decode_bench.py`。

## 口径(与 C1F 不可混比)

- **B = 全复制口径**:同一 B 条序列复制到每 stage 的 4 个 TP rank,每卡 KV=B 条;
  复制口径下每 rank 计算量 ≡ DP-attention 在 B_global=4B 时的每 rank 计算量
  (换算依据)。
- ctx 2048 起步(种子残留,ratio-4 index 饱和之上);8K 敏感性点:step wall +4.6%。
- microbatch=1 **serial** closed-loop(token 真实回环:head→argmax→回环 pair→embed);
  吞吐 = B/Σstage,流水交织未实现——DP+满流水数字是由实测 stage 时间的模型换算。
- 每 B:132 步 settle + 3 轮×300 步计时;3 轮 p50 漂移 <2%。

## 正确性抽查

B=1 与 B=128 各 132 步 graph vs eager(fused 同 fused):四 stage 全部
**132/132 逐位**,KV digest 逐层相等,cursor/teardown 干净,logits 全程有限。

## B 扫描(rank0 step wall p50/p95;replay = 各 stage graph p50)

| B(=bl/卡) | step p50/p95 (ms) | 实测吞吐 (tok/s) | s0/s1/s2/s3 replay (ms) | max send (ms) |
|---:|---|---:|---|---|
| 1 | 36.3 / 39.0 | 27.5 | 8.3 / 8.4 / 8.7 / 7.6 | 0.14 |
| 8 | 56.5 / 59.7 | 142 | 13.5 / 13.0 / 13.6 / 12.0 | 0.55 |
| 32 | 80.9 / 84.1 | 396 | 19.8 / 18.2 / 19.2 / 16.8 | 0.95 |
| 64 | 110.5 / 114.0 | 579 | 26.2 / 24.7 / 25.8 / 22.9 | 1.79 |
| 128 | 155.9 / 159.8 | 821 | 36.4 / 35.0 / 38.2 / 32.6 | 3.67 |
| 192 | 208.8 / 212.8 | **920** | 47.8 / 47.5 / 49.7 / 44.0 | 6.92 |
| 256 | OOM(复制口径容量上限 192) | — | — | — |

四 stage replay 相差 <15%(均衡);handoff 发送侧 0.9–6.9 ms,IB 跨机与机内
跨 socket 同量级,no-GDR 非瓶颈。B=1 36 ms/步 = graph 化前 fused eager 55–57 ms
的 0.65×(单用户 ~27.5 tok/s,可行性预估 ~30 方向一致,MTP 前)。

## 对照容量模型

DP+满流水换算(4B / max-stage-replay,含 ~2 ms/跳 handoff 摊销):

| 口径 | 换算吞吐 | 对照 |
|---|---|---|
| bl=128 ≡ B_global=512 | 13.4k(含 handoff ~12.7k) | 修正后预估 ~14k:差 4–9%,**带内** |
| bl=192 ≡ B_global=768 | 15.5k(~14.9k) | §5.2 B∈[384,768] 带、原始带 15–25k 下沿**够到** |

8K ctx 折扣 ~5%。**结论:修正后容量模型(~14k @B512)被满配实测支持;
原始 15–25k 带的下沿在 B_global=768 运营点可及**。尚未兑现的部分
(把换算变实测):真 DP-attention 序列切分、≥4 microbatch 流水交织;
再往上:MTP(~1.5×)、handoff overlap。

## 意外发现

1. B=256 复制口径 OOM 于 ratio-4 eager warmup 的 einsum(每卡 22.06 GiB 已占);
   B=192 为 ctx-2048、4 MoE slot 下上限。OOM 后 16 rank 集合楔死需手工清理——
   后续扫描脚本应加显存预算预检。
2. 跨 TP lane 输出非逐位(reduce_scatter 求和序),种子残留下 lane argmax 会
   漂移分叉(真实 prompt 时 e0ef2e 实测 lane agreement 为 True);计时无影响。

Artifacts:`results/`(每 B 的 rank JSON + result.json、ctx-8192 点、logs、
sweep-table.json);驱动 `run_sweep.sh`、汇总 `aggregate_results.py`。
