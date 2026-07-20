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

## 交织流水与吞吐-容量前沿(2026-07-20 追加,E1IF)

4-microbatch 轮转交织(`runtime/e1if_interleaved_bench.py`,无串扰 gate:交织轨迹与
各 lane 单跑逐位相同;瓶颈 stage busy 93.7%,气泡 ~6%)。实测前沿(DP 口径,
graph+fused HC,聚合 tok/s,best round):

| ctx | bl_mb=32 | 40 | 48 | 64 | 96/128 | 实测天花板 |
|---|---:|---:|---:|---:|---|---|
| 2048 | 5885 | — | 7368 | **8570** | OOM | 8.6k(B_total=1024,显存限) |
| 8192 | 5693 | **6392** | OOM | — | — | **6.4k(B_total=640,显存限)** |

replay 次线性:bl 32→48→64 为 20.3→23.7→26.2 ms(权重流固定成本主导);
2K 下外推 bl128 ≈14k 但 bf16 KV+工作区在 bl96 即 OOM——**约束是容量,不是算力**。

## 容量模型修正(goal 条款触发:假设被实测证伪)

原 §5.2 "decode ≈ 512/t_stage ≈ 19–21k(带 15–25k)@8K" 依赖两个假设,均不成立:

1. **t_stage 线性缩放**:实测强次线性(bl=32 时 20.3 ms,是 bl=128 的 36.6 ms
   的 55% 而非 25%)——流水交织把大 batch 拆小 mb 时吞吐显著低于线性换算。
2. **管线填充的 KV 同驻**:满流水需 4 mb 的 KV 同驻(×4);"B_global=512 全局"
   吞吐点实际需要 2048 条在飞,8K bf16 KV 下不可行(实测 8K 上限 B_total=640)。

**修正后的 decode-only 实测结论:8K/bf16-KV ≈ 6.4k output tok/s**(全链路
graph 化、fused HC、DP、94% busy——不是工程未完成,是容量边界)。证据链:
E1F(t_stage-bl 曲线)→ E1IF(交织无串扰+busy 占比)→ 前沿扫描(OOM 判界),
全部 3 轮重复、绑定 checkpoint/env/拓扑。

**回收杠杆(可行性自列,待验证)**:FP8 KV(×2 行数,须按 §5.4-5 在 Flash
h=16 几何重验速度/质量)→ 8K 投影 bl≈80/mb ≈ 10–11k;MTP(~1.5×)→
**~15–16k,带下沿在两杠杆齐备时可及**;短 ctx 运营(4K/2K)另有余量
(2K 实测 8.6k,FP8 KV 后投影 ~14k)。head 入 graph(2.9 ms,13% of s3)
与计算/传输 overlap 为小杠杆。
