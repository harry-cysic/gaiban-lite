# E3F — 8 卡 TP4×PP2 容量判决

第二十九竖条。**结论：8 卡 TP4×PP2 装不下 43 层模型。**这不是"紧张"——
在 B=1、max_seq 3328（可能的最省配置）下，一个 22 层 stage 在加载第 19 层时
OOM。实测每卡最多 **18 层**（19 层为极限且无余量），PP2 需要 **22 层**。

**因此：M7（方案 A，2× 独立单机）按当前形态不成立；M4 的家也不在 8 卡。**
即便做满 E2F 识别出的两个 attention 侧杠杆，余量也只有 1.13 GiB——够装下
模型，不够装 KV 池。

---

## 1. 为什么这个测试是直接的、不是外推

TARGET §7.2 把"8 卡形态验证"列为性价比最高的一次验证，因为它同时卡住
M4 家园、M7 与全部单机口径计划值。

8 卡 TP4×PP2 的**一个 stage 就是 4 张卡上的一个 TP4 super-stage**，
层数 21–22。E2F 的探针（`e2f_decode_phase_probe.py`）建的正是这个对象，
只是层数可配。所以 `--layers 0-21` **就是**目标形态的 stage 0，
不是近似、不是外推、无需新代码。

## 2. 实测

titan064 GPU0-3，B=1，max_seq_len 3328，bf16 KV，fused HC：

| 层数 | 结果 | after_load 占用 | 余量 |
|---|---|---:|---:|
| 11（= 16 卡 PP4 的 stage） | ✅ 跑通 | 13.86 GiB | 9.66 GiB |
| 18 | ✅ 跑通（含捕图与 replay） | 22.12 GiB | **1.40 GiB** |
| **22（= 8 卡 PP2 的 stage）** | ❌ **OOM 在第 19 层** | 23.12 GiB 时耗尽 | — |

OOM 原文：`Tried to allocate 512.00 MiB. GPU 0 has a total capacity of
23.52 GiB of which 403.94 MiB is free`，四个 rank 同时。

**边际成本（11→18 层实测）：1.180 GiB/层。**
22 层外推 = 22.12 + 4×1.180 = **26.84 GiB**，超卡容量 3.3 GiB——与实测
OOM 位置一致（第 19 层时已用 23.12 GiB）。

顺带的性能标度（同批运行）：

| 层数 | normal replay p50 | 每层边际 |
|---|---:|---:|
| 11 | 7.652 ms | — |
| 18 | 12.573 ms | **0.7031 ms/层** |

与 E2F 的 0.696 ms/层一致（E2F §3.1），**replay 时间对层数是线性的**。

## 3. 容量为什么不够：字节账

用 E2F §3.5 的实测驻留字节，每 TP rank 每层：

| 项 | 每层每卡 | 22 层 | 占卡容量 |
|---|---:|---:|---:|
| routed experts（FP4，已按 TP4 分片） | 0.797 GiB | **17.53 GiB** | 75% |
| shared expert + gate | 0.020 GiB | 0.43 GiB | 2% |
| attention（**BF16/FP32，每 rank 完整副本**） | 0.232 GiB（均） | **5.09 GiB** | 22% |
| **权重小计** | | **23.05 GiB** | **98%** |
| 非权重（KV + 图池 + workspace） | 0.131 GiB | 2.88 GiB | 12% |
| CUDA context 等固定项 | — | 0.91 GiB | 4% |
| **合计** | | **26.84 GiB** | **114%** |

非权重两项由 11 层（2.35 GiB）与 18 层（3.27 GiB）两个实测点解出：
每层 0.131 GiB + 固定 0.91 GiB。合计 26.84 GiB 与第 2 节的外推一致，
也与"第 19 层 OOM"一致。

**光是权重就要 23.05 GiB，而卡只有 23.52 GiB。**所以装不下与 KV 大小、
batch、图池都无关——是权重的硬容量问题，且 KV 还要另外占 2.88 GiB。

对照 16 卡 PP4（11 层/卡）：权重 11.51 GiB，占 49%，余量充足。这也修正了
TARGET §3 的"权重/卡：16 卡 ~9.4 GiB；8 卡 ~20 GiB"——**实测是 11.51 GiB
与 23.05 GiB**。差额正是 E2F 发现的 attention BF16 常驻税（checkpoint 里
非专家权重是 FP8，runtime 里是 BF16，compressor 还是 FP32）。

## 4. 什么能改变结论（推导，非实测）

以第 3 节的实测字节为输入：

| 形态改动 | 权重/卡（22 层） | + 非权重 3.79 GiB | 余量 | 判定 |
|---|---:|---:|---:|---|
| 当前 | 23.05 GiB | 26.84 | −3.32 | ❌ 超 3.3 GiB |
| attention TP4 分片 | 19.23 | 23.02 | **+0.50** | ⚠️ 极限 |
| attention 分片 + FP8 常驻 | 18.60 | 22.39 | **+1.13** | ⚠️ 勉强 |

**注意这三行的大头都是 routed experts 的 17.53 GiB——它已是 FP4 且已按
TP4 分片，没有进一步压缩余地。**所以 8 卡形态的可行性完全取决于 attention
侧的两个杠杆，而这**正是 E2F 为 M4 识别出的同两个杠杆**。

但即便两个杠杆都做满，余量也只有 **1.13 GiB**，而且这 3.79 GiB 的非权重
基数是在 **B=1、max_seq 3328** 下测的——KV 随 B 与序列长度线性增长，
吞吐模式要的 KV 池是几个 GiB 量级。**结论：两个杠杆能让 8 卡"装下模型"，
但装不下一个有意义的 KV 池。8 卡形态只可能支撑 B 极小的延迟模式，
不可能支撑 M7 所要的 10–16K 吞吐。**

## 5. 对目标的影响

1. **M7（方案 A：2× 独立单机 + 前置路由，计划 decode 10–16K）不成立**——
   它要求整模型驻留在 8 卡上；当前超 3.3 GiB，做满两个 attention 杠杆后
   余量 1.13 GiB，装不下 10–16K 所需的 KV 池。**建议移入 TARGET §5（已证伪）
   而非继续挂在模式矩阵里。**
2. **M4 的家不在 8 卡**。TARGET §7.2 的"单用户是单机游戏"在**容量上不成立**；
   延迟模式只能跑在 16 卡 PP4 上（或先做 attention 分片）。
3. **单机口径的全部计划值**（5–8K 聚合、20K prefill）与对外承诺里
   "单机聚合 ≥2,000"的书面口径，**都失去了形态基础**，需重新表述。
4. **attention TP4 分片从"M4 的一半"升级为"三条线的共同前提"**：
   M4 延迟目标、8 卡形态、M7 全都指向它。这是当前最高价值的单项形态改动。

## 6. 未了结

1. 只测了 decode 侧驻留。prefill 的 workspace 峰值更高，8 卡即便装下权重
   也未必跑得动 prefill——未测。
2. attention TP4 分片的**实现代价**（每层多一次 allreduce、KV 按 head 摆放、
   新图族、质量门）未评估。第 4 节的表只是字节账，不是可行性结论。
3. 未探索其他 8 卡形态（TP8×PP1 等）。TP8 被 TARGET §4.1 以 attention
   头数 padding 为由否决，但**该否决前提是 attention 被 TP 分片**；
   在 DP-attention 下 §4.1 的推导不直接适用。若要复活 8 卡，这条值得重看。

## 7. Artifact

复用 E2F 的探针与驱动，无新代码：

| 路径 | 内容 |
|---|---|
| `../../runtime/e2f_decode_phase_probe.py` | 探针（`--layers` 决定 stage 层数） |
| `../../runtime/run_e2f_probe.sh` | 驱动 |
| `../E2F-decode-latency-profile/results/out-e2f-pp2stage0/` | 22 层 OOM（rank JSON 里带 traceback） |
| `../E2F-decode-latency-profile/results/out-e2f-pp2max18/` | 18 层跑通 + 内存与 replay |
| `../E2F-decode-latency-profile/results/logs/e2f-pp2stage0-titan064.log` | OOM 原文与前后 nvidia-smi |
| `../E2F-decode-latency-profile/results/out-e2f-stage0-t064/` | 11 层对照 |
