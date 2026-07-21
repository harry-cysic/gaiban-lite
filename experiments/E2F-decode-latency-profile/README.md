# E2F — B=1 decode latency profile (M4 第一步)

第二十八竖条。**结论先行：M4 的 200–350 tok/s 计划值在当前形态下不可达，
且原因不是"延迟栈没优化"。**TARGET §6.1 的 335 tok/s 单用户带宽天花板依赖
"激活权重按 TP4 分片"这一前提；实测表明 **attention 权重在每个 TP rank 上
是完整副本**（DP-attention），占每 stage 每 rank 每 token 3.077 GB 中的
2.718 GB（88%）。按实测字节数重算，当前形态的带宽天花板是 **76.2 tok/s**，
不是 335。实测 27.5 tok/s 是该天花板的 36%，不是 335 的 8%。

第二个独立上限同样是实测的：一次 replay 里有 **2,584 个 elementwise 小核、
共 3.06 ms（39.5%）**，把 batch 从 1 提到 8 时核数几乎不变（1,980→1,983），
时间只涨 16.5%——它是**每步固定成本**，四级流水下单独就占 12.2 ms/token，
自成一个 ~82 tok/s 的上限。

**因此 M4 需要两处形态改动同时做到，缺一不可**：把 latency 模式的 attention
改成 TP4 分片（字节数 ÷4）、并把 elementwise 尾巴折叠掉。任一单独完成都仍被
另一个卡在 ~80 tok/s。

---

## 1. 动机

TARGET §2 把 M4 profile 排在第一优先级，并明确"**第一步是 profile 而不是
优化**"。进入本竖条时，B=1 只有两个数：端到端 36.3 ms/token 与每 stage
8.3 ms，**11.5× 的缺口没有任何分项归因**。TARGET §7.1 列了四个候选：
M=1 Marlin 异常、残留 eager 算子链、sinkhorn 微 kernel、86 次 allreduce。

本竖条的任务只有一个：把 8.3 ms 拆开。**四个候选里三个被实测判为次要，
一个被证实；而头号发现（带宽天花板本身算错了）不在候选清单里**——与 C4F
的经验一致（HANDOFF §2）。

## 2. 方法

### 2.1 平台：单机 4 卡隔离 stage

一个 PP stage 是 TP4-local 的：唯一的集合通信是 MoE 的 all_gather/
reduce_scatter（TP 组内），PP 交接在图外。所以一台机器的 4 张卡可以复现
E1F 的一个 stage，加载时间 1/4，且不依赖 IB。

**平台有效性不是假设而是前置门**：round A 先测未插桩 replay，必须落在 E1F
冻结的 per-family p50 上，否则下面的相位表没有意义。见 §3.1——它没有完全
落上，差 7.4%，该差值被单列为未了结项而不是抹平。

### 2.2 图内相位计时（本仓库首次）

`dsv4_direct.phase_timer.GraphPhaseRecorder`。既有的 `stage_marker` 管线
已贯通 superstage → block → attention/MoE，但此前只在 eager 路径用过
（C4F）。decode 是 graph replay，标记必须活在图里。

实测（titan065, CUDA 13.2）：`torch.cuda.Event` 默认标志在 capture 期间
能被录成 event-record 节点，但 `elapsed_time` 返回
`cudaErrorInvalidValue`；**必须 `external=True`**，external event-record
节点的计时才可查询。这一条是测出来的，不是查文档得到的。

代价是每个标记是图里一个真实节点：244 标记 = **+14.8%**，约 4.6 µs/标记。
所以相位表只用于定性分相，**定量归因以 nsys kernel 级为准**（§3.3）。

### 2.3 kernel 级：nsys

`nsys profile -t cuda --cuda-graph-trace=node`——**没有 `=node` 时一次
replay 在报告里是一个不透明的 range**，什么也看不到。探针用
`--cuda-profiler-range` 发 `cudaProfilerStart/Stop`，把 load/warmup/capture
排除在报告外。只 trace LOCAL_RANK 0，其余三 rank 裸跑。

注意：被 trace 的 rank 会通过 MoE 集合通信把自己的减速传给另外三个 rank，
四个 rank 都读到 8.23 ms（未 trace 时 7.62）。**所以 nsys 运行的绝对
ms 带 ~8% trace 膨胀，可信的是份额**；下文凡是 per-token 预算，都用份额
乘未插桩 wall。

## 3. 结果

### 3.1 平台复现与 16-rank 固定偏移

先按规矩用冻结脚本复跑 E1F bl=1（`run_e1f_dual.sh 1 bitwise 3 300 2048`，
结果落在本目录 `results/repro-e1f-bl1/`，未覆盖冻结产物）：

| 项 | 冻结值 | 复跑 | 判定 |
|---|---:|---:|---|
| 吞吐 p50 | 27.5–27.6 tok/s | 27.85 tok/s | 复现 |
| stage 0 replay p50 | 8.316 ms | 8.22 ms | 复现 |
| stage 3 replay p50（10 层） | — | 7.53 ms | — |
| head（eager） | 2.619 ms | 2.594 ms | 复现 |

隔离平台（round A，未插桩，3 轮 ×160 步）：

| family | titan065 | titan064 | 轮间离散 |
|---|---:|---:|---:|
| normal | 7.617 ms | 7.652 ms | 0.07–0.09% |
| ratio4_boundary | 8.248 | 8.284 | 0.04–0.07% |
| ratio4_ratio128_boundary | 8.719 | 8.756 | 0.46–0.88% |

两台机器相差 0.46%，**机器不是原因**。但隔离平台 7.65 vs E1F 8.22，
差 **0.57 ms（7.4%）**。该差值是**每 replay 固定的**，不随层数缩放——
E1F stage 3 只有 10 层、7.53 ms，减去同样的 0.57 得 6.96 ms，
÷10 = 0.696 ms/层；隔离平台 11 层 7.652 ÷ 11 = 0.696 ms/层，两者相等。

**未了结项 E2F-1：16-rank 上下文给每次 stage replay 加了固定 +0.57 ms
（全模型 +2.3 ms/token，占 36.3 的 6.3%），成因未归因。**候选：NCCL
communicator 数量、或 8 进程自旋下 host 端 `synchronize` 返回延迟。
⚠️ 二者都只是候选，**在补测前不得当作已知**（本仓库已被同形状的假说坑过
两次，见 TARGET §7.5.1）。它不影响下文任何份额。

### 3.2 replay 是不是 launch-bound？不是

| 项 | 值 |
|---|---:|
| kernel 时间合计 / replay | 7.739 ms |
| 被 trace 的 replay wall | 8.230 ms |
| **kernel 忙碌份额** | **≥94%** |
| kernel 数 / replay | **2,998** |
| 平均 kernel 时长 | 2.58 µs |

图里没有可观的空隙。**"残留 launch 开销"这一类假设到此判死**：CUDA graph
已经把 launch 拿掉了，剩下的全是 kernel 在跑。问题不是核之间的间隙，
是核本身太多、太小。

### 3.3 8 ms 花在哪（nsys，B=1，stage 0 = 11 层）

| 类别 | ms/replay | 份额 | kernel/replay | 平均 µs |
|---|---:|---:|---:|---:|
| dense 投影：cublas GEMV | 2.294 | 29.6% | 83 | 27.6 |
| **elementwise：其他** | **2.040** | **26.4%** | **1,980** | **1.03** |
| dense 投影：cutlass GEMM | 1.090 | 14.1% | 44 | 24.8 |
| elementwise：copy | 0.640 | 8.3% | 371 | 1.73 |
| MoE：Marlin fp4 | 0.394 | 5.1% | 22 | 17.9 |
| elementwise：reduce | 0.379 | 4.9% | 234 | 1.62 |
| collective：NCCL | 0.257 | 3.3% | 25 | 10.3 |
| sort / topk | 0.196 | 2.5% | 48 | 4.07 |
| dense 投影：fp32 SGEMM | 0.160 | 2.1% | 39 | 4.10 |
| fused tilelang（HC / 稀疏 attn） | 0.150 | 1.9% | 42 | 3.56 |
| 其他 + index/gather | 0.140 | 1.8% | 111 | 1.26 |
| **合计** | **7.739** | 100% | **2,998** | 2.58 |

归并：**dense 投影 3.544 ms（45.8%，166 核）**、**elementwise 尾巴
3.059 ms（39.5%，2,584 核）**、其余 1.137 ms（14.7%）。

**TARGET §7.1 四个候选的判决**：

| 候选 | 实测 | 判定 |
|---|---:|---|
| M=1 Marlin 异常 | 0.394 ms，5.1%，17.9 µs/核 | **次要**，且无 M=1 异常 |
| 86 次 allreduce | 0.257 ms，3.3%，25 次/replay | **次要** |
| sinkhorn 微 kernel | 无单独大项；混在 elementwise 尾巴里 | 并入下条 |
| 残留 eager 算子链 | **3.059 ms，39.5%，2,584 核** | **证实，且是头号可攻项** |

elementwise 尾巴没有单一主犯：71 种核型，最大一项（`direct_copy`，
232 次/replay）只有 0.45 ms。按名字看是 `add / div / mul / sum / mean /
max / clamp / sqrt / rsqrt / where / pow` ——未融合的 eager 逐元素链。

### 3.4 elementwise 尾巴是固定成本（B=1 vs B=8）

同一探针、同一 stage，`--local-batch 8`（全局 32 行）：

| 类别 | B=1 | B=8 | 倍数 |
|---|---:|---:|---:|
| elementwise 合计 | 3.059 ms | 3.565 ms | **1.17×** |
| 其中"其他"核数/replay | 1,980 | 1,983 | **1.00×** |
| dense 投影合计 | 3.544 | 4.605 | 1.30× |
| Marlin | 0.394 | 1.269 | 3.22× |
| NCCL | 0.257 | 0.656 | 2.55× |
| kernel 总数/replay | 2,998 | 3,134 | 1.05× |
| replay 合计 | 7.739 | 10.793 | 1.39× |

**8× 的 batch 只让 elementwise 涨 17%、核数不变**——它按步收费，不按行
收费。这正是 TARGET §7.1 说的"该固定成本同时限制吞吐"的机制，现在有了
数字。dense 投影只涨 30% 也同时证实它是**权重带宽**主导，不是行数主导。

### 3.4b elementwise 尾巴在哪：77% 在 attention 体内

相位标记有 2–15% 的自身开销，不适合定量。trace 不需要标记：一次 replay 内
kernel 序列是固定的，MoE 的集合通信天然是分隔符——`ReduceScatter` 每层恰好
一次（层分隔），层段内第一个 `AllGather` 是 MoE 入口（分 attention / MoE）。
两个计数都先对层数校验再报数（`analyze_regions.py`）。

241 个稳态层段：

| 区域 | µs/层 | kernel/层 | 其中 elementwise |
|---|---:|---:|---:|
| attention + HC | 547.3 | 211 | **213.8 µs / 180.6 核** |
| MoE | 166.6 | 74 | **62.5 µs / 52.5 核** |

**elementwise 尾巴 276.2 µs/层，77% 在 attention 体内。**

而**尾巴是按核数收费、不是按字节收费**：`elementwise: other` 平均
**1.03 µs/核**，`copy` 1.73，`reduce` 1.62——都在 4090 的最小 kernel 时长
附近。这解释了 §3.4 的 B=1 vs B=8：8× 的行数不改变核数，所以时间几乎不变。
**推论：任何减少核数的融合，几乎 1:1 换成时间。**

用一个已知对象校准这条推论。两条独立读数：

- **层段步长**：ratio-4 层 291 核、ratio-128 层 223 核，差 **68 核**。
  ⚠️ 这 68 个是 **ratio-4 相对 ratio-128 的全部增量**（indexer GEMV、topk/排序、
  index score einsum 都在内），**不等于 QAT 链本身**——它只是 QAT 链核数的上界。
- **细相位表**：`index_query_done` 91.0 µs/层，减去 `index_wq_b` 的
  16.777 MB ÷ 806 GB/s = 20.8 µs，**余 ~66 µs 非 GEMV**。

两者量级一致，且 E4F 的孤立微基准随后给出第三个读数：同一条链在 decode 形状下
**68.61 µs**。三条独立路径吻合，所以"QAT 链 ~66–69 µs/层"是可用的；
但**"68 核"不要当成 QAT 链的核数**。

**这直接回答了 TARGET §2 对 M4 profile 提出的点名问题**："indexer QAT 链
在 B=1 是否显著，先验估算只值 0.2–0.4%"。**实测 5 个 ratio-4 层 × 66 µs
= 330 µs = replay 的 4.3%（step 的 3.6%）——比先验估算高一个数量级。**

**后续（E4F）已把它兑现**：C4F 的融合核在 decode 形状下逐位相等，层内成对交替
A/B 实测 **−0.2925 ms/stage（−3.82%）**、全模型闭环 **+2.31%**（27.740 →
28.381 tok/s）。注意收益机制与 prefill 完全不同——不是省带宽，是省 kernel 启动，
所以 C4F 的 90.5× 不可折算。

### 3.5 字节账：天花板算错在哪

用 `build_physical_stage` 实际驻留的张量逐个量（不是模型级估算），
每 TP rank 每 token：

| 层型 | attention 权重 | 备注 |
|---|---:|---|
| window（L0/L1） | 213.916 MB | |
| ratio-128 | 230.957 MB | |
| ratio-4 | 273.183 MB | 多 indexer 链 |

**这些张量是完整尺寸的**：`wq_b (32768,1024)` 是全部 64 头，`wo_a
(8192,4096)`、`wo_b (4096,8192)` 同理——**每个 TP rank 一份完整副本**
（DP-attention）。三个 67.1 MB 的张量就占了 ratio-4 层的 74%。
compressor 两项是 **FP32**（各 16.8 MB）。

MoE 相反，**是分片的**：`w13_q (256,256,2048) int32` = 每 rank 持每个
expert 的 1/4 中间维。每 token 取 6 个 routed expert + 1 shared：
6×3.343 + 12.582 = **32.64 MB/层**。

shared expert 那 12.582 MB 是 **BF16 反量化副本**——`moe_runtime.py` 的
`shared_path = "bf16_dequant_correctness_fallback"`，FP8 常驻副本
（6.291 MB）同时存在但热路径不读它。名字表明这是当初的正确性回退而非
预期的生产路径；改用 FP8 常驻可省 6.291 MB/层 = 270 MB/token（占总量
2.2%），需质量门，列为未了结项。

stage 0（2 window + 5 ratio-4 + 4 ratio-128）合计：

| 项 | 每 rank 每 token |
|---|---:|
| attention（复制） | 2.718 GB |
| MoE（分片） | 0.359 GB |
| **合计** | **3.077 GB** |

可达带宽由本次 trace 就地标定：最大的一次 GEMV 读 67.109 MB 用 72.513 µs
= **925.5 GB/s**，与项目沿用的 928 GB/s 常数吻合。

- stage 屋顶 = 3.077 GB ÷ 928 GB/s = **3.32 ms**；实测 7.65 ms = 屋顶的
  43%（2.31×）。
- dense 投影核实际吞吐 = 2.856 GB ÷ 3.544 ms = **806 GB/s = 可达带宽的
  87%**。**投影核本身几乎没有余量，能省的只有字节数。**

全模型（四级串行，各 stage 层型分布不同）：

| stage | 层 | attention | MoE | 合计 |
|---|---|---:|---:|---:|
| 0 | L0–L10 | 2.718 GB | 0.359 | 3.077 |
| 1 | L11–L21 | 2.752 | 0.359 | 3.111 |
| 2 | L22–L32 | 2.794 | 0.359 | 3.153 |
| 3 | L33–L42 | 2.521 | 0.326 | 2.847 |
| **合计/token** | 43 层 | **10.784 GB** | **1.404** | **12.187** |

**每 token 带宽下界 = 12.187 GB ÷ 928 GB/s = 13.13 ms → 76.2 tok/s。**

TARGET §6.1 写的是 `5.5 GB ÷ (4×928 GB/s) ≈ 1.48 ms/级` → 335 tok/s。
**那个 ÷4 要求权重按 TP4 分片；实测 88% 的字节（attention）没有分片。**
MoE 那 1.404 GB 确实分片了，attention 那 10.784 GB 没有。

### 3.6 36.3 ms/token 的完整收支

⚠️ 本节的 per-stage / per-layer 分项是**诊断量**（TARGET §9.13 允许）；
本竖条的 headline 口径是**一套 16 卡系统的单路 decode**，即 27.5→29.1 tok/s。

用 §3.3 的份额乘未插桩 wall（7.652 ms，titan064）：

| 项 | ms/token | 来源 |
|---|---:|---|
| dense 投影（4 级） | 14.02 | 45.8% × 7.652 × 4 |
| elementwise 尾巴（4 级） | 12.09 | 39.5% × 7.652 × 4 |
| 其余图内（Marlin/NCCL/topk/tilelang） | 4.50 | 14.7% × 7.652 × 4 |
| head（eager，stage 3） | 2.59 | 实测 |
| 16-rank 固定偏移（§3.1，未归因） | 2.28 | 4 × 0.57 |
| embed + 交接 + token 回环 | ~0.8 | 实测 |
| **合计** | **36.3** | 实测 36.3 ✓ |

收支闭合。**36.3 ms 里没有"神秘的延迟栈"**：72% 是投影字节 + 固定
elementwise 尾巴两项。

## 4. 推论（推导，非实测——标注清楚）

以实测字节数与实测固定成本为输入的两个上限：

| 形态 | 权重字节/token | 带宽下界 | elementwise 下界 | 可达上限 |
|---|---:|---:|---:|---:|
| **当前（DP-attention，BF16 常驻）** | 12.19 GB | 13.13 ms | 12.1 ms | **~76 tok/s** |
| attention TP4 分片 | 4.10 GB | 4.42 ms | 12.1 ms（不变） | ~82 tok/s（被尾巴卡住） |
| 仅折叠 elementwise 尾巴 | 12.19 GB | 13.13 ms | →0 | ~76 tok/s（被带宽卡住） |
| **两者都做** | 4.10 GB | 4.42 ms | →0 | **~226 tok/s** |
| 再加 attention 权重 FP8 常驻 | ~2.75 GB | 2.97 ms | →0 | ~337 tok/s |

**这是本竖条对计划的唯一硬结论：M4 的 200–350（+MTP 300–600）不是
"优化到位就能到"，是要求两处形态改动同时到位。**任一单独完成都停在
~80 tok/s。

第四行落在计划带内，说明**计划值本身没错，错的是达成路径的假设**。

第五行（FP8 常驻 attention）需要单独的质量门：实测 `wq_b` 等在 runtime
里是 **BF16**、compressor 是 **FP32**，而 checkpoint 里非专家权重是 FP8
（`quantization_config` e4m3）——即 runtime 在 dense 侧付了 BF16 税。
TARGET §3 的"无 BF16 税"说的是 on-disk，**runtime 侧有**。是否能改需要
按冻结质量门方法论验证，本竖条不作判断。

## 5. 未了结

1. **E2F-1**：16-rank 上下文的固定 +0.57 ms/stage 未归因（§3.1）。
2. elementwise 尾巴已定位到区域（§3.4b：77% 在 attention），但**未逐条链
   拆解**。已确认的第一条是 indexer QAT 链（68 核/层、4.3% of replay）。
   其余 ~113 核/层（attention 内）与 ~52 核/层（MoE 内）归属未定。
3. attention TP4 分片的**可行性与代价**未评估（每层多一次 allreduce、
   KV 摆放变化、需要新的图族与质量门）。
4. head 的 2.59 ms（7.2% of step）未归因——vocab 129,280 的 eager GEMM，
   未测其是否分片。
5. shared expert 走 BF16 反量化副本（§3.5），FP8 常驻副本闲置；
   省 270 MB/token，需质量门。

## 5b. 尾巴的剩余部分：按"非 GEMV 时间"排序的待攻清单

把细相位表的每个相位减去 **该相位投影权重字节 ÷ 806 GB/s**（§3.5 实测的 dense
投影实际吞吐），余下的就是不由权重带宽解释的时间——也就是可融合的部分。
每项再减去 4.62 µs 的标记开销（由本次 fine 运行的 overhead ÷ spans 反推）。

**ratio-4 层（µs/层）**：

| 相位 | 实测 | 屋顶 | 非 GEMV |
|---|---:|---:|---:|
| `index_query_done` | 86.4 | 20.8 | **65.6** ← **E4F 已取（实收 58.5，89%）** |
| `index_topk_done` | 44.5 | 0 | **44.5** |
| `sparse_done` | 32.2 | 0 | **32.2** |
| `raw_kv_done` | 35.3 | 5.2 | **30.1** |
| `compressor_projection_done` | 59.9 | 41.6 | 18.3 |
| `ffn_prepare_done`（HC） | 17.9 | — | 17.9 |
| `state_write_done` | 17.9 | 0 | 17.9 |
| `query_done` | 111.1 | 93.7 | 17.4 |
| `block_done`（HC） | 16.9 | — | 16.9 |
| `output_transform_done` | 5.6 | 0 | 5.6 |
| `wo_a_done` | 80.4 | 83.3 | **−2.9** |
| `output_done` | 76.3 | 83.3 | **−7.0** |

**最后两行是这套减法的自检**：纯 GEMV 相位应当落在屋顶上，实测给出 −2.9 与
−7.0——略微"超过"屋顶，因为 806 GB/s 是全体投影核的平均而这两个是最大的核
（就地标定的峰值 925 GB/s）。自检通过，说明减法口径是校准的。

ratio-128 层同法为 **121.8 µs/层**（该层 compressor 更小，上表字节数是 ratio-4
专用的，故 L3 的 compressor 行无效，已剔除）。

**合计：一个 stage（5×r4 + 4×r128 + 2×window）约 1.65 ms 的非 GEMV 时间**，
E4F 只取走了其中 0.29 ms。**尾巴的大头仍在**，下一个目标很清楚：
`index_topk_done`（44.5）与 `raw_kv_done`（30.1）。

⚠️ 这是**排序用的估算，不是实测收益**：屋顶用全体平均吞吐、标记开销按均值扣除，
且 E4F 的实收/估算比是 89%——**引用前须按 §9.4 用层内 A/B 实测**
（工具已有：`e2f_decode_phase_probe.py --mode ab`，含对照臂）。

## 6. Artifact

| 路径 | 内容 |
|---|---|
| `../../runtime/e2f_decode_phase_probe.py` | 探针（round A 未插桩 / round B 图内相位） |
| `../../runtime/run_e2f_probe.sh` | 单机 4 卡驱动（含前后 nvidia-smi 快照） |
| `../../runtime/run_e2f_nsys.sh` | nsys 驱动（`--cuda-graph-trace=node`，只 trace rank 0） |
| `../../runtime/dsv4_direct/phase_timer.py` | `GraphPhaseRecorder`（external event 图内计时） |
| `analyze_nsys.py` | kernel summary → 类别预算表 |
| `analyze_regions.py` | kernel trace → 每层 attention/MoE 区域拆分 |
| `results/out-e2f-stage0/` | titan065 相位表（fine，244 标记，overhead 14.8%，coverage 99.76%） |
| `results/out-e2f-stage0-coarse/` | 粗相位表（47 标记，**overhead 2.01%**，coverage 99.82%）——attention 65.8% / block_done 26.9% |
| `results/out-e2f-stage0-t064/` | titan064 未插桩复现 |
| `results/out-e2f-nsys-stage0/` | B=1 kernel 级 trace 与类别预算 |
| `results/out-e2f-nsys-stage0-bl8/` | B=8 kernel 级 trace（固定成本对照臂） |
| `results/repro-e1f-bl1/` | E1F bl=1 冻结脚本复跑 |

`.nsys-rep` / `.sqlite` 留在 titan065 `~/e0f-runtime/out-e2f-nsys-*/`，
不进 Git；CSV 与 JSON 已取回。
