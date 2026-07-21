> **归档说明（2026-07-21）**：本文是 2026-07-18 的立项技术报告，保留为历史记录与
> 推导链参考（尤其 §1.1.1 的 batch-1 字节账、§0.3 硬规则的论证）。**其性能预估已由
> `TARGET-v4-flash.md` 取代**；其中被实测证伪的条目见该文 §5.2。请勿据本文的
> 预估值做规划。

# Flash 部署拓扑与性能预估(内部技术版)

**日期**:2026-07-18 · **状态**:全部为分析外推,Flash 本体尚无实测点
**读者**:自己人。数字带置信标签:【锚】= Pro 实测锚点直接换算;【推】= roofline 推算;【估】= 粗估;【未建】= 依赖未开发组件。
**对外口径对照**:客户资料承诺 单路≥150 / 单机聚合≥2,000 / prefill≥15,000 / 128K–512K–1M 三档。本文所有计划值都应显著高于对外承诺。

---

## 0. 共同基础

### 0.1 模型账(定数)

| 项 | 值 |
|---|---|
| 权重 on-disk | 159.6 GB(FP4 专家 + FP8 其余,官方即此精度,无 BF16 税) |
| 激活参数 | 13B/token(top-6 of 256 + shared) |
| KV | **6.6 KB/token**(bf16,43 层合计:r4 5,120B + r128 160B + indexer 1,344B);fp8 KV 减半【锚,容量杠杆,质量待验】 |
| PP 边界载荷 | mHC 隐态 4×4096×bf16 = **32 KB/token**,×B 为每步载荷(B=64 → 2 MB/步) |
| 每层专家权重 | 3.2 GB/层(FP4);TP4 per-expert inter 切分后 0.8 GB/层/卡 |

### 0.2 单卡显存预算模板(24 GB)

| 配置 | 权重/卡 | 运行时+图池+workspace | KV 预算/卡 | 集群 KV 池 |
|---|---|---|---|---|
| 8 卡(TP4×PP2) | ~21.5 GB | ~1.5–2 GB | **~2.5–3 GB** | ~22 GB |
| 16 卡(TP4×PP4) | ~11.5 GB | ~2 GB | **~10–11 GB** | ~170–180 GB |
| 24 卡(TP4×PP6) | ~8.2 GB | ~2 GB | **~13 GB** | ~310 GB |

KV 归属规则(决定上下文上限):DP-attention 下每行(序列)由每个 stage 的**一个** owner 卡持有该 stage 全部层的 KV。单会话每卡 KV/token = 6.6KB × (stage 层数/43):PP2 → 3.3KB,PP4 → 1.65KB,PP6 → 1.1KB。

**推论(单会话上下文上限)**:
- 8 卡:512K = 1.65 GB ✔;**1M = 3.3 GB ✘(超预算,这就是单机 1M 放不下的根因)**
- 16 卡:1M = 1.65 GB ✔,且池内可并发 ~25 路 1M 会话【推】
- fp8 KV 全线减半,是不动拓扑的容量翻倍杠杆【未验质量】

**token-slot 预算**(静态预分配,B 与 max_seq 耦合):B × max_seq ≤ 集群 KV 池 / 6.6KB。
8 卡 ≈ 3.3M slots;16 卡 ≈ 27M;24 卡 ≈ 47M。例:8 卡 @max_seq 8K → B≤410;@64K → B≤52;@256K → B≤13。

### 0.3 硬规则(不因目标而变)

1. **TP=4,永不 TP8**:Flash 64 头 ÷8 = 8 头/卡,sparse_attn32 padding 到 16 → 50% 注意力算力白烧;TP4 = 16 头恰好。TP 只在 socket 内(P2P patch ~25GB/s),绝不跨 socket。
2. **P2P patch(aikitoria 590.48.01)是前提**:没有它 batch-1 掉到 100–150。
3. **CUDA graphs 强制**,B 是图的编译期常量——每个部署档位冻结自己的 B,改 B = 重推 fixed-B kernel 几何 + 重捕获(Pro 的教训:代价大,选点要先扫)。
4. **PP 边界按时间切,不按层数切**:43 层不整除任何 stage 数;Pro E1b2x/w 证明 fractional boundary(把某层的 attention/pre-MoE 与 MoE 尾拆开跨 stage)可行且必要。
5. **kernel 路线**:decode = Marlin W4A16 MoE + sparse_attn32 + W8A16/FP8 投影(质量门锚定官方 FP8 参考,非 Pro 的 BF16 锚);prefill = Marlin W4A8 + D4 hybrid-prefix。autotune 全部按 Flash 维度(K=4096/2048/1024)重推,Pro 的 plan 不能直接抄。

### 0.4 Pro 实测移植的关键校准

- **PP 交接固定税**:传输"在场"即收 ~2.3–2.7 ms/步(B240、4.3MB、PCIe SYS),字节数是次要项(E1b3g:砍半只省 0.44ms)。**不要假设 Flash 载荷小(2MB)就便宜**。E1b3i(GPU-timeline IPC)判决未出;NODE 路径比 SYS 快 2×,卡位规划留后手。
- 无传输时 overlap 包络 ~0.5 ms(E1b2z);带宽利用率校准系数:Pro 实测 stage 时间 ≈ 字节 roofline ÷ **0.54**(含 attention、对齐、气泡)。本文吞吐推算给出 [roofline × 0.54, roofline] 双界。

---

## 1. 单机(8×4090,TP4×PP2)

```
socket0: GPU0-3 = TP4 → stage0(L0..~L21 + 可能的 L22 前半)
socket1: GPU4-7 = TP4 → stage1(其余层)     ← 1 个跨 socket 边界/步(即 Pro 台架形状)
```

### 1.1 追求单用户吞吐(latency 模式,B=1–8)

| 指标 | 数值 | 置信 |
|---|---|---|
| batch-1 decode(短 ctx) | **200–350 tok/s** | 【锚】带宽+延迟 roofline,含 graph+P2P patch |
| +MTP 投机解码 | 300–600 tok/s | 【未建】×1.5–1.8 |
| batch-1 @512K ctx | ~100–200 tok/s | 【估】r128 全扫 + indexer 扫描主导 |
| TTFT(8K prompt) | ~0.4 s | 【锚】prefill 20K |
| 单会话 ctx 上限 | 512K(bf16)/接近 1M(fp8 KV,勿承诺) | 【推】 |

要点:单用户速度是**单机游戏**——加机器不涨 batch-1(见 §2.3),涨速只有 MTP/投机一条路。deep-think(384K+)单机 fp8 KV 勉强可用,正式口径放双机。

#### 1.1.1 200–350 的推导链(留档)

**① 字节账**:batch-1 每 token 读全部激活权重 ≈ **11 GB**:
路由专家 6×3×2048×4096×0.5B ≈ 75MB/层 + 共享专家 13MB/层(FP4 合计仅 ~3.5GB)
+ 注意力/投影 wq_a/wq_b/wkv_a/wo_a/wo_b + indexer + compressor ≈ 150–190MB/层(FP8/bf16,~7GB)
+ lm_head 530MB。**大头是 dense 侧不是专家**——FP4 把瓶颈挪到了 FP8 注意力上。

**② 结构事实**:B=1 时 PP2 两级**串行**(流水线只对多请求有效),任意时刻仅 4 卡工作:
`5.5GB ÷ (4×928GB/s) ≈ 1.48ms/级 × 2 ≈ 3.0ms → 带宽天花板 ~335 tok/s`(928 = Pro Marlin 实测有效带宽)。这就是区间上沿。

**③ 延迟栈**(M=1 GEMM 跑不满带宽,固定成本浮出):
- ~900–1000 kernel/token(43 层 × ~20 余算子),graph replay 后每 kernel 仍有 1.5–3µs 执行地板 → ~2ms;
- **86 次 TP4 allreduce**(每层 2 次,mHC 隐态 32KB),P2P patch 下 ~15µs/次 → ~1.3ms(= "TP4 税"在 batch-1 的形态);
- sinkhorn(20 迭代微 kernel × 86 处)、fp32 compressor 等 → ~0.5–1ms【估】。
叠加(与带宽部分重叠)→ **3.5–5ms/token → 200–350 tok/s**。下沿 = 延迟栈全额兑现。

**④ 敏感性自洽检验**:无 P2P patch → allreduce 50–100µs/次 → +4–8ms → 100–150 tok/s ✔(与 §1.1 无 patch 口径吻合)。

**⑤ 为什么不是 TP8**:TP8 理论带宽下界砍半(8 卡同读 → 1.5ms),但 64 头÷8=8 头/卡被 sparse_attn32 pad 到 16(注意力 50% 白烧)+ 86 次 allreduce 被迫跨 socket(每次 +20–30µs → +2ms)——罚款超过收益。**TP4×PP2 是算出来的最优,不是妥协。**

**⑥ MTP 的物理来源**:B=1 时 PP2 有 ~50% 级空转,draft/verify 恰好错级流水填泡——×1.5–1.8 不只是"一次出俩"的摊销,还白捡了闲置的那半台机器。故 MTP 在单用户路线图上优先级最高。

**待实测校准的软肋**:Flash 小维度下 M=1 kernel 地板、sinkhorn/compressor 真实串行成本、graph replay 在 ~1000 节点图上的行为——bring-up 后用实测替换本节的【估】项。

### 1.2 追求聚合吞吐(throughput 模式)

工作点由 max_seq 档位决定(token-slot 预算),B 冻进图:

| max_seq 档 | B | 步长 | 聚合 decode | 置信 |
|---|---|---|---|---|
| 8K | ~400 | ~20–37 ms | **8–15K roofline / 计划 5–8K** | 【推】专家全触发,近算力界 |
| 32K | ~100 | ~16–30 ms | 3.5–7K / 计划 3–5K | 【推】(即此前对话里的 3–5K 口径) |
| 128K | ~26 | ~14–26 ms | 1–2K | 【推】 |
| prefill(独立) | — | — | **~20K tok/s** | 【锚】 |

第一批 bring-up 实验就是把这张表变成实测:**B × max_seq 二维扫描优先于一切 kernel 特化**(与 Pro 相反:Pro 容量宽裕先冻 B,我们容量饥饿必须先扫)。

---

## 2. 双机(16×4090)

### 2.1 方案 A:2× 独立单机实例 + 前置路由(求稳/求聚合的保守解)

- 聚合 decode = 2×单机:@8K 计划 **10–16K**;prefill 40K。零 IB 依赖、零新风险,serving 闭环未成熟前的默认形态。
- 缺:ctx 仍 ≤512K,无 1M;KV 池不合并,prefix cache 命中率减半(会话需粘滞路由)。

### 2.2 方案 B:单实例 TP4×PP4(求容量/长上下文)

```
机A socket0=stage0, socket1=stage1;机B socket0=stage2, socket1=stage3
每 token 周期:2 次 PCIe 跨 socket + 2 次 IB 跨机(含回绕)
层切分:~11+11+11+10,按时间 fractional 微调
```

| 指标 | 数值 | 置信 |
|---|---|---|
| 集群 KV 池 | ~180 GB(fp8 ~360 GB) | 【推】 |
| 1M 会话并发 | ~25 路 | 【推】 |
| B 上限 @8K | ~2,000 slots 意义上不再是约束,B 由图内存/算力选点(240–500) | 【推】 |
| 聚合 decode @8K,B≈400 | roofline 15–18K / **计划 8–12K** | 【推】关键折扣:2×IB 交接税,量级未知,待 Pro E1b3 系列判决 |
| IB 交接载荷 | 32KB×B=13MB/步 @B400 → GDR ~1.4ms 线速 + 固定税 | 【锚+推】 |

**风险声明**:PP4 的交接税是本方案最大不确定项。若 Pro 证明固定税无解(E1b3i 失败且无 NODE 拯救),方案 B 的计划值向 6–8K 收缩,方案 A 变为长期主力,1M 档位改为"低吞吐专用实例"。

### 2.3 双机对单用户的真实收益(澄清)

- decode 速度**不变**(200–350):batch-1 是权重流带宽+延迟游戏,PP 加深只加交接;EP 跨 IB 每层双向 all-to-all 延迟 ~0.5–1ms/token【估】,直接毁盘。
- 真收益:① **1M 上下文可用**;② 巨型 prompt 的 TTFT——两机 PP 流水 prefill ~1.8× → 1M prompt TTFT ~28–35s(单机装不下,无从谈起)【推】;③ deep-think 正式解锁。

---

## 3. 三机(24×4090)

### 3.1 方案 A:3× 独立实例——纯聚合最大化的保守解

@8K 计划 **15–24K** decode + 60K prefill。适合短 ctx 批处理/API 转售型负载。

### 3.2 方案 B:1P+2D 分离(推荐给 agent/API 混合负载)

```
P 机(8卡, TP4×PP2):纯 prefill。W4A8+D4 路径,无 CUDA graph 约束,KV 算完即迁,
   容量饥饿弱点不暴露 → 20K tok/s 进料【锚】
D 池(16卡, TP4×PP4):纯 decode。全部 KV 预算给会话常驻(agent prefix cache 一等公民),
   纯 decode 图,TPOT 无 prefill 毛刺 → 8–12K 出料【推】
KV 迁移:27MB/4K-seq,GDR ~3ms,IB 上无感【锚】
```

- 平衡点 in:out ≈ 2:1 至 1:5,正对 reasoning/agent 流量;8K入/1K出型流量会 P 侧饥饿(那种负载用方案 A)。
- 每流 TPOT 稳定性是三方案最优;serving 闭环工程量最大(迁移调度器)。
- 附带价值:替 Pro 趟 P/D 子系统(Pro 结构性做不了)。

### 3.3 方案 C:24 卡单实例 TP4×PP6(暂缓)

KV 池 310GB、B 上限更高,roofline 20–25K【推】,但每 token 5 次交接(其中 2–4 次 IB),固定税×5 在 Pro 判决前不可控;层切分 43/6 更碎。**除非 E1b3i 大胜,否则不排期。**

---

## 4. 汇总速查

| 目标 \ 规模 | 单机 8 卡 | 双机 16 卡 | 三机 24 卡 |
|---|---|---|---|
| **单用户速度** | TP4×PP2:200–350(+MTP 300–600) | 同左;+1M ctx、TTFT 1.8× | 同左;边际收益趋零 |
| **聚合吞吐(短 ctx)** | B≈400@8K:计划 5–8K | 2×DP:10–16K | 3×DP:15–24K |
| **聚合吞吐(长 ctx/容量)** | 512K 顶格,B 骤降 | 单实例 PP4:8–12K + 25 路 1M | 1P+2D:20K 进 + 8–12K 出 |
| **agent/API 混合** | 可用但 KV 池小 | 单实例 PP4(prefix cache 池化) | **1P+2D(首选)** |
| 对外承诺对照 | 2K/15K → 余量 ≥2.5× | 6K → 余量 ≥1.3–2× | 定制 → 自由 |

---

## 5. 依赖与开放问题(按优先级)

1. **B × max_seq 扫描**(bring-up 第一批):把 §1.2 表变实测,再冻 B、再特化 fixed-B kernel。
2. **Pro E1b3 传输判决**:决定 PP4/PP6 交接税模型 → 方案 2.2/3.3 的生死;NODE 路径卡位方案同步预研。
3. **Flash model contract**(4096/43L/256E/o_groups 8/q_lora 1024/index_topk 512)+ autotune 全量重推。
4. **质量阶梯**:锚定官方 FP8 参考,L2(teacher-forced Δppl/KL,~10 分钟/次)做量化与 kernel 变更判决;勿抄 Pro 的 BF16 route-exactness 锚(其官方参考自身不过该门)。
5. **serving 闭环**【未建】:continuous batching、prefix caching/会话 KV 常驻(agent 硬需求)、P/D 迁移调度器。所有"计划值"含 30–40% serving 折扣,闭环质量决定折扣兑现与否。
6. **MTP**【未建】:单用户速度唯一的大杠杆(×1.5–1.8),排期建议在 8 卡吞吐达标后立即启动。
7. **fp8 KV 质量验证**:L2 阶梯十分钟一次,验过即全线容量翻倍。
