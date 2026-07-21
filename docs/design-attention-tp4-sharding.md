# 设计note：attention TP4 分片（M4 与 8 卡形态的共同前提）

- 2026-07-21 · **这是设计分析，不是实验**：输入全部是既有实测
  （E2F 的逐张量字节、E3F 的专家/非权重字节、E2F 的 NCCL 单次代价），
  推导部分**未经实测**，不得进入任何"实测"列。
- 结论先行：**E3F §4 的"attention 分片后 ÷4"是乐观的。**按 head 自然分片
  只到 **2.49×**，8 卡仍装不下（余 −0.27 GiB）；要拿到 ~4× 必须再对
  compressor / q_lora 做 column-parallel，那会**每层多 3–4 次集合通信**。

---

## 1. 结构：哪些能分、哪些不能

读 `Ratio4TorchAttention` 的 decode 前向（`ratio4_attention.py`）得到输出路径：

```
sparse_output [b,1,64,512]
  → reshape [b,1,o_groups=8, 4096]          # 每组 4096 值 = 8 个 head
  → einsum("bsgd,grd->bsgr", ·, wo_a[8,1024,4096]) → [b,1,8,1024]
  → flatten → [b,1,8192]
  → linear(wo_b[4096,8192]) → [b,1,4096]
```

**几何上是干净的**：64 head ÷ 4 rank = 16 head/rank = **恰好 2 个 o_group**
（每组 8 个 head）。所以 rank r 拿 head [16r,16r+16) = o_group [2r,2r+2)，
`wo_a` 按组切、`wo_b` 按输入列切，末尾一次 all-reduce 合并 [b,1,4096]。
这是标准 Megatron TP attention，Flash 的 `o_groups=8` 与 TP4 正好整除。

**但 KV 不随 head 分片**：`StaticRatio4KV` 的 `latent` 是
`[seq, LATENT_DIM=512]`，**没有 head 维**（MLA 式：所有 head 共用同一个 latent，
各自用自己的权重展开）。所以每个 rank 仍需完整 latent。
——E3F §4 把 KV 计在"非权重 3.79 GiB"里且未随分片缩小，这一点是对的。

按能否分片给 ratio-4 层的每张量归类（字节为 E2F 实测）：

| 类别 | 张量 | MB | 说明 |
|---|---|---:|---|
| **head 分片**（不新增集合通信） | `wq_b`, `wo_a`, `wo_b` | 201.3 | 末尾 all-reduce 顶替原有的那次 |
| head 分片（需 score all-reduce） | `index_wq_b` | 16.8 | indexer 也是 64 head，但 topk 必须全 rank 一致，`scores.sum(dim=2)` 前要合 |
| **column-parallel**（需 all-gather） | `compressor_wkv/wgate`（**FP32**）、`wq_a`、`wkv`、`index_compressor_*` | 54.5 | 产出的是**全 rank 共享的状态**，切了要 gather 回来 |
| **必须复制** | `index_weights_proj`、各 norm/ape/sink | 0.55 | |

## 2. 字节账（推导）

| 方案 | ratio-4 层/rank | 相对现在 |
|---|---:|---:|
| 现状（DP，全复制） | 273.2 MB | 1.00× |
| **只做 head 分片** | 109.6 MB | **2.49×** |
| head + column-parallel | 68.7 MB | 3.98× |

**2.49× 而不是 4×，是因为 FP32 的 compressor（33.6 MB/层）与 q_lora 下投影
不是 head-parallel。**

8 卡 PP2 一个 stage（22 层 = 2 window + 10 ratio-4 + 10 ratio-128），
接上 E3F 实测的 experts 17.53 + shared 0.43 + 非权重 3.79 GiB：

| 方案 | attention/卡 | 合计 | 卡 23.52 GiB |
|---|---:|---:|---:|
| 现状 | 5.09 GiB | 26.84 | **−3.32 ❌** |
| 只做 head 分片 | 2.04 | 23.79 | **−0.27 ❌ 仍不够** |
| head + column | 1.32 | 23.07 | +0.45 ⚠️ |
| ~~E3F §4 记的~~ | ~~1.27~~ | ~~23.02~~ | ~~+0.50~~ |

**E3F §4 的 +0.50 GiB 只在最激进的分片下才成立，而它没有说明这一点、
也没有计入随之而来的集合通信代价。**该表已按本文修正。

## 3. 代价：集合通信

E2F 实测 B=1 时 NCCL 每次约 **10.3 µs**（0.257 ms / 25 次）。

- **head 分片**：末尾 all-reduce 顶替现有的那次 → **净新增 0**；
- **indexer 分片**：+1 次/层（score all-reduce）；
- **column-parallel**：q_lora、compressor、index-compressor 各需 all-gather
  → +2~3 次/层。

一个 11 层 stage：head-only ≈ +0；全量 ≈ +33~44 次 ≈ **+0.34~0.45 ms/replay**。

对照收益：attention 字节 2.718 GB/stage → head-only 约 1.09 GB、全量约 0.68 GB，
按 E2F 就地标定的 806 GB/s，分别省约 **2.0 / 2.5 ms/stage**。
**即使全量分片，集合通信代价（~0.4 ms）也远小于带宽收益（~2.5 ms）。**
所以对 **M4 延迟**，做到哪一档都是净赚；**分档的意义在容量**（第 2 节）。

## 4. 数值：这**不是**逐位改动

E4F/E5F 两次融合都是逐位的，这次不是：

- o_proj 从"单 rank 上 8 组一次算完"变成"4 个 rank 各算 2 组再 all-reduce"，
  **求和顺序变了**；
- indexer 的 `scores.sum(dim=2)` 同理。

按 TARGET §9.6，改变求和序**不可能逐位，别浪费时间追**。所以放行判据是
§1.3 的两条软门：**分数不降（≥494/512）且 `top2_gap` 不越包络（≤0.9595）**，
而不是逐位对拍。这也意味着 E2F 探针的 `--mode ab` 每步逐位对拍**不适用**，
A/B 需要改成"数值见证 + 独立跑长门"。

## 5. 建议的推进顺序

1. **先做 head-only 分片**：几何干净、**不新增集合通信**、拿到 2.49×
   （M4 带宽收益的大头），且改动面最小；
2. 用它跑长门，确认软门通过（这是本项目第一次需要软门放行一个形态改动）；
3. 再决定是否加 column-parallel——**它的唯一动机是容量**（把 8 卡从
   −0.27 GiB 推到 +0.45 GiB），M4 的速度收益已经在第 1 步拿到大部分；
4. 容量结论出来后，才谈 §7.7 的标准版五行逐行重算。

⚠️ 注意第 3 步的性价比问题：column-parallel 用"每层 3 次集合通信"换
0.72 GiB 容量。8 卡是否值得，取决于 §7.7 五行里哪几行真的需要 8 卡。
**先别做，等五行的容量需求明确。**

## 6. 未决

- FP32 compressor 能否降到 BF16/FP8——这会同时改善容量与带宽，且与分片正交，
  但是**语义变更**，要独立的质量门。
- 8 卡 prefill 的 workspace 峰值仍未测（E3F §6.1），它可能比权重更早成为墙。
- 本文全部为推导；**任何数字进入 TARGET"实测"列前必须有实验背书**。
