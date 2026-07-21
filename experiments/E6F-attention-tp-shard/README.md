# E6F — attention TP4 分片（方案 A）

第三十二竖条，**进行中**。设计分析见
[`docs/design-attention-tp4-sharding.md`](../../docs/design-attention-tp4-sharding.md)。
本目录记实验步骤；已完成 **step 1（o-path 代数等价）**、**step 2（切片精确性）**、
**step 3（4 rank 真实 all-reduce 接线）** 与 **step 4（三种层型全部落地）**；
`tp_size=1` 时全部恒等。**尚未做**：stage 级集成（`physical_stage` 需要
把 tp_group 与切片接进 `new_attention`）、CUDA graph 捕图下的 all-reduce、
D0L 软门。

---

## step 1：分片后的 o-path 还是原来那个函数吗

在写任何 runtime 管线之前，先在**真实权重**上验证整个计划所依赖的那一条：
把输出路径按 head 切成 4 份、各自过自己的 `wo_a`/`wo_b` 切片再求和，
是否等于不切的结果——即差异**只来自求和顺序**，而不是算错了。

### 为什么这一步必须先做

TARGET §9.6：改变求和序**不可能逐位**。所以一旦上了 runtime，
"数值对不上"会有两种完全不同的成因——**索引算错**与**重排误差**——
而在满配流水里这两者很难区分。先在单层上把前者排除掉。

### 方法

用一个 FP64 参照当仲裁：三条 BF16/混合路径**各自**与 FP64 比，
而不是两两相比（否则分不清"谁动了"）。

三个变体，对应三种可选的实现：

| 变体 | wo_b GEMM | 四份求和 | 含义 |
|---|---|---|---|
| `bf16 reduce` | BF16 | BF16 | 最朴素的 TP 实现（Megatron 默认口径） |
| `upcast reduce` | BF16 | 升 FP32 后加 | 集合通信字节 ×2（8→16 KB/层），延迟无感 |
| `fp32 gemm floor` | FP32 | FP32 | **不是可发布选项**，用来测数值地板 |

### 结果（4 个 ratio-4 层，各 8 组真实幅度输入）

相对 FP64 的最大相对误差，以**未分片路径自身的误差**为 1.00×：

| 层 | 未分片 | bf16 reduce | upcast reduce | **fp32 gemm 地板** |
|---|---:|---:|---:|---:|
| L2 | 3.503e-03 | 2.034× | 1.305× | **1.000×** |
| L4 | 3.098e-03 | 2.078× | 1.494× | **1.000×** |
| L6 | 3.182e-03 | 1.679× | 1.404× | **1.000×** |
| L10 | 3.232e-03 | 1.921× | 1.251× | **1.000×** |

**地板四层全部恰好 1.000×。**这就是本步要的结论：

1. **索引代数完全正确**——把 BF16 舍入拿掉之后，分片路径落回未分片路径
   *自己的*误差上，一分不差。head↔o_group 的对齐、`wo_b` 的列切分与组的配对
   都没有错。
2. 多出来的误差**全部**是 BF16 舍入，且分两处、可分别处置：
   - `wo_b` 的 GEMM 输出就是 BF16——**每份 partial 在被任何人看到之前就已按
     全幅舍入过**。这是未分片路径不付的代价（它在一个 GEMM 里累加全部 8192 项）。
     **事后升 FP32 救不回来。**
   - 四份相加时的舍入——这一处升 FP32 就能救。

### 由此定下的实现选择

**发布用 `upcast reduce`**：把 partial 升到 FP32 再 all-reduce。
代价是集合通信字节 ×2（8→16 KB/层，在 ~10 µs 的延迟绑定下无感），
换回约 40% 的新增误差（2.03×→1.31× 一类）。

`fp32 gemm` 地板虽然是 1.000×，但**不可发布**：它要把 `wo_b` 读成 FP32，
而"少读 `wo_b` 的字节"正是分片的全部目的。记录它是为了知道地板在哪——
**万一软门在 1.3–1.5× 上过不去，这里有一个已知的、代价明确的备选。**

### 这一步**不能**决定什么

⚠️ 它**不能**说明模型级质量能不能过。1.3–1.5× 的相对误差是否可接受，
只有 D0L 软门说了算（TARGET §1.3：分数不降 + `top2_gap` 不越包络）。
本步是**必要不充分**。脚本自己的 `accepted` 字段因此只判"代数是否正确"
（地板是否落在 1.00×），不判发布与否。

### 覆盖范围说明

只在 ratio-4 层上跑了，但 **o-path 的三个张量在三种层型上形状完全相同**
（`wq_b`/`wo_a`/`wo_b` 各 67.109 MB，见 E2F 的逐张量实测），
所以代数结论对纯滑窗层与 ratio-128 层同样成立。
脚本目前只接 ratio-4 的 config 类，故未直接在另两种层型上跑。

---

## step 2：切片是否精确——把"整层"归约到 step 1

step 1 测的是 o-path。整层与 o-path 之间还差一段：`wq_b` 的按 head 切、
`attn_sink` 的按 head 切。step 2 证明**这一段是精确的**，于是整层的数值后果
就等于 step 1 那个数，不多不少。

论证分两半，只有一半需要测。

**由构造（读 einsum 下标得到，不是假设）**：主 attention 的两个收缩是
`bshd,bskd->bshk` 与 `bshk,bskd->bshd`——`h` 出现在两者的输出里、
**从不被求和**。所以每个 head 的结果只依赖它自己的 query 与共享 latent，
把 head 切到不同 rank **不可能改变任何一个 head 的输出**。
唯一对 head 求和的是 indexer 的 `scores.sum(dim=2)`，而**方案 A 刻意不切
indexer**（每个 rank 都算全部 64 个 index head）——这正是方案 B 需要一次
score all-reduce 的原因。

**由实测（本脚本）**：切片本身是否逐位。按行切 GEMM 的权重*应该*给出输出的
对应切片，但"应该"不是保证——cuBLAS 对 8192 行与 32768 行的操作数可能选不同
kernel、不同 split-K，而 split-K 会改变每个点积内部的求和序。所以要测。

| 检查 | 逐位 |
|---|---|
| `wq_b` 四片拼回 == 原张量 | ✅ |
| `wo_b` 四片按列拼回 == 原张量 | ✅ |
| `wo_a` 四片按组拼回 == 原张量 | ✅ |
| `attn_sink` 四片拼回 == 原张量 | ✅ |
| **切片后的 q 投影 == 完整投影的对应切片**（64 次试验） | ✅ |

**五项全逐位** ⇒ **整层的数值后果 = step 1 的 o-path 数（1.25–1.49×，
upcast reduce 口径），没有额外来源。**

> 走过的弯路：本想直接跑整层前向对比，但 `prepare_decode_plan` 要求的
> overlap 元数据谱系与 `seed_decode_payload` 给出的不一致（E1F 用的是
> **stateful** 路径，不是这个）。与其为一个测试去凑状态机，不如把问题
> 分解掉——分解之后的结论反而**更强**：它不仅说"整体差多少"，还说清了
> **差异只可能来自哪一处**。

## step 3：4 个真实 rank + 真实 all-reduce

step 1/2 在单卡上把数值定死了；step 3 验的是**接线**：四个进程各持自己的
分片、前向里走 `dist.all_reduce`，能否复现单实例不分片的结果。

刻意做窄：**一层、无 super-stage、无 CUDA graph**。目的是把"分片接对了没有"
与"它能不能扛住捕图和流水"分开——这两件事一起失败时很难区分。

titan065，4 rank，layer 4，B=1，8 步：

| 项 | 值 |
|---|---|
| 每 rank 局部几何 | 16 head / 2 组，`wo_b [4096, 2048]` |
| 最坏相对差（vs 不分片） | **6.897e-03** |
| 逐位 | ❌（**预期如此**，TARGET §9.6） |

> ⚠️ **口径提醒**：这里的 6.9e-3 是 **shard vs full**，而 step 1 的
> 1.25–1.49× 是 **各自 vs FP64**。两者不是同一个量：若二者都离真值约 3–4.6e-3
> 而方向不同，它们之间相差 ~6-7e-3 完全一致。**不要把这两个数并列比较。**

走过的第二个弯路：`prepare_decode_plan` 默认 `advance_overlap_state=False`，
于是 step 0 过、step 1 起因为 pending 槽仍是 −1 而失败。
（第一次遇到这个报错时误以为是 seeding 不对——实测 seeding 在 2048 处
给出的 `[2044,2045,2046,2047]` 与期望完全一致，问题在推进而非播种。）

## step 4：三种层型全部落地（ratio-128 与滑窗层）

一个 stage 有三种层型，只改 ratio-4 无法做 stage 级集成。ratio-128 与滑窗层
用的是**同一套 o-path**（`wq_b`/`wo_a`/`wo_b` 在三种层型上形状完全相同，
E2F 逐张量实测各 67.109 MB），所以改法一字不差地照搬：

| 文件 | 改动 |
|---|---|
| `attention.py`（ratio-128） | 配置 `tp_size/tp_rank` + 三个派生属性、`shard_ratio128_attention_weights()`、`_tp_reduce_output()`、3 处 o-path reshape + 3 处 query reshape 改 local、4 处输出接 reduce |
| `window_attention.py`（滑窗） | 同上；3 处 o-path、3 处 query、3 处输出 |

**注意这两个类的 prefill 也走同一份权重**（`__call__` 与 decode 方法共用
`self.weights`），所以两条路径必须一起改——只改 decode 会让 prefill 拿着
16 head 的权重按 64 head reshape。ratio-4 不同：它的 prefill 用的是另一个类
（`Ratio4FullPositionAttention`），本竖条没有动它。

顺带一个省事的地方：这两个文件的 sink 视图用的是 `query.shape[2]` 而不是
`cfg.num_heads`，所以自动跟随局部 head 数，不用改。

## 已落地的 runtime 改动（默认不改变行为）

| 改动 | 说明 |
|---|---|
| `Ratio4AttentionConfig.tp_size/tp_rank` | 默认 1/0；`validate()` 仍把 **全局** 几何钉在 Flash 的 64 head / 8 组上，另加"切分必须整除"的检查 |
| `local_num_heads` / `local_o_groups` / `group_width` | 派生属性；`group_width` 是**全局**量（切分只取整组，从不切窄一个组） |
| `shard_ratio4_attention_weights()` | 切 `wq_b`（行）、`wo_a`（组）、`wo_b`（列）、`attn_sink`（head） |
| 前向的 8 处 `cfg.num_heads` | 改为 `local_*`，两条 decode 路径各 4 处 |

**`tp_size=1` 时全部为恒等**：切片函数直接返回原对象，`local_*` 等于全局值。

## 下一步

1. ~~权重切片 + 前向改造 + 分布式接线 + 三种层型~~ **全部已落地**
   （step 3/4）。剩下：**stage 级集成**——`physical_stage.new_attention()`
   要接受并传下 `tp_group` 与分片配置；以及 **CUDA graph 捕图下的 all-reduce**
   （MoE 已有先例，但需实测）；
2. 单 stage 数值见证 + 速度 A/B（**注意：E2F 探针的逐步逐位对拍不适用**，
   见设计note §4——本次不是逐位改动）；
3. D0L 软门——本项目第一次用软门放行形态改动；
4. 通过后再谈 §7.7 五行的容量重算。

## Artifact

| 路径 | 内容 |
|---|---|
| `../../runtime/e6f_shard_equivalence.py` | step 1 脚本（三变体 + FP64 仲裁） |
| `results/layer{2,4,6,10}/o_path_equivalence.json` | step 1 四层结果 |
| `../../runtime/e6f_slice_exactness.py` | step 2 脚本 |
| `results/slice-exactness/slice_exactness.json` | step 2 五项检查 |
| `../../runtime/e6f_dist_layer_gate.py` | step 3 脚本（4 rank + 真 all-reduce） |
| `results/dist-single-layer/rank*.json` | step 3 结果 |
| `../../runtime/dsv4_direct/ratio4_attention.py` | 配置字段、`shard_ratio4_attention_weights()`、`_tp_reduce_output()`、前向 local 化 |
