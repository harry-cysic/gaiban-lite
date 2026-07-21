# C3F — 增量 chunked prefill:给 direct runtime 补上 `start_pos > 0` 的多 token 前向

第二十五竖条(2026-07-21,titan064 + titan065 实测)。Phase 4(chunked prefill
与 decode 交错)的前置能力。

## 动机

第二十四竖条([`D0L-long-prompt-oracle`](../D0L-long-prompt-oracle/README.md) §3.2)
在把 E2E 门推进 chunk 区间时,发现了一个**能力缺口**而不是性能问题:

> `Ratio4FullPositionAttention.__call__`(`ratio4_fullpos.py:550-553`)对
> `seqlen > 1` **强制要求 `start_pos == 0`**……即当前 runtime 不支持
> `start_pos>0` 的多 token prefill,增量 chunked prefill 今天写不出来。

三种层型全都有这条守卫(**改动前**的行号:`attention.py:1484`、
`window_attention.py:891`、`ratio4_fullpos.py:552`;本竖条把它们换成了
`is_chunk` 分支),所以整个 C2F 系列里的 "chunk" 只是**一次整段 `start_pos=0`
prefill 的行数**,从来不是把 prompt 切成几段喂。本竖条补上真能力。

## 结论速览

| 项 | 结果 |
|---|---|
| **语义门(逐位)** | 三种压缩器状态机在 9 种分段方案下**全部逐位相等**(27/27);top-k 索引选择在 2 种 ratio × 9 种分段下**全部精确相等**(18/18) |
| **层级等价** | 非逐位,且**不可能逐位** —— 整段与分段喂给每个 GEMM 的 M 不同。最差 `branch` rms_rel **1.2e-2**,在本仓库既有的 BF16 双 lane 容差 0.02 之内 |
| **整段控制臂** | `--prefill-chunk 0` 在改动后的代码上**精确复现** D0L 基线 **494/512**、逐 prompt 62/62/63/63/61/61/59/63、`top2_gap` max 0.9595 —— 改动对整段路径**完全惰性** |
| **长门(分段)** | 控制臂 494 → chunk=1000 **497**、chunk=1024 **491**、chunk=512 **491**、chunk=999 **489**。**位置级** 97.5–98.2% 的 token 与整段完全相同 |
| **短门回归** | torch **468/482**、tilelang **472/482** —— 两条冻结基线**逐 prompt 精确复现** |
| **prefill 吞吐** | 分段**更快**(与预期相反):4096 提示 2967 → 1387 ms(**2.14×**,chunk=1024)、→ 1290 ms(**2.30×**,chunk=512);2048 提示 1481 → 955 ms(**1.55×**) |
| **峰值显存** | 4096 提示单 rank 峰值 18.356 → 14.652 GiB(**−3.70 GiB**);stage 0 全程峰值 20.671 → 15.091 GiB(**−5.58 GiB**),收尾 free 0.615 → 7.182 GiB |

**一句话**:能力补上了,语义部分**可以证明是精确的**(状态机逐位 + 索引精确),
算术部分**证明不可能精确**(GEMM 形状),token 级差异在控制臂两侧**双向摆动**
(−5 ~ +3),而分段 prefill 在这台机器上**又快又省显存** —— 后者是意料之外的方向。

**验收口径提醒**:分段与整段**不是同一个算术**,所以"一致"不能按杠杆的
"输出必须相同"来判(见 §2.4"怎么读这张表")。个别分段方案的 `top2_gap` 已经
**超出基线包络**(chunk=999 的 1.4929、chunk=512 的 1.1664 vs 基线 0.9595),
所以本竖条**不建议**把 chunked prefill 当作默认打开的等价优化;它是 Phase 4 的
**能力前置**,应当带着自己的 golden / 容差进入下一竖条。

---

## 1. 三种层型的增量语义与 reference 依据

reference 里**没有**多 token `start_pos > 0` 这条分支可抄:`model.py` 每处都只有
`start_pos == 0`(整段)和 `start_pos > 0`(单 token)两支。所以本竖条的语义是
**推导**出来的,推导的锚点是一个关键观察:

> **reference 自己的 prefill 终态与 decode 运行态是同一个不变量。**

正因为两支收敛到同一个状态,"在中途停下"才是良定义的;否则分段根本没有正确答案。
逐层型的不变量与出处(每处决策的行号都写进了
[`runtime/dsv4_direct/chunked_prefill.py`](../../runtime/dsv4_direct/chunked_prefill.py)
的注释):

### 1.1 窗口层(`compress_ratio == 0`,L0/L1/L42)

**不变量**:槽位 `pos % 128` 存绝对位置 `pos`,只保留最近 ≤128 个位置。
prefill 分支 `model.py:518-523`(`seqlen <= win` 直写,否则 wrap 切分)与 decode
分支 `model.py:530`(`kv_cache[:, start_pos % win]`)是**同一条放置规则**的两种写法。

**增量写法**:一段 `[P, P+L)` 只写最后 `min(L, 128)` 行,槽位取绝对位置取模。
必须只写这些行 —— 对更长的段做整段 `index_copy_` 会出现**重复索引**(未定义行为)。

**难点不在压缩,在注意力的重索引**。整段 prefill 的注意力跑在**完整 raw 序列**上
(`model.py:528` 的 `kv` 就是这一段的全部行,窗口索引即绝对位置);decode 跑在
128 槽 ring 上。一个 chunk 两者都不行:它的 query 需要绝对位置
`[P-127, P+L)`,横跨上一段(只在 ring 里)和本段(还没进 ring,而且 `L > 1` 时
ring 装不下两者)。本实现用**并集布局**

```
[ ring 快照 (128) | 本段 raw (L) | 压缩行 (C) ]      offset = 128 + L
```

在 ring 推进**之前**构建,绝对位置 `q` 映射为

```
q >= P  ->  128 + (q - P)      本段行,保序
q <  P  ->  q % 128            ring 槽(model.py:530 的放置)
```

后者良定义,因为窗口分支只会问 `q >= p - 127 >= P - 127 > P - 128`,严格落在 ring
仍持有的 128 个位置内。列**顺序**保持按绝对位置升序 —— 与 reference 一致 ——
所以 sparse core 在 `k` 轴上的求和顺序不变,gather 到的行与顺序都与整段路径相同。

### 1.2 ratio-128 层(无 overlap)

`overlap = compress_ratio == 4`(`model.py:290`)对 ratio-128 为 False,所以:

- **不变量**:`kv_state[:, 0:r]`(`r = pos % 128`)= 当前**未满组**的 token。
  prefill 走 `model.py:333-335` 且 `offset = 0`(`model.py:329`);decode 走
  `model.py:356-357`(槽位 `pos % 128`)。**没有前一组的槽位**。
- **池化**:直接对每 128 行分组 softmax(`model.py:337-338`、`:342`),不经过
  `overlap_transform`。
- **增量**:段边界只需把**未满组缝合**起来 —— 本段第一组 = `kv_state[:, :r]` ++
  本段前 `head = 128 - r` 个 token,之后是 4 对齐(128 对齐)的整组,reshape 即可
  (零拷贝,与 reference 的 `kv.unflatten(1, (-1, ratio))` 同形)。尾部余数回写
  `kv_state[:, :tail]`。

### 1.3 ratio-4 层(**overlap 压缩器**,本竖条的重点)

overlap 压缩器的 `kv_state` / `score_state` 是**跨步进的破坏性状态**,这是任务点名的
难点。拆开看,不变量其实很干净:

| 槽位 | 内容 | prefill 出处 | decode 出处 |
|---|---|---|---|
| `[0:4]` | 最后一个**已完成**组的**全宽**投影(score 已加 `ape`) | `model.py:331-332`(`kv[:, cutoff-4:cutoff]`) | `model.py:353-354`(`state[:, :4] = state[:, 4:]` 滚动) |
| `[4:4+r]` | 当前**未满**组的 token,`r = pos % 4` | `model.py:333-335`(`remainder` 切分,`offset = ratio`) | `model.py:347-348`(槽位 `4 + pos % 4`) |
| `[4+r:8)` | **陈旧**,两支都不清 | — | — |

第三行是容易踩的坑:两支都把 `[4+r, 8)` 留成垃圾,但**都不会在覆盖前读到它** ——
池化只在 `(pos+1) % 4 == 0` 触发(`model.py:349`),那时四个槽位全是新的。所以
比较分段与整段的终态时,**只能比较存活槽位**,否则是在比垃圾。

**池化本身**(`model.py:339-342` 经 `overlap_transform`,`model.py:307-314`):
压缩行 `g` 对 8 个槽位做 softmax 加权和 —— 槽位 0..3 取**前一组**的**前半宽**
(`[..., :d]`),槽位 4..7 取**本组**的**后半宽**(`[..., d:]`)。decode 边界在
`model.py:350-351` 显式拼出同样的八行。

**增量的关键决策**:一个 chunk 的**第一组**,其 "前一组" 必须来自
`kv_state[:, :4]`,而不是 `overlap_transform` 给 group 0 的填充值(kv 填 0、score
填 `-inf`)。这正是**唯一**会写错的地方 —— 直接复用整段路径会让每个 chunk 的第一个
压缩行丢掉 overlap。

而且这**不是特例而是严格推广**:`start_pos == 0` 时 `kv_state[:, :4]` 恰好还是构造器
初值 `(0, -inf)`(`model.py:303-304`),与 `overlap_transform` 对 group 0 的填充
**逐位相同**。所以同一段代码同时覆盖首段与后续段 —— §2.1 的逐位门就是在验这一点。

**indexer 的压缩器同理**(`model.py:398` 用同一个 `Compressor` 类,只是
`rotate=True`、宽度 128),走完全相同的状态机;两者的完成行数必须一致,代码里有断言。

**topk 候选集在 `start_pos > 0` 的构造**:可见性用绝对位置写,`model.py:424-426`
的 `(p+1) // ratio` 原样搬过来即可 —— 压缩行 `g` 覆盖到位置 `4g+3`,所以
`(p+1)//4 > g` ⟺ `p >= 4g+3`,因果性与分不分段无关。`offset` 从整段的 `seqlen`
(`model.py:509`,raw 在前)改成 `128 + L`(并集布局)。

> **顺带一个不影响结果的细节**:`topk_count = min(index_topk, compressed_count)`
> 在分段时更小(候选池更小)。但对某个 query 行,实际选中的**有效**行数是
> `min(index_topk, visible)`,而 `visible <= compressed_count` 恒成立,所以两边选中
> 的真实集合相同,只是 `-1` 填充列数不同 —— 而 `-1` 被 torch core 与 tilelang kernel
> 一律忽略。§2.2 的索引门把这条验成了精确相等。

### 1.4 落地位置

| 文件 | 改动 |
|---|---|
| [`runtime/dsv4_direct/chunked_prefill.py`](../../runtime/dsv4_direct/chunked_prefill.py) | **新增**。三个层型共用的增量原语:`overlap_chunk_compress`、`plain_chunk_compress`、`chunk_window_topk_indices`、`chunk_compressed_topk_indices`、`chunk_raw_index_map`。模块 docstring 写了完整推导 |
| `runtime/dsv4_direct/ratio4_fullpos.py` | `__call__` 增加 `is_chunk` 分支 + `_chunk_compress`;可见性掩码改绝对位置;稀疏核/行分块选择由 `start_pos == 0` 改为 `seqlen > 1` |
| `runtime/dsv4_direct/attention.py` | `Ratio128TorchAttention.__call__` 增加 chunk 分支(含 ring 快照) |
| `runtime/dsv4_direct/window_attention.py` | `WindowTorchAttention.__call__` 增加 chunk 分支 |
| `runtime/dsv4_direct/static_kv.py` | `StaticLayerKV.chunk_write` |
| `runtime/dsv4_direct/static_window_kv.py` | `StaticWindowKV.chunk_write` |
| `runtime/e0ef2e_golden_gate.py` | `--prefill-chunk`(默认 0 = 原行为);MoE per-shape 缓冲按**分段长度**而非 prompt 长度注册;每 prompt 记录 chunk 长度/前向数/墙钟/激活峰值;`per_prompt` 增加 `predicted_tokens` |

三种层型的守卫都**只放开**了 `start_pos > 0 && seqlen > 1` 这一种新形状,
`start_pos == 0` 与单 token decode 两条既有路径一行未动。

---

## 2. 正确性

### 2.1 单层门 —— 状态机逐位([`c3f_chunked_prefill_gate.py`](../../runtime/c3f_chunked_prefill_gate.py))

直接驱动三个压缩器状态机,喂**同一份**预先算好的 FP32 投影(从一个张量切片,
所以任何 GEMM 的形状都不变),对照 `model.py:325-342` 的整段池化转写。
这样把**推导出的语义**与**浮点混杂因素**彻底分开,所以这里的目标就是精确相等。

- 流长 2200,9 种分段方案,含 **不整除组大小**(999/1201、5/7/11/2177)、
  **不整除窗口**(1000/1096/104、129/2071)与 **极短首段**(1/2199)。
- 三个压缩器:`ratio4_main`(宽 1024)、`ratio4_index`(宽 256)、`ratio128`(宽 512)。
- 比较 pooled 行 + 存活状态槽位(§1.3 的陈旧槽位排除在外)。

**结果:27/27 全部 `bitwise=True`。** 池化行、`kv_state`、`score_state` 逐位相等。

### 2.2 单层门 —— 索引精确

另一半新代码是索引数学(ring 重索引 + 因果掩码)。这一项比较的是**索引集合**,
所以没有任何浮点核能糊弄它:把分段产出的布局索引经
`chunk_raw_index_map` 的逆映射还原成绝对位置,与整段分支
(`model.py:262-264` / `:273-275`)在同样行上的输出逐元素对照(补齐 `-1` 填充后)。

**结果:18/18(2 种 ratio × 9 种分段)全部精确相等。**

### 2.3 单层门 —— 层级等价(数值门,附归因)

真实 checkpoint 权重(TP rank-0 复制)、单 GPU、序列 1024,10 种分段方案
(含 1000/24、999/25、517/507、3/1021、130/894、1/1/1022):

| 层型 | 最差 `branch` rms_rel | 最差 `max_abs` |
|---|---:|---:|
| window | 1.19e-2 | 2.50e-1 |
| ratio128 | 1.26e-2 | 2.50e-1 |
| ratio4 | 9.32e-3 | 2.50e-1 |

**非逐位,而且不可能逐位** —— 同一个门里的 `gemm_shape_probe` 直接证明了这一点:
只把该层自己的 `wq_a` 权重作用在**同样的行**上,整段(M=1024)与分段(M=256)
就已经有 **40.2% 的元素不同**(`max_abs` 9.77e-4)。sparse core 的 gather einsum
更是以**本次前向的 seqlen 为批维**,分段必然换 kernel。

两条佐证说明差异确实来自算术而非语义:

1. **`compressor_wkv` 的 FP32 GEMM 最差只差 1.57e-7**(FP32 几乎形状稳定),
   与之对应,§2.1 的压缩器状态机逐位相等。
2. **末段接近满长的方案里,ring 终态逐位相等**(`3/1021`、`130/894`、`1/1/1022`
   三种方案 ring `rms_rel = 0.000e+00`)—— 即 raw KV 的值与放置完全正确,
   残差只出现在 GEMM 形状真正变化的地方。

量级参照:本仓库既有的同类比较(`e0e2e_ratio4_selfcheck.py:62`,
"full-sequence GEMMs vs single-row GEMMs legitimately differ in low bits")
用的容差就是 **0.02**,本门最差 1.26e-2 在其内。

### 2.4 长门(D0L)

冻结配置与 D0L 基线完全一致(tilelang prefill sparse、eager HC、顺序 MoE、
`--max-seq-len 4224`、`--max-steps 64`、`--share-moe-buffers`),唯一变量是
`--prefill-chunk`。

#### 分段确实发生了 —— 证据

门逐 prompt 记录 `chunk_lengths` / `prefill_forwards`(纯观测):

| 臂 | 1024 提示 | 2048 提示 | 4096 提示 | MoE per-shape 形状 |
|---|---|---|---|---|
| chunk=0 | 1 段 `[1024]` | 1 段 `[2048]` | 1 段 `[4096]` | `[4, 4096, 8192, 16384]` |
| chunk=1024 | 1 段 | **2 段** | **4 段** `[1024]×4` | `[4, 4096]` |
| chunk=1000 | **2 段** `[1000,24]` | **3 段** | **5 段** `[1000×4, 96]` | `[4, 96, 192, 384, 4000]` |
| chunk=999 | **2 段** `[999,25]` | **3 段** | **5 段** `[999×4, 100]` | `[4, 100, 200, 400, 3996]` |
| chunk=512 | **2 段** | **4 段** | **8 段** | `[4, 2048]` |

`chunk=999` 是**最狠的一臂**:`999 % 4 == 3`,4096 提示的四个段边界依次落在
ratio-4 相位 **3 / 2 / 1 / 0** —— 把 §1.3 的未满组续接四种情况全部走了一遍。

#### 得分

| 臂 | 得分 | Δ vs 整段 | `top2_gap` max | **位置级一致**(与整段逐 token 比) |
|---|---:|---:|---:|---:|
| **整段(chunk=0)** | **494/512** (0.9648) | — | 0.9595 | — |
| chunk=1024 | 491/512 (0.9590) | **−3** | 0.7138 | **503/512 = 0.9824** |
| chunk=1000 | **497/512** (0.9707) | **+3** | 0.9492 | 501/512 = 0.9785 |
| chunk=999 | 489/512 (0.9551) | **−5** | **1.4929** | 499/512 = 0.9746 |
| chunk=512 | 491/512 (0.9590) | **−3** | **1.1664** | 499/512 = 0.9746 |

**完全确定性**:chunk=1024 独立跑了 3 次(`chunk1024` / `b` / `c`),3 次
491/512 且逐 prompt 完全相同;chunk=0 跑了 2 次,2 次 494/512 且逐 prompt 完全相同。
所以这些 Δ 不是 run 噪声,而是各自分段方案的确定性结果。

#### 怎么读这张表

分段与整段**不是同一个算术**,也不可能是(§2.3)。所以问 "分不分段一致吗" 有三层:

1. **语义层:精确相等**(§2.1 逐位 + §2.2 精确)。新代码算的那两样东西都被证明了。
2. **算术层:不可能相等**。同一份权重、同一批行,只要 M 变了 GEMM 就换 kernel。
3. **token 层:97.5%–98.2% 的位置产出完全相同的 token**,剩下 2–3% 是 BF16
   末位差异把 argmax 顶翻的位置。对 golden 的得分在控制臂两侧**双向摆动**
   (−5 / −3 / −3 / +3),四臂均值 492,极差 489–497。

第 3 层的**双向性**是关键判据:如果增量语义写错了,分段只会**单向变差**,而且
段越多越差。实际是 chunk=1000 比整段**高 3 分**,而段数最多的 chunk=512(4096 切
8 段)和段数较少的 chunk=999(切 5 段)都是负的 —— 与段数不单调,与"错误累积"
不相容,与"离散 argmax 放大末位噪声"相容。

**两处必须如实记下的负面项**:

- chunk=999 的 `top2_gap` max **1.4929** 与 chunk=512 的 **1.1664** 都**超出了
  基线自己的包络 0.9595**。这与 D0L 判否杠杆 A 时用的信号(1.4920 > 0.960)同量级。
  也就是说:分段不只是把边缘 token 翻面,个别位置的分歧幅度比基线见过的都大。
- 因此 **chunked prefill 不能当作"输出等价的优化"放行**。它是一条**语义等价、
  算术不同**的新路径,该按"新路径"的标准对待(需要自己的 golden 或自己的容差),
  而不是按"不得劣化"的杠杆标准。

#### chunk=999 最差,是不是未满组续接写错了?——不是

`chunk=999` 是唯一让 `pending != 0` 的臂,所以这个怀疑必须正面回答。三条证据:

1. §2.1 的状态机门里,`pending != 0` 的分段方案(999/1201、5/7/11/2177、1/2199、
   129/2071)**全部逐位相等**;§2.2 的索引门同样精确。
2. §2.3 的单层门里,`pending != 0` 与 `pending == 0` 的方案**误差量级一模一样**:
   ratio-4 上 `uneven_999_25`(pending=3)**1.217e-3** vs `uneven_1000_24`
   (pending=0)**1.193e-3**;`uneven_517_507`(pending=1)**9.316e-3** vs
   `2x512`(pending=0)**8.685e-3**。续接若写错,前者会爆掉而不是持平。
3. chunk=1000 与 chunk=999 段数完全相同(2/3/5),只差 `pending`,得分却是
   **+3 与 −5** 分居两侧 —— 差异随机而非随 `pending` 系统性偏移。

### 2.5 短门回归

本竖条动的是**共享代码**(三条 `__call__` 与门的 step 计划),所以冻结的 D0 短门
必须原样复现。短 prompt 只有 10–22 token,`--prefill-chunk` 在这里永远不会切分 ——
**这正是要点**:这一臂证明整段路径未被扰动。

短门有**两个**冻结基线,区别只在 prefill 稀疏核(`DSV4_PREFILL_SPARSE_BACKEND`):

| 臂 | 基线出处 | 基线 | 本竖条 | 逐 prompt |
|---|---|---:|---:|---|
| torch,eager | `out-e0e2e` | 468/482 | **468/482** ✅ | `[2, 27, 127, 124, 11, 22, 32, 123]` **逐条相同** |
| torch,fused | `out-e0e2e` | 468/482 | **468/482** ✅ | `[2, 28, 127, 124, 11, 22, 32, 122]` **逐条相同** |
| tilelang,eager | `out-e0e2e-tl-tilelang` | 472/482 | **472/482** ✅ | `[2, 29, 127, 124, 12, 22, 32, 124]` **逐条相同** |
| tilelang,fused | 无冻结基线 | — | 469/482 | `[2, 27, 127, 125, 12, 22, 32, 122]`(记录用) |

三条有基线的臂**逐 prompt 精确复现**,不只是总分相等。
(`out-e0e2e-tl-tilelang` 只跑过 `--hc-backends eager`,所以 tilelang+fused
没有可比基线,这里只作记录。)

> 一个容易含混的点已经查清:任务里说的"短门 472/482"是 **tilelang** 臂
> (`out-e0e2e-tl-tilelang` / `-tl-d0lregress` / `-tl-moealloc` 三次都是 472,
> 逐 prompt `[2, 29, 127, 124, 12, 22, 32, 124]`);**torch** 臂的冻结值是 468
> (`out-e0e2e` / `-c2f-fusedidx` / `-c2f-moealloc` 三次都是 468)。两条都要复现。

---

## 3. 性能:分段 vs 整段

任务预期是"分段会损失一些并行度但降低峰值显存"。**显存部分成立且幅度可观;
吞吐部分方向相反 —— 在这台机器上分段 prefill 明显更快。**

### 3.1 prefill 墙钟(rank 12 = stage3 tp0,含 PP 上游等待,即整条流水的 prefill 时延)

prompt 0 带 tilelang JIT 冷启(~16.5 s),下表已排除。

数字取自 `results/long-*.json` 的 prompt 1 / 3 / 6(每个长度各一条,均非 prompt 0)。

| prompt | 整段(chunk=0) | chunk=1024 | chunk=1000 | chunk=999 | chunk=512 |
|---:|---:|---:|---:|---:|---:|
| 1024 | **732 ms**(1 段) | 732 ms(1 段) | 804 ms(2 段) | 807 ms(2 段) | 565 ms(2 段) |
| 2048 | **1481 ms**(1 段) | 955 ms(2 段) | 1028 ms(3 段) | 1023 ms(3 段) | 798 ms(4 段) |
| 4096 | **2967 ms**(1 段) | 1387 ms(4 段) | 1454 ms(5 段) | 1456 ms(5 段) | 1290 ms(8 段) |

相对整段的加速比:

| prompt | chunk=1024 | chunk=1000 | chunk=999 | chunk=512 |
|---:|---:|---:|---:|---:|
| 1024 | 1.00× | **0.91×** | **0.91×** | 1.30× |
| 2048 | 1.55× | 1.44× | 1.45× | 1.86× |
| 4096 | **2.14×** | 2.04× | 2.04× | **2.30×** |

规律很干净:**prompt 越长、段越小,分段越划算**;唯一变慢的一格是把 1024 切成
`[1000, 24]` / `[999, 25]` —— 多一次几乎空转的 24–25 行前向,赔了 9%。整段路径
在长度上不是线性劣化而是**超线性**:732 → 1481 → 2967 ms 看着像线性,但同样
4096 行拆成 4 段 1024 只要 1387 ms,即同样的总行数、同样的核,只因**一次前向的
行数**从 4096 降到 1024 就快了一倍多。长 prefill 有明显的规模惩罚。

### 3.2 峰值显存

两个口径,都是从拉回本地的 JSON 复核的。

**(a) 每 prompt 激活峰值**(rank 12,每条 prompt 前 `reset_peak_memory_stats`):

| prompt | 整段 | chunk=1024 | chunk=1000 | chunk=999 | chunk=512 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 16.528 GiB | 14.652 | 14.686 | 14.688 | 14.191 |
| 2048 | 17.137 GiB | 14.652 | 14.686 | 14.688 | 14.192 |
| 4096 | **18.356 GiB** | **14.652** | 14.687 | 14.688 | **14.192** |

分段臂**与 prompt 长度无关地平**在各自的段长决定的水位上 —— 这正是 chunked
prefill 的主要价值。4096 上省 **3.70 GiB(−20%)**。

这 3.70 GiB 可以干净地拆成两项,因为 chunk=1024 臂**没有切分** 1024 的 prompt
(1 段),它们的峰值却已经比整段低 1.876 GiB:

- **1.876 GiB = MoE per-shape 缓冲注册**。整段臂每个不同的 prompt 长度都要注册
  一套 Marlin 缓冲(`global_row_shapes = [4, 4096, 8192, 16384]`);分段后所有
  prefill 前向的行数相同,只剩 `[4, 4096]`。
- **1.828 GiB = 前向自身的激活**(4096 行 vs 1024 行)。

**(b) stage 0 全程峰值与收尾 free**(rank 0,11 层,负载最重的 stage):

| 臂 | `global_row_shapes` | 峰值 allocated | 收尾 free |
|---|---|---:|---:|
| 整段(chunk=0) | `[4, 4096, 8192, 16384]` | **20.671 GiB** | **0.615 GiB** |
| chunk=1024 | `[4, 4096]` | 15.091 GiB | 7.182 GiB |
| chunk=1000 | `[4, 96, 192, 384, 4000]` | 15.181 GiB | 7.109 GiB |
| chunk=999 | `[4, 100, 200, 400, 3996]` | 15.183 GiB | 7.113 GiB |
| chunk=512 | `[4, 2048]` | **14.474 GiB** | **8.232 GiB** |

整段路径跑 4096 时 stage 0 只剩 **0.615 GiB** —— 已经贴着 OOM。分段后
free 回到 7–8 GiB。对照 D0L §1.3 记录的"reference 侧 4096 只剩 0.05 GiB、
8192 直接 OOM":**分段 prefill 正是把长 prompt 从容量悬崖上拉回来的那个杠杆**,
而且它同时还更快。

> chunk=1000 注册了 5 个形状(1000/24/48/96 各一套)比 chunk=1024 的 2 个多,
> 峰值因此略高(15.181 vs 15.091)。**段长整除 prompt 长度是有价值的**:
> 它让所有前向共享同一套 per-shape 缓冲。

---

## 4. 意外发现

### 4.1 分段 prefill 比整段**快** 1.5–2.3×,方向与预期相反

任务给的预期是"分段会损失一些并行度但降低峰值显存"。显存对了,吞吐**反了**:
4096 提示 2962 → 1291 ms(chunk=512)。整段路径在长 prefill 上有超线性的规模惩罚,
拆小后反而更快。这条改变了 Phase 4 的成本模型:chunked prefill 在这台机器上
**不是**用吞吐换显存的取舍,而是两头都赢。**唯一变慢的情形**是把一个已经不长的
prompt 切出一个很短的尾段(1024 → `[1000, 24]`,赔 9%)。

### 4.2 显存节省有一半来自 MoE per-shape 缓冲注册,不是激活

这一项事先没想到。Marlin per-shape 缓冲按**全局行数**注册(~80 KiB/行),整段臂
每个不同的 prompt 长度都要一套 `[4, 4096, 8192, 16384]`;分段后所有 prefill 前向
行数相同,只剩 `[4, 4096]`。在 chunk=1024 臂上,**未被切分**的 1024 提示峰值就已经
比整段低 1.876 GiB —— 那部分完全是缓冲注册,与切分无关。4096 上总共省 3.70 GiB
= 1.876(缓冲)+ 1.828(激活)。

推论:**段长整除 prompt 长度是有价值的**。chunk=1000 因为尾段长度各异,注册了
5 个形状,峰值反而比 chunk=1024 的 2 个形状高 0.09 GiB。

### 4.3 任务里的"短门 472/482"是 tilelang 臂;torch 臂冻结值是 468

这个歧义差点让我把一次**完美复现**读成 −4 回归。仓库里两条短门基线并存:
`out-e0e2e` / `-c2f-fusedidx` / `-c2f-moealloc` 三次都是 **468**,
`out-e0e2e-tl-tilelang` / `-tl-d0lregress` / `-tl-moealloc` 三次都是 **472**,
区别只在 `DSV4_PREFILL_SPARSE_BACKEND`。`run_c3f_short_regress.sh` 现在把后端做成
第二个参数,两条都跑。

### 4.4 `e0ef2e_golden_gate.py` 的 `result.json` 是**手挑子集**,新字段会被静默丢弃

`result.json`(`:1226-1244`)不是把 `result` 整个写出去,而是逐字段挑。我加的
`result["prefill_chunk"]` 因此没出现在 `result.json` 里 —— 而 `rank0.json`
(`:1224`,写的是完整 `result`)里有。**给后续竖条的提醒:分析要读 `rank0.json`,
往 `result.json` 加字段必须同时改那份白名单。**本竖条已把 `prefill_chunk` 补进去,
`summarize.py` 也直接读 `rank0.json`。

### 4.5 chunked prefill 把 runtime 从容量悬崖上拉了回来

D0L §1.3 记录 reference 侧 4096 只剩 0.05 GiB、8192 直接 OOM。runtime 侧同样紧:
整段跑 4096 时 stage 0 收尾只剩 **0.615 GiB**。分段后 free 回到 **7.1–8.2 GiB**。
按实测的两项(激活随段长而非 prompt 长度、per-shape 缓冲只需一套)外推,
chunk=1024 跑 8192 的峰值应当仍然是那条平线 ~15.1 GiB —— 但这是**外推,不是实测**:
D0L 的 golden 只到 4096(reference 侧 OOM),8192 目前**没有可比的 golden**,
要真正验证需要先解决 reference 侧的 `hc_post` 广播(D0L §1.3)。

### 4.6 单层门里 ring 终态在若干分段方案下**逐位相等**

`3/1021`、`130/894`、`1/1/1022` 三种方案的窗口层 ring 终态 `rms_rel = 0.000e+00`。
这些方案的末段接近满长(1021/894/1022 vs 1024),cuBLAS 显然选了与 M=1024 同族的
kernel,于是 raw KV **逐位相同** —— 顺带把"ring 放置与取值完全正确"钉死了,
残差只出现在 GEMM 形状真正变化的地方。

---

## 5. 复现

```bash
cd runtime

# 单层门(单 GPU)
./run_c3f_gate.sh layers                    # 三项检查全跑
./run_c3f_gate.sh smoke --skip-layers       # 只跑状态机 + 索引(无需权重)

# 长门(16 rank 双机)
./run_c3f_long_arm.sh chunk0    0           # 整段控制臂
./run_c3f_long_arm.sh chunk1024 1024
./run_c3f_long_arm.sh chunk1000 1000        # 不整除窗口
./run_c3f_long_arm.sh chunk999  999         # 不整除组大小,遍历全部 ratio-4 相位

# 短门回归
./run_c3f_short_regress.sh shortregress

# 汇总
python3 ../experiments/C3F-chunked-prefill/summarize.py \
  ../experiments/C3F-chunked-prefill/results/long-*.json
```

### 产物

| 文件 | 内容 |
|---|---|
| `results/single-layer-gate.json` | 单层门:27 条状态机记录 + 18 条索引记录 + 3 层型 × 10 分段的层级数值 + `gemm_shape_probe` |
| `results/long-chunk{0,512,999,1000,1024}.json` | 长门五臂的 `rank0.json`(含 `prefill_chunk`、逐 prompt `chunk_lengths` / `wall_ms` / `peak_allocated_gib` / `predicted_tokens`) |
| `results/short-{torch,tilelang}.json` | 短门两条基线的回归 |

**读 `rank0.json` 而不是 `result.json`** —— 原因见 §4.4。

### 环境指纹

titan064 + titan065,各 8×RTX 4090(24564 MiB),world_size 16(TP4×PP4),
torch 2.11.0+cu130,CUDA 13.2,`NCCL_P2P_LEVEL=SYS` + `NCCL_IB_DISABLE=0` +
`NCCL_SOCKET_IFNAME=enp33s0f0`,长门 `DSV4_PREFILL_SPARSE_BACKEND=tilelang`,
golden 为 D0L 冻结的 `oracle-long.json`(8 条,1024×3 / 2048×3 / 4096×2,
每条 64 token,共 512 个比较位)。

收尾两机显存已核:titan064 与 titan065 全部 16 张卡回到 **1 MiB**,无残留进程。
