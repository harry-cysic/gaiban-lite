# prefill 两杠杆:HC 边界融合(A)与 MoE 集合重叠(B)(2026-07-21)

第二十三竖条。起点是第二十二竖条的同口径实测 **tilelang attention 臂
25,308 input tok/s/stage**(3 轮,轮间 0.03%),归因 MoE 33.4%、attention 40.0%、
HC 24.1%。目标是把 prefill 推向 §5.3 的 30–40k 带。

**两杠杆分别计量的结论:**

| 杠杆 | 判活 | 同口径吞吐(3 轮均值) | 对 25,308 | E2E golden(冻结 472/482) |
|---|---|---:|---:|---|
| **A — prefill HC 边界融合** | **活** | **30,345** | **+19.9%** | **469/482 → 未过(−3)** |
| **B — MoE 集合/计算重叠** | 活但弱 | **25,691** | **+1.5%** | **473/482 → 过(+1)** |
| **A+B** | | **30,767** | **+21.6%** | 未跑(A 已单独未过) |

**一句话**:杠杆 A 是真杠杆,单独就把 prefill 送进 30–40k 带的下沿;但它**没过既定
的 E2E 验收门**(469 < 472),且**现有 E2E 门在结构上根本覆盖不到 prefill chunk
区间**(golden prompt 只有 10–22 token,§1.6)—— 门本身的这个缺口比那 3 个 token
更值得先修。杠杆 B **过了门**、显存不变,但收益被**硬件的 NCCL/计算并发上限
(实测 51.8%)**卡死在 +1.5%,已到结构性天花板,不建议继续投入。

过程中发现 **vLLM `mhc_fused_post_pre_tilelang` 在 num_tokens ≥ 1024 时数值是错的**
(不是精度低,是错),这是一条**新的、比已记录的 `with_norm` ≥128 token 更高的阈值**,
且它正好压在每一个 prefill chunk 上(§1.2)。decode 不受影响(§1.3)。

---

## 1. 杠杆 A — prefill HC 边界融合

### 1.0 为什么值得做

第二十二竖条的 component_walls(tilelang 臂,chunk 8192,11 层):

| 桶 | s | 占比 |
|---|---:|---:|
| total_instrumented | 1.3002 | |
| moe | 0.4337 | 33.4% |
| attention_ratio4 | 0.3611 | 27.8% |
| **hc** | **0.3134** | **24.1%** |
| attention_ratio128 | 0.1588 | 12.2% |
| norm | 0.0322 | 2.5% |

`hc` 桶是 44 个 eager HC op(11 层 × 4:attn 侧 hc_pre/hc_post + ffn 侧
hc_pre/hc_post)= **7.12 ms/op**。11 层链上有 **21 个可融合边界**
(11 个层内 attn→ffn + 10 个层间 ffn→下一层 attn),另有 2 个无融合搭档的
链首 `hc_pre`+norm 与链尾 `hc_post`。

### 1.1 微基准判活(`c2f_hc_prefill_gate.py`,单卡 titan065,真实 layer-11 HC 权重)

一个边界 = `hc_post → hc_pre → rms_norm`,两 backend 同输入:

| 布局 | 行数 | eager (ms) | fused 行切分 (ms) | 加速 | worst rel_fro |
|---|---:|---:|---:|---:|---:|
| prefill | 1024 | 1.998 | 0.663 | 3.02× | 2.31e-04 |
| prefill | 2048 | 3.894 | 1.214 | 3.21× | 2.43e-04 |
| prefill | 4096 | 7.813 | 2.617 | 2.99× | 2.38e-04 |
| **prefill** | **8192** | **15.516** | **5.316** | **2.92×** | **2.36e-04** |
| decode 对照 | 512 | 1.554 | 0.310 | 5.02× | 2.24e-04 |

**eager 侧与整机口径自洽**:8192 行边界 15.516 ms,整机 component_walls 推得
2×(0.3134/44) + 0.0322/22 = **15.71 ms**,差 1.2%。

判活。预测收益 21 × (15.516 − 5.316) = **−0.2142 s/pass**。

### 1.2 意外发现:kernel 在 num_tokens ≥ 1024 数值是错的

第一版微基准用合成权重(`hc_scale=1`)测到 worst rel_fro **0.45**;换成真实权重后
仍有 **0.107**,而 A5F 在 decode 形状记录的是 ≤9e-6。逐行数二分(真实权重):

| 行数 | 16 | 64…896 | **1024** | 2048 |
|---|---:|---:|---:|---:|
| worst rel_fro | 1.03e-03 | **2.2–2.4e-04** | **1.066e-01** | 1.082e-01 |

阈值干净地落在 (896, 1024],**且与布局无关**:decode 布局 `[B,1,hc,d]` 在
B=1024 时同样是 1.066e-01(与 prefill 布局逐位同值),B=512 时是 2.24e-04。
即**这不是"prefill 形状不支持",是"行数 ≥1024 走了另一个 kernel"**。

根因在 vLLM 侧,`model_executor/kernels/mhc/tilelang.py`:

```python
def _tilelang_hc_prenorm_gemm(x, fn, out, sqrsum, hidden_size, hc_mult,
                              tile_n=12, n_thr=512, n_splits=1):
    use_default_config = tile_n == 12 and n_thr == 512
    if n_splits == 1 and use_default_config and x.shape[0] >= 1024:
        hc_prenorm_gemm_block_m_tilelang(...)   # <-- 这一支
        return
```

而 `mhc_fused_post_pre_tilelang` 调它时**不转发 `tile_n` / `n_splits`**:

```python
_tilelang_hc_prenorm_gemm(residual_cur_2d, fn, gemm_out_mul,
                          gemm_out_sqrsum, hidden_size, hc_mult)
```

所以 `use_default_config` 恒为 True,`≥1024` 那一支**无法通过公开参数绕开**
(外层的 `tile_n=1` 只作用于 `num_tokens ≤ 16` 的 small-FMA 路径)。
`hc_prenorm_gemm_block_m_tilelang` 在本 sm_89 栈上**输出是错的**:`residual`
(即 hc_post 半边)仍对到 1.5e-05,但 hc_pre 半边的 `post` / `comb` / `hidden`
全部与参考失去相关性。JIT 日志里那句
`TileLang begins to compile kernel hc_prenorm_gemm_block_m_tilelang` 就是它。

> 这与已记录的"`with_norm` 分支在 ≥128 token 不等价"是**两条独立的阈值**。
> 本竖条这条更高(1024)、更隐蔽(走的是 `norm_weight=None` 的"安全"路径),
> 而且**每一个 prefill chunk(1024/2048/4096/8192)都在它上面**。

### 1.3 修法:按行切分(边界是逐行独立的)

`mhc_post`、pre-norm GEMM(每行 16384→24 的收缩加该行自己的平方和)、逐行的
sigmoid/sinkhorn —— 全部只碰一行。所以把一次调用切成 ≤896 行的块与一次大调用
**语义等价**。`FusedTilelangHCBoundaryBackend.MAX_ROWS = 896`,超过就均分成块
(`blocks = ceil(rows/896)`,块长 `ceil(rows/blocks)`,无短尾),输出写进一组
预分配缓冲。

代价:8192 行 3.704 ms(错的)→ 5.316 ms(对的),即**为正确性付 1.6 ms/边界**,
仍是 eager 的 2.92×。

**decode 不受影响**:已记录的最大 `local_batch` 是 192,远低于 896,所以 decode
走的仍是单次调用路径,**逐位不变**。微基准里 decode 对照(512 行)split 与
nosplit 两臂 rel_fro **完全相同**(2.24e-04),即切分逻辑未触发。

### 1.4 链级数值门(`--gate-hc`,真实 11 层,chunk 8192)

21 个边界,两 backend 锁步(同输入,链沿 eager 前进):

| 张量 | worst rel_fro |
|---|---:|
| residual | 1.60e-05 |
| **hidden** | **1.24e-03** |
| post | 7.07e-05 |
| comb | 3.83e-05 |

**校准**:已被接受并冻结的 tilelang attention 臂(第二十一竖条),其层级门
worst_branch_rel_fro 是 **4.68e-03** —— 比本竖条边界的 1.24e-03 **大 4 倍**,
且它 E2E 拿到 472/482(优于 468 基线)。所以单边界误差量级本身不是问题。

自由复合的 11 层 stage 输出:**rel_fro 0.225**(max_abs 10.54,|ref|max 22.25,
无非有限值)。这是链的混沌放大(每层 MoE top-k 路由会把 bf16 级扰动放大成
不同专家选择),不是单点误差 —— 但它也说明**单靠 stage 输出无法判定**,必须
看 token 级。

### 1.5 E2E golden 门 —— **469/482,未过**

`e0ef2e_golden_gate.py`,16 rank 双机,tilelang prefill sparse core,
**只把 prefill 的 HC 换成 fused,decode 的 HC 保持 eager**
(新增 `--fused-scope decode|prefill|both`,本臂用 `prefill`;冻结臂的
`--hc-backends eager` 在 decode 上走的是与默认逐位相同的重构链,所以两次
run 的**唯一差异就是 prefill HC**)。

| 臂 | 合计 | 逐 prompt |
|---|---:|---|
| 冻结 tilelang-prefill(eager HC) | **472/482** | 2, **29**, 127, **124**, **12**, 22, 32, 124 |
| 本竖条 fused-prefill HC | **469/482** | 2, **28**, 127, **123**, **11**, 22, 32, 124 |

三个 prompt 各早一 token 分歧(#1、#3、#4)。match_rate 0.9730,
mismatch_top2_gap max 0.8425 / median 0.1390 / min 0.0462
(冻结臂 0.5621 / 0.3254 / 0.0145)。

**按本竖条既定验收门("不劣化于 472/482"),杠杆 A 未过。** 如实记录,不放宽。
注意 469 仍高于最初的 eager torch-prefill 基线 468。

### 1.6 这个门覆盖不到 prefill chunk 区间(必须记录的门本身的局限)

golden oracle 的 8 条 prompt 长度是 **10, 13, 16, 16, 16, 18, 22, 13 token**。
也就是说 E2E 门里的 prefill 每 lane 只有 10–22 行:

- **完全没有触发 §1.3 的行切分**(远低于 896),
- 更没有触发 §1.2 的 ≥1024 错误分支,
- 它测到的是 fused 边界在**小行数**下(rel_fro ~2.3e-04)对 token 的影响。

结论有两面:一面是**即使在"好"的精度档位,这条链也会掉 3 个 golden token**,
说明链对 bf16 级扰动确实敏感;另一面是**这个门无法为 chunk 8192 的部署背书** ——
要上杠杆 A,需要一个能跑到 chunk 区间的 token 级正确性仪器(现有 chunk 区间证据
只有 §1.4 的 stage 输出 rel_fro 0.225,不足以判定)。

### 1.7 吞吐(同口径,chunk 8192,11 层 L11-21,iters 5 / warmup 2,3 轮)

见 §3。

---

## 2. 杠杆 B — MoE 集合/计算重叠

### 2.1 实现

`TP4MoE.enable_collective_overlap(blocks)`。原来 `__call__` 顺序执行
all_gather → gate → marlin → shared → combine → reduce_scatter,两个集合
17.7 ms/层全裸露。两集合之间的每一步都是逐行的,所以按行块切开做流水:

- **布局**:`all_gather_into_tensor` 按 rank-major 写,所以逐块 gather 得到的是
  **block-major** 缓冲 `gathered[k·B·W + r·B + j]`(原来是 `gathered[r·L + i]`)。
  这是一个行置换,而且它对**两个集合都自洽**:配对的
  `reduce_scatter_tensor(reduced[kB:(k+1)B], combined[kBW:(k+1)BW])` 取回的正好是
  本 rank 自己的第 k 块行,拼起来就是正常行序。两边都是**连续视图**,不需要重排拷贝。
- **流水**:K 个 all_gather 一次性 `async_op=True` 发出(排在 NCCL 流上边算边排空),
  第 k 块算完立刻发它的 reduce_scatter,与第 k+1 块的计算重叠。
- **回退**:只有"纯 prefill 配置"走流水路径(learned 路由、无 route_override、
  无 trace/digest/observer/capture、无注入 alignment provider、行数整除块数),
  其余一律回退顺序路径。

### 2.2 数值 —— 非逐位,语义变更(如实报告)

`--gate-moe-overlap`,同一 hidden 过顺序与流水两条路径(真实 layer 11,8192 行/rank):

| blocks | 逐位相同 | rel_fro | max_abs | 不一致元素 |
|---:|---|---:|---:|---:|
| 2 | **否** | 4.11e-05 | 1.95e-03 | 1,856 / 33.6M |
| 4 | 否 | 2.86e-04 | 3.91e-03 | 117,733 |
| 8 | 否 | 2.92e-04 | 3.91e-03 | 121,047 |

**不是逐位的。** 行置换改变了每行落在哪个 Marlin M-block —— 每行的数学本身不变
(K 方向收缩与行的 M 位置无关),但 M 分组会改变 Marlin 的 kernel/grid 选择,
W4A8 下还可能改变动态激活量化的分组。量级(4e-05 ~ 3e-04)在 bf16 舍入档,
比已接受的 tilelang attention 臂(4.68e-03)小一个量级,但**它是语义变更,
不是"预期逐位"**,按门处理。

> 与第二十二竖条的 combine 重写形成对照:那一条是**穷举 2³² 对、0 位差**的
> 真·逐位等价;这一条不是。不要混为一谈。

### 2.3 吞吐与块数扫描

| 配置 | tok/s | 对 25,308 | moe 桶 (s) | ms/层 |
|---|---:|---:|---:|---:|
| 顺序(冻结基线) | 25,308 | — | 0.4337 | 39.29 |
| **blocks=2** | **25,759** | **+1.8%** | **0.4113** | **37.39** |
| blocks=4 | 25,700 | +1.5% | 0.4156 | 37.78 |
| blocks=8 | 24,989 | **−1.3%** | 0.4509 | 40.99 |

块数越多越差,blocks=8 比不重叠还慢。可隐藏的量:blocks=2 时
(8.74+9.06)/2 = 8.9 ms/层,**实际只藏下 1.90 ms/层(21%)**。

### 2.4 归因:硬件的 NCCL/计算并发上限是 ~52%(`c2f_moe_overlap_probe.py`)

不带任何 MoE 机制的三段式探针(4 rank,真实 MoE 载荷,titan065):

| | p50 (ms) |
|---|---:|
| all_gather 单独 | 8.549(23.55 GB/s,确认走 P2P) |
| BF16 GEMM 单独 | 6.234 |
| 两者同时发出 | **11.555** |
| 串行和 | 14.783 |
| 完美重叠 | 8.549 |
| **实测重叠率** | **51.8%** |

**硬件只能藏下一半。** 原因与第二十二竖条同源:GPU0–3 是 NODE 距离(跨 PCIe host
bridge),NCCL 的 P2P 传输由 **SM 驱动**而非独立 DMA 引擎,于是集合通信直接和
marlin 抢 SM,"并发"退化成分时。

把这个上限代回:blocks=2 理论可藏 0.52 × 8.9 = **4.6 ms/层**,实测 1.90 ms/层,
差额是**每块的额外开销**(更多 kernel launch、更多 alignment、M 更小的 Marlin
效率下降)—— 这也解释了 blocks=8 为什么会翻负。

**判定:杠杆 B 实现成立、方向正确,但在本硬件上结构性封顶。** 即便把每块开销压到 0,
blocks=2 也只值 ~4.6 ms/层 ≈ 0.05 s/pass ≈ **+4%**,拿不到当初按"17.7 ms 全裸露"
估的那个量级。

### 2.5 E2E golden 门 —— **473/482,过**

见 §3.2。

---

## 3. 验收

### 3.1 吞吐 —— 同口径 3 轮(chunk 8192,11 层 L11-21,iters 5/warmup 2,W4A8 + fused indexer + tilelang sparse)

每一份结果 JSON 的 `moe_collective_selfcheck` 均落在 **23–24 / 22–23 GB/s**,
确认全部走 P2P(未静默回退 SHM)。

| 臂 | 轮 | input tok/s/stage | moe (s) | hc (s) | norm (s) | attn4 (s) | attn128 (s) | total_instr (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 冻结基线(22 竖条) | 均值 | 25,308 | 0.4337 | 0.3134 | 0.0322 | 0.3611 | 0.1588 | 1.3002 |
| **杠杆 A** | r1 | 30,391 | 0.4321 | 0.1292 | 0.0015 | 0.3610 | 0.1585 | 1.0829 |
| 杠杆 A | r2 | 30,332 | 0.4330 | 0.1294 | 0.0015 | 0.3613 | 0.1589 | 1.0847 |
| 杠杆 A | r3 | 30,312 | 0.4334 | 0.1293 | 0.0015 | 0.3619 | 0.1590 | 1.0857 |
| **杠杆 A 均值** | | **30,345** | **0.4328** | **0.1293** | 0.0015 | 0.3614 | 0.1588 | 1.0844 |
| **杠杆 B** | r1 | 25,704 | 0.4129 | 0.3134 | 0.0322 | 0.3612 | 0.1586 | 1.2792 |
| 杠杆 B | r2 | 25,682 | 0.4134 | 0.3138 | 0.0322 | 0.3614 | 0.1587 | 1.2805 |
| 杠杆 B | r3 | 25,687 | 0.4140 | 0.3137 | 0.0322 | 0.3613 | 0.1587 | 1.2808 |
| **杠杆 B 均值** | | **25,691** | **0.4134** | 0.3136 | 0.0322 | 0.3613 | 0.1587 | 1.2802 |
| **A+B** | r1 | 30,781 | 0.4143 | 0.1295 | 0.0015 | 0.3635 | 0.1608 | 1.0701 |
| A+B | r2 | 30,755 | 0.4143 | 0.1297 | 0.0015 | 0.3634 | 0.1610 | 1.0704 |
| A+B | r3 | 30,764 | 0.4144 | 0.1294 | 0.0015 | 0.3636 | 0.1608 | 1.0703 |
| **A+B 均值** | | **30,767** | **0.4143** | **0.1295** | 0.0015 | 0.3635 | 0.1609 | 1.0703 |

**轮间离散与增益**:

| 臂 | tok/s | 离散 | 对 25,308 | 变的是哪个桶 | 单池 T(`1/T = 1/D + 8/P`,D=8733) |
|---|---:|---:|---:|---|---:|
| 冻结基线 | 25,308 | 0.03% | — | — | 2,322 |
| **杠杆 A** | **30,345** | 0.26% | **+19.9%** | hc+norm 0.3456 → **0.1308**(−0.2148) | **2,644** |
| **杠杆 B** | **25,691** | 0.09% | **+1.5%** | moe 0.4337 → **0.4134**(−0.0203) | **2,348** |
| **A+B** | **30,767** | 0.08% | **+21.6%** | 两者叠加,近似可加 | **2,670** |

**A/B 干净**:杠杆 A 的 moe 桶(0.4328)与基线(0.4337)差 0.2%,attention 两项
差 ≤0.1% —— 只有 hc+norm 动了;杠杆 B 的 hc/norm/attention 逐项与基线吻合到
0.1% —— 只有 moe 动了。

**微基准预测校验**:杠杆 A 微基准预测 21 × (15.516 − 5.316) ms = **−0.2142 s**,
整机实测 **−0.2148 s**,命中到 **0.3%**。

### 3.2 正确性

`e0ef2e_golden_gate.py`,16 rank 双机,tilelang prefill sparse core:

| 臂 | 合计 | 逐 prompt | 判定 |
|---|---:|---|---|
| 冻结 tilelang-prefill(eager HC,顺序 MoE) | **472/482** | 2, 29, 127, 124, 12, 22, 32, 124 | 基线 |
| **杠杆 A**(prefill HC fused,decode HC 保持 eager) | **469/482** | 2, **28**, 127, **123**, **11**, 22, 32, 124 | **未过**(−3) |
| **杠杆 B**(MoE 重叠 blocks=2,HC 全 eager) | **473/482** | 2, 29, 127, **125**, 12, 22, 32, 124 | **过**(+1) |

- **杠杆 B 过门**:473 ≥ 472。逐 prompt 只有 #3 由 124 变 125(多对一个),
  mismatch_top2_gap max 0.5621 与冻结值 0.56206 一致。已核实 16 个 rank 的结果
  JSON 都带 `moe_overlap_blocks = 2`,即流水路径确实生效(8 条 prompt 里 6 条
  长度为偶数、整除块数,另 2 条长度 13 自动回退顺序路径)。
  注意 +1 不是"更准",与 −3 同性质,都是 bf16 级扰动在混沌链上的重新落点。
- **杠杆 A 未过**:见 §1.5、§1.6。
- **A+B 合并臂未跑 E2E**:杠杆 A 已单独未过,合并臂不可能优于它,故未消耗机时;
  若后续要推 A,需先解决 §1.6 的门覆盖问题再重跑合并臂。

### 3.3 显存(两机收尾均已检查,run 后 8 卡全部回到 1 MiB)

| 臂 | peak allocated | peak reserved |
|---|---:|---:|
| 冻结基线 | 20.409 GiB | 20.502 GiB |
| 杠杆 A | 20.409 GiB | 20.502 GiB |
| 杠杆 B | 20.414 GiB | 20.508 GiB |
| A+B | 20.414 GiB | 20.508 GiB |

**高水位实质不变**(+5 MiB)。杠杆 A 的行切分缓冲与杠杆 B 的分块 alignment 都不在
进程峰值相位上。(注:`--gate-hc` 那一次 run 的 peak 是 21.22 GiB,因为门本身要同时
建两条 lane 做 A/B —— 那是门的开销,不是臂的开销。)

---

## 4. 距离 30–40k 带 与 下一步

**当前位置**:杠杆 A 单独 **30,345**、A+B **30,767** —— 都已进入 §5.3 的
**30–40k 带,但只在下沿**;杠杆 B 单独 25,691 不足以进带。

**A+B 之后的预算**(total 1.0703 s/pass):

| 桶 | s | 占比 | 备注 |
|---|---:|---:|---|
| moe | 0.4143 | 38.7% | 已是最大项;其中集合通信裸露 ~0.174 s |
| attention_ratio4 | 0.3635 | 34.0% | **新的第二大项,且是最大的单一非 MoE 项** |
| attention_ratio128 | 0.1609 | 15.0% | |
| hc | 0.1295 | 12.1% | 已从 24.1% 降到 12.1% |
| norm | 0.0015 | 0.1% | |

**到 40k 需要再砍 0.251 s**(total → 0.819 s)。可动的量:

1. **MoE 集合的剩余裸露 ~0.174 s** —— 但 §2.4 的 52% 硬件上限意味着**最多再拿
   ~0.087 s**,且要求把每块开销压到接近 0。这是有上限的、已量化的一条路。
2. **attention_ratio4 0.3635 s** —— 现在是最大的非 MoE 项,且第二十一竖条只换了
   sparse core(torch → tilelang),indexer / topk / gather 侧没动过。**这是下一
   竖条投入产出比最高的方向**,也是唯一有 0.25 s 量级空间的桶。
3. marlin 本体 ~0.167 s(15.17 ms/层)—— W4A8 已上,继续压需要换 kernel。

**下一步建议(按优先级)**:

1. **先修门,再推杠杆 A。** 杠杆 A 值 +20%,但 §1.6 暴露的问题比这 3 个 token
   更严重:**现有 E2E 门看不见 prefill chunk 区间**。建议做一个 chunk 区间的
   token 级仪器(例如用长 prompt 重采 golden,或在 chunk 8192 下做
   eager-vs-fused 的 logits/top-k 一致性统计),否则任何 prefill 侧语义变更都
   缺少可信验收。做完之后杠杆 A 的 469 才有意义可判(是"这条链本来就对 bf16
   敏感",还是"融合边界确实更差")。
2. **杠杆 B 可以直接收下**(过门、+1.5%、显存不变),但**不要再往上投**:
   52% 的硬件上限已经量化,剩余空间 ~+2–3%,不值一条竖条。
   若要收,建议默认 `blocks=2`(不是 4,更不是 8)。
3. **下一条竖条打 attention_ratio4**,那里有 0.36 s、且尚未被系统性优化过。
4. 顺带:把 §1.2 的 ≥1024 行 kernel bug 报给 vLLM / 在本仓 backend 里保留
   `MAX_ROWS` 硬上限,防止未来有人在 decode 侧把 batch 推过 896 而静默出错。

---

## 5. 意外发现(原样记录)

1. **`mhc_fused_post_pre_tilelang` 在 num_tokens ≥ 1024 数值错误**(§1.2)。
   根因是 vLLM `_tilelang_hc_prenorm_gemm` 的 `x.shape[0] >= 1024` 分支加上
   外层不转发 `tile_n`,**无法用公开参数绕开**。已用行切分(MAX_ROWS=896)规避。
   **影响面**:decode 侧最大 `local_batch` 192,从未触发,所有已冻结 decode 数字
   不受影响;但这是 `FusedTilelangHCBoundaryBackend` 一条**此前未记录的行数上限**,
   任何未来把它用到 ≥1024 行的地方都会静默出错。
2. **A5F 的 decode 数值门在真实权重下"被遮蔽"了一部分敏感度**。真实 `hc_scale` 是
   **[0.157, 0.059, 0.199](ffn)/[0.131, 0.031, 0.128](attn)** —— 很小,它乘在
   GEMM logits 上再进 sigmoid/sinkhorn,所以 GEMM 的差异对 `post`/`comb` 的
   传导被压扁。用 `hc_scale=1` 的合成权重同一 kernel 的 rel_fro 是 **0.45**,
   真实权重是 **0.107**。结论不变(都是错的),但提醒:**HC 边界的数值门只在真实
   `hc_scale` 下有意义,合成权重会同时高估和低估**。
3. **E2E golden 门覆盖不到 prefill chunk 区间**(§1.6):8 条 golden prompt 只有
   10–22 token。凡是"只在长 chunk 下才改变行为"的 prefill 改动(本竖条的行切分、
   任何 ≥1024 行的 kernel 分支、按 chunk 分块的实现),**这个门都看不见**。
   建议:prefill 侧的语义变更需要一个 chunk 区间的 token 级仪器。
4. **NCCL P2P 在本机吃 SM,通信/计算并发上限 ~52%**(§2.4)。这是一条可复用的
   平台常数:任何"把集合通信藏进计算"的 prefill/decode 设计,收益上限都是
   裸露时间的一半左右,不是全部。
5. **杠杆 A 的收益预测精度**:微基准预测 −0.2142 s/pass,整机实测
   −0.2148 s/pass(hc+norm 桶 0.3456 → 0.1308),**命中到 0.3%**。与第二十二竖条
   combine 重写的 0.2% 一样,说明 component_walls 口径对这类结构改动是可信的预测器。

---

## 6. 产物

| 文件 | 内容 |
|---|---|
| `hc-boundary-micro-gate.json` | 杠杆 A 微基准(真实权重,prefill 1024–8192 + decode 512 对照,split/nosplit 双臂) |
| `hc-boundary-rowcount-bisect.json` | ≥1024 行阈值的逐行数二分 |
| `hc-chain-gate.json` | 21 边界锁步门 + 复合 stage 输出 |
| `e2e-hcfused-prefill.json` | 杠杆 A E2E golden(469/482) |
| `moe-overlap-gate.json` | 杠杆 B 数值门(blocks 2/4/8) |
| `moe-overlap-blocks2.json` / `moe-overlap-blocks8.json` | 块数扫描 |
| `moe-overlap-concurrency-probe.json` | NCCL/计算并发上限探针(51.8%) |
| `c2f-chunk8192-lever{A,B,AB}-r{1,2,3}.json` | 吞吐各 3 轮 |

代码:
`runtime/dsv4_direct/hc_boundary_backend.py`(行切分 + `fused-nosplit` 诊断臂)、
`runtime/dsv4_direct/moe_runtime.py`(`enable_collective_overlap` 与流水路径)、
`runtime/c2f_prefill_stage_bench.py`(`--hc-backend` / `--moe-overlap` /
`--gate-hc` / `--gate-moe-overlap` 与融合边界链)、
`runtime/e0ef2e_golden_gate.py`(`--fused-scope` / `--moe-overlap-blocks`)、
`runtime/c2f_hc_prefill_gate.py`、`runtime/c2f_moe_overlap_probe.py`、
`runtime/c2f_overlap_summarize.py`;
launcher `run_c2f_hc_gate.sh`、`run_c2f_prefill_overlap.sh`、
`run_c2f_overlap_probe.sh`、`run_e0e2e_hcfused_arm.sh`(全部带
`NCCL_P2P_LEVEL=SYS`)。
