# E7F 单路 serving：Blocker A / B 的设计判断（design-first）

- 日期：2026-07-22 · 状态：**设计判断，实现前先读**
- 目标：**专业版 16 卡系统上，把真实 prompt 接到图 decode 路径**（单路操作点）。
- 口径纪律：**每个操作点独立**。golden gate 跑 8192 档 prefill 是它的职责（冻结长
  prefill 质量），**不是单路 serving 的操作点**。单路 = 交互式短轮次、prefill 按真实
  请求长度。约束出现时，先问**是哪个操作点产生的**。
- 标注纪律：**推导（derived）**与**实测（measured）**分列，不混。

---

## Blocker A：sub-2047 无图路径（先做，独立且便宜）

### 问题（measured）

图/stateful decode 在 `ratio4_attention.py:956` 断言**饱和的定形 index top-k**：
`(start_pos+1)//COMPRESS_RATIO ≥ index_topk`。Flash 的 `index_topk=512`、
`COMPRESS_RATIO=4` ⟹ **start_pos ≥ 2047**。低于此，ratio-4 层没有可捕图的路径，
只剩 eager（§4.3 冻结事实 210 ms/步）。**这个区间正是单路承诺针对的短交互轮次。**

**层型定位（measured，E7F §7.9 隔离实验）**：sub-2047 的墙**只在 ratio-4**。
- ratio-128 stateful decode 只要 `start_pos ≥ WINDOW_SIZE(128)`（`attention.py:1217`），
  用"masked direct control"处理可变压缩行，**128 后即饱和**；
- window 层滑窗 128，**128 后即饱和**。
故 A **只改 ratio-4**。

### 机制（measured：runtime 与 reference 两侧都已核对）

**非 stateful（变形，reference-faithful、D0L 已验）**——`prepare_decode_plan`：
- `compressed_after = (start_pos+1)//4`；
- `index_topk_count = min(index_topk, compressed_after)`（**变**）；
- `sparse_width = WINDOW_SIZE + index_topk_count`（**变形**）。

**stateful（定形，需饱和）**——`prepare_stateful_decode_plan`：
- `total_topk = WINDOW_SIZE + index_topk`（**定形**，这正是图要的）；
- 断言饱和，因为 topk 恒取 `index_topk` 个；低于饱和会取到 padding。

**关键（measured）**：inline decode 的 padding 现状。`ratio4_attention.py:1315`
`scores.masked_fill_(~visible, -inf)` 只掩 **indexer 选择分数**（选哪些压缩行），
topk(index_topk) 之后 gather + **attention softmax（:1360）不再掩 padding**。
所以低于饱和时，padding 候选（index 分 -inf 但仍被 topk 选进）会 **污染 attention
softmax**——这正是断言饱和的原因。

**reference 怎么掩非候选（measured，`reference/inference/model.py`）**：
- `:427 topk_idxs = index_score.topk(min(index_topk, end_pos//ratio))` —— 变数目；
- `:429-430` 越界候选置 **-1 哨兵**：`torch.where(mask, -1, topk_idxs+offset)`；
- `:528/533 sparse_attn(q, kv, sink, topk_idxs, scale)` —— **kernel 把 -1 当掩码**。

即 **reference 用 -1 哨兵 + sparse_attn 掩码**表达"padding 不进 softmax"。
runtime 的 inline decode 路径**没有**这一步——A 要补上等价的 softmax 掩码。

### A 的做法（derived，实现即验）

保持定形 `total_topk = WINDOW_SIZE + index_topk`（图口径不变），补 padding 掩码：

1. **放松饱和断言**：`prepare_stateful_decode_plan` 允许 `start_position ≥ WINDOW_SIZE`
   （不再要求 ≥ 2047）；`candidate_width = max(stop_position//4, index_topk)`，
   仍受 `compressed_capacity` 上界（故该 mode 的 `max_seq_len ≥ index_topk*4 = 2048`，
   即便请求短——图定形本就要固定 max_seq，2048 很小）。
2. **attention softmax 掩 padding**：topk 之后，padding 候选可由
   `compressed_indices ≥ compressed_after` 唯一识别（`compressed_after` 是
   `:1311` 的 device 标量，定形、可捕图）。在 `attention_scores`（:1352-1358）的
   压缩部分对 padding 置 `-inf`，则 `exp→0`，不进 softmax、不进输出。
   window 部分在 `pos ≥ 128` 恒满，无需掩；`pos < 128` 是更短子区间（golden 最短
   prompt 1024 > 128，不在首个 D0L 测试内），同法可掩，A 首版可先要求 `pos ≥ 128`。
3. **等价性（derived）**：非 stateful 在低饱和时 attend 的是**全部 visible 压缩 + window**；
   padded-stateful 掩 padding 后 attend 的也是**同一 visible 集**。故两者
   **应当逐位一致**（modulo §9.6 求和序，即 §7.9 那类 ratio-128-style ULP）。
   **无 kernel 改动**：decode 走 inline torch 路径（`_sparse_attention_backend` 为 None），
   A 只改该 inline 路径与 plan。**不是新 kernel group**（符合 framing 要求）。

### A 的实测结果（2026-07-22，measured）

已实现（`ratio4_attention.py`：放松饱和断言 + candidate_width pad 到 index_topk +
attention softmax 掩 padding + backend 护栏）。oracle = E7F handoff gate 在
**单 ratio-4 层（`--layers 2`）、TP4** 上跑（A 前会因饱和断言直接报错）。
artifact：`experiments/E7F-single-path-serving/results/e7f-{sat2048,unsat1024,pad12,pad1}/`。

**padding-count 扫描（arm-S 掩码-stateful vs arm-R 参考 fullpos）**：

| prefill | padding 数 | 失配 | max_abs |
|---:|---:|---:|---:|
| 2048（**饱和**） | 0 | **0/16** | **0.0** |
| 2044 | 1 | 3/16 | 7.81e-3 |
| 2000 | 12 | 14/16 | 9.77e-3 |
| 1024 | 256 | 16/16 | 1.17e-2 |

**结论（measured）**：
1. **A 让未饱和路径跑起来了**（A 前直接报饱和错），且**可捕图**——
   同跑 arm G（capture）vs arm S（stateful eager）**0/16 逐位**，padded 路径捕图精确。
2. **A 语义正确**：饱和档（0 padding）**逐位**，且差异**随 padding 数单调增长**——
   这正是"定形 reduction 加零 padding 改变 bf16 求和舍入"的 §9.6 类，
   **不是 bug**（掩码错会给大且与 padding 数无关的错，且 0 padding 不会逐位）。
   量级 1.17e-2 在 latent 分支幅值（~1–2）上约 **1 bf16 ULP**。
3. **A 非逐位** vs 非 stateful（变形 reduction）——如 §9.6 预期，释放走软门。
4. **饱和路径零回归**（0/16，与 §7.9 隔离的 attrib-L2 一致）。

⚠️ **仍待 B 解锁的一件事**：这 1.17e-2 会不会翻一个近平局 golden token，
是 **D0L golden 判据**（需 16 卡 golden gate 的 stateful 臂，卡在 Blocker B）。
即 **A 已"正确且可捕图"，其质量签字（golden token 不翻）等 B**。这是诚实边界。

### A 的释放（derived → 用 oracle 实测）

A 改的是 **stateful** 路径（golden gate 现用 **非 stateful**，故 D0L 现状不测它）。
释放判据 = **padded-stateful decode ≡ 非 stateful（reference-faithful）decode**
在未饱和位（128–2047）：
- **decode-mirror oracle**（单卡、单 ratio-4 层，仿 `e0e2e_ratio4_selfcheck` 的 check 1，
  但把 seed 位置放在**未饱和**区）：prefill 到某未饱和位 → 非 stateful decode 一臂、
  padded-stateful decode 一臂，逐步对拍。**bitwise where 求和序许可，否则 §9.6 包络内**。
- 该 oracle **单卡、无 golden、不碰 share_moe_buffers**，与 B **正交**。
- 端到端 golden-token 确认（真实 prompt 短 ctx → 图 → golden）**待 B 解锁 16 卡
  stateful 臂**后自然发生；A 的独立释放不依赖它。

⚠️ **A 是 sparse 层语义变更**：按 §9.6 求和序类，**大概率非逐位**，走"不越包络"的软门；
seed/种子敏感性沿用 §9.9 纪律（勿随手改 e0ff 种子）。

---

## Blocker B：P/D 缓冲能否共存于一套 material（A 之后、在单路操作点上实测）

### §7.10 的更正（重要）

§7.10 记的排他"prefill 与 decode 缓冲一套 material 互斥"是**在 golden gate 的
8192 档缓冲下测得**（load 后仅剩 5.61 GiB），我**把它当成了普适约束**——
**这违反了"每操作点独立"**。8192 档是 golden gate 的操作点，不是单路的。
`build_physical_stage` 每次重载 11.5 GiB 权重 → 第二套装不下，**这条只在
"想要第二套完整 material"时成立**。B 的真问题不是"两套 material"，是
**能否在一套 material 上，让大的共享 prefill 缓冲与小的独立 decode 缓冲共存**。

### B 的推导（derived）

- 单路 prefill 由真实请求定长。设该 mode 支持到 4096-ctx、chunk 4096 →
  prefill rows = 4096×TP4 = **16384**，与 golden gate 的 8192-chunk-4096 **同缓冲**。
  即 golden 的 5.61 GiB free 已是这个 prefill 缓冲下的余量（同或更宽）。
- decode 独立缓冲（B=1）：每层 4 行 × `slots_per_shape`(4) × 11 层，
  是 marlin 分组 GEMM 在 4 行下的 workspace——**KB–MB 级**，
  合计**估 < 1 GiB**（derived，需实测）。
- 故 **§7.10 的"必须重载权重 → OOM"高估了代价**：真正需要的不是第二套权重，
  是在现有 MoE 上**再注册一小块独立的 decode slot 缓冲**（各层不别名），
  与大的共享 prefill 缓冲并存。这正是 framing 说的
  "small distinct race-guard region over a large shared region"。

**decode 缓冲的量（derived，从 `moe_runtime.py:748 _register_shape` 算）**：
每个 slot 缓冲随 `global_rows` 缩放；decode `global_rows=4`（B=1×TP4），主项
`cache13 = 4 × topk(6) × max(2×local_inter(512), hidden(4096)) × 2B ≈ 192 KB`，
连同 cache2/output/gathered/combined/workspace 约 **300–400 KB/缓冲**。
每层 × `slots_per_shape`(取 4，覆盖 3 图族+余量) ≈ 1.6 MB/层 × 11 层
≈ **~18 MB/stage**。**对比 5.61 GiB 余量 = ~300× 富余。**

**⟹ B 的 derived 结论：独立 decode 缓冲 ~18 MB，远小于 5.61 GiB 余量，
几乎必然容得下。§7.10 的"OOM"全来自那个幻影的"第二套权重(11.5 GiB)"，
与 decode 缓冲无关。** 单进程双缓冲 serving 路径 derived-可行，无第二份权重、
无 P/D 拆分、无单机容量工作。

### B 要测的（measured，A 之后）

在**单路操作点**（prefill chunk 取真实请求档、decode B=1、max_seq 取该 mode）
实测：现有 MoE 注册**独立 decode slot 缓冲**（满足 `TP4DecodeStage` 的每层不别名）
后，load 后余量是否容得下？
- **若容得下** → 单进程、双缓冲 serving 路径：**无第二份权重、无 P/D 拆分、
  无单机容量工作**。B 结束，步 3 的 16 卡 golden 可跑。
- **若不够** → 记录**差多少**（实测），再考虑 buffer-lifetime 工作
  （大共享区上叠一小块独立 race-guard 区），**仍不做 P/D 拆分**。

### B 的实测结果（2026-07-22，measured）—— 已解决

实现（`e0ef2e_golden_gate.py:_build_stateful_decode_stage`）：每个 decode block 建一个
**decode-only MoE，共享 prefill MoE 的 resident 权重**（`TP4MoE(resident=pm.resident,
gate=pm.gate, global_row_shapes=(4,), slots_per_shape=4)`），只新分配 decode 小缓冲。
**未改任何 frozen 安全检查**——各 decode MoE 天然满足 `TP4DecodeStage` 的每层不别名。
护栏（`--with-stateful`+`--share-moe-buffers` 的 SystemExit）已删。

**实测（16 卡 golden gate，6 prompt smoke，1024×3+2048×3）**：
- **load 后 free 8.11 GiB**（decode MoEs 装下，无 OOM，无 alias 错）——
  §7.10 的"OOM"彻底证伪，**单进程双缓冲成立，无第二份权重**。B 解决。

### 步 3 达成（2026-07-22，measured）—— 第一条真实 prompt → 图 decode

**A+B 合起来，专业版 16 卡、单路操作点上，真实 prompt 经 prefill → 交接 →
stateful 图族 decode → golden token 的完整路径跑通了。** 这是 §10 Phase 1 的目标那一步。
artifact：`experiments/E7F-single-path-serving/results/step3-smoke/`。

**质量（stateful serving 路径 vs eager 非 stateful 基线，同 6 prompt）**：

| | 分数 | 近平局包络（自身失配最大 gap） |
|---|---|---|
| eager（非 stateful，基线） | **370/384** | 0.885635 |
| **stateful（serving 路径）** | **366/384** | **1.127426** |

**字面按 §1.3：两条都"FAIL"**（366 < 370；1.127 > 包络常数 0.959503）。
**但要看清是什么性质的失配**（否则会误判）：
1. **包络那个 1.127 是共享失配**：在 (prompt2 len1024 step36)，
   **eager 与 stateful 预测同一个错 token 8842**（都漏了 golden 59819），
   stateful 只是把 gap 从 0.841 **加宽**到 1.127（top1 36.96→37.23，ULP 经全栈放大）。
   **不是 stateful 新造了一个自信的错判**，是同一个失配、gap 值被扰动。
2. **5 个真回归**（eager 对、stateful 错）**全部落在近平局区**（gap < 0.959）：
   0.538 / 0.465 / 0.403 / 0.035 / 0.007——正是 §1.3 说的"换一组 prompt 就可能翻号"
   的 ULP 翻转类。saturated(2048) 与 unsaturated(1024,A 路径) 各有份额，
   即 §7.9 的 ratio-128 ULP 与 A 的 padding 求和序都会翻近平局。

**⟹ 与 E6F 的 −3 同性质**：字面 FAIL，但失配是 ULP 近平局翻转 + 一个 gap 被加宽的
共享失配，**不是系统性质量回归**。**定性判断（噪声 vs 回归）需**：
(a) 全集（含 8192）跑完拿完整分数/包络；(b) E6F 方法学（独立重抽集 + 符号检验）；
(c) 与 E6F 同构，**最终放行是人的门基准迁移决定**，不在无人值守下作。

⚠️ **诚实边界**：步 3 的**路径**已达成（跑通、出 golden）；步 3 的**质量签字**
（serving 路径过 D0L）**未过字面门**，需上面 (a)(b)(c)。这与"E6F 已放行"是同一类
待人裁定的迁移，不是自动 PASS。

### 明确 out-of-scope（现在不做）

- **P/D 分离（1P+nD）**：任何 P/D 拆分要求至少一台机独立扛 43 层，
  把 E3F 的单机容量缺口变成**前置**——与"尽快拿到可用 serving 路径"相反。
- **单机标准版容量**。
两者都推迟。**目标只是：专业版 16 卡、单路操作点、第一条真实 prompt → 图 decode。**

---

## 执行顺序

1. **A 先**（独立、便宜、单卡 oracle 释放）：改 ratio-4 inline decode + plan，
   加 padding 掩码，放松饱和断言；decode-mirror oracle 在 128–2047 释放。
2. **B 后**（A 之后、单路操作点实测）：在现有 MoE 上注册独立 decode 缓冲，
   实测能否与共享 prefill 缓冲共存；容得下则单进程双缓冲，步 3 可跑。
3. A 降低 B：A 给短请求可捕图路径后，B 才在真实单路 prefill 尺度上有意义地被问。
