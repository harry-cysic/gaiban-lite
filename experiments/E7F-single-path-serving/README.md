# E7F — 单路 serving 骨架（TARGET §10 Phase 1）

**状态**：进行中。本文件随竖条推进更新；结论未落盘前不得被引用。

## 动机

TARGET §10 把 Phase 1 定为"专业版单路 serving 做实"，其工作项表把"最小单路 serving"
描述为 **HTTP + tokenizer（`reference/encoding/` 现成）+ 单请求循环**，并注明
"不需要调度器、槽位回收、批处理准入"。

**这个描述隐含了一个前提，而该前提不成立**：它假定裸引擎已经能对一个真实 prompt
以 graph 速度生成 token，剩下的只是外面套一层 HTTP。实际不是。

### 本竖条开工时的实测结构（代码事实，非推断）

仓库里有两条互不相连的路径：

| 路径 | 入口 | prefill | decode | 质量证据 | 速度证据 |
|---|---|---|---|---|---|
| **真实 prompt 路径** | `e0ef2e_golden_gate.py` | 真实 prompt，`Ratio4FullPositionAttention` | **eager**（无图） | D0L **614/640** | — |
| **快路径（图）** | `e1f_full_decode_bench.py` | **无**——状态由 `build_seed_payload` 合成 | stateful CUDA graph | — | **39.2 tok/s** |

核对方式（可复核）：
- `grep -c capture_stateful_graph e0ef2e_golden_gate.py` → **0**：唯一的真实 prompt
  路径从不捕图；
- `e1f_full_decode_bench.py:365 build_seed_payload` 用 `deterministic_tensor` /
  `seed_nonzero_ratio4_state` 合成状态，`--start-position` 是**人为设定的**位置，
  没有任何 prompt 经过它。

**推论：所有质量数字出自 eager 真实 prompt 路径，所有速度数字出自合成状态图路径，
两者从未同时成立过。** 单路 serving 是第一件需要它们同时成立的事，所以 Phase 1 的
第一个工作项不是 HTTP，是 **prefill → graph-decode 的状态交接**。

⚠️ 这不是说 39.2 tok/s 有问题——它测的是 decode 步本身，口径清楚（TARGET §2 M4）。
说的是：它**还不能由一个真实请求触发**。

## 交接面有多大：只有 ratio-4

`e0ef2e_golden_gate.py:222-243` 的 lane 构造里：

- **window / ratio-128 层**：`material.new_state()` + `material.new_attention(state)`
  ——prefill 直接写进 `StaticWindowKV` / `StaticLayerKV`，**就是 decode 用的那个状态对象**；
- **ratio-4 层**：单独的 `Ratio4FullPositionAttention`（自带一套同形状的状态张量），
  因为静态 ratio-4 decode 类的 plan 路径冻结在 position ≥ 128，做不了从空状态起的 prefill。

所以需要搬运的只有 ratio-4 一种层型，而 `Ratio4FullPositionAttention.__init__`
（`ratio4_fullpos.py:213-262`）持有的字段与 `StaticRatio4KV.seed_decode_payload`
（`static_ratio4_kv.py:444`）要求的**逐个同名同形**：
`raw / compressed / indexer_kv / main_kv_state / main_score_state / index_kv_state /
index_score_state`，外加 `next_position`、`compressed_count`。

两侧的 raw 环形约定也一致：fullpos 写 `raw[:, pos % WINDOW_SIZE]`
（`ratio4_fullpos.py:880`），`_seed_decode_metadata` 按
`absolute_position.remainder(WINDOW_SIZE)` 建槽位（`static_ratio4_kv.py:549`）。

### 已有的冻结先例

`e0e2e_ratio4_selfcheck.py` 的 check 1（"decode mirror"）**已经做过这次交接并判逐位**：
fullpos prefill(16) + teacher-forced decode → `seed_decode_payload` 装入
`StaticRatio4KV` → `Ratio4TorchAttention.forward_decode_tensor` 跑接下来 16 个位置，
要求每步 branch 输出**逐位相同**（真实权重、layer 2、单卡）。

⚠️ **该先例的边界**（不可外推的部分）：
1. **早于 E6F**：跑的是 `tp_size=1` 未分片形态，而分片现在是默认；
2. **单层**，不是整 stage；
3. **非图路径**：对的是 eager 的 `forward_decode_tensor`，不是捕获的 stateful graph；
4. **无 PP**，无跨机。

故它证明的是"状态字段的布局契约成立"，**不是**"真实 prompt 能以图速度接着往下解码"。
后者是本竖条要测的。

## 计划（分级推进）

| 步 | 范围 | 判据 |
|---|---|---|
| 1 | 复跑冻结的 `e0e2e_ratio4_selfcheck` | 在今天的树上仍逐位通过 |
| 2 | 交接契约在**分片默认 + 图路径**下是否仍成立（4 卡，减层） | 逐位 |
| 3 | 16 卡真实 prompt：prefill → 装入 → 捕图 → decode，对 D0L golden | token 与冻结 golden 一致 |
| 4 | 套 tokenizer + HTTP + 单请求循环，测框架口径单路 tok/s | 与裸引擎 39.2 之比 = **实测 serving 折扣** |

第 4 步的产出是 TARGET §1.2 里**从未实测过**的那个折扣（单路一栏现记 20%，是推断值）。

## 结果

### 步 1：冻结的 `e0e2e_ratio4_selfcheck` 在今天的树上仍通过

titan065 单卡，`out-e7f-selfcheck-repro`（**未覆盖**冻结的 `out-e0e2e-selfcheck`）。
退出码 0，`decode_mirror.accepted=true`、`prefill_consistency=true`：
seed_position 192，其后 16 个位置**逐位相同**（每步 `max_abs` 0.0），
`state_bitwise` 的 raw / compressed / indexer_kv 三项均 true。
跑前跑后 8 张卡都是 1 MiB。

⚠️ **但这次复跑不覆盖今天的默认**：该脚本从不设置 `tp_size`，
`Ratio4AttentionConfig.tp_size` 的 dataclass 默认是 1，所以它跑的是**未分片**形态，
**且它的 JSON 里没有任何分片见证**——正是 TARGET §9.11 说的那种"臂在自己的
artifact 里看不见"。所以步 1 只说明冻结契约没被树的演进弄坏，
不说明分片默认下也成立。那是步 2 的事。

### 步 2：prefill 出来的状态**是**一个合法的 decode 状态（分片默认下，逐位）

`runtime/e7f_handoff_gate.py`，titan065 GPU0-3，一个 TP4 stage，
层 0/1/2/3 = window / window / **ratio4** / ratio128（三种层型都覆盖），
prefill 256 token → 交接 → 16 步 decode。
artifact：`results/handoff-shard/rank{0..3}.json`。

| | 结果 |
|---|---|
| 判据 | 16/16 步**逐位相同**，每步 `max_abs` **0.0** |
| 分片见证（解析侧） | `tp_size=4`、`local_num_heads=16`、`local_o_groups=2`、`wo_b` **[4096, 2048]** |
| 要求侧 | `requested.attention_tp_shard=true` + 完整 `argv` |
| 比较范围 | **整块链**（attention + MoE + HC），不只是 attention 分支 |

分片确实生效：未分片时是 64 头 / `wo_b [4096, 8192]`（见 E6F 的
`e1f-bl1-tp1-regression`），这里两项都是分片值。

**顺带的一致性观察**（非本竖条的新证据，只是没有矛盾）：见证显示两条臂的
ratio-4 indexer QAT 模式**不同**——arm R 的 fullpos 对象是 `indexer_qat_mode=ref`，
arm C 的 decode 块走 E4F 设为默认的 `fused`。两者仍然逐位，与 E4F 冻结的
"融合核逐位"相容。

### 步 2 的阴性对照：这个门能失败

一个"跑不失败的门"什么也没证明（TARGET §9.12 同理）。三个对照臂：

| 对照 | 结果 | 说明 |
|---|---|---|
| `ratio4-skip`（不装 ratio-4 状态） | **报错**：`decode start_pos does not match ratio-4 state` | 撞的是状态自校验，**不是比较** |
| `static-skip`（不还原 window/ratio128 状态） | **报错**：`start_pos 256 != static KV next position 0` | 同上 |
| **`perturb-compressed`**（`compressed[0,0,0] += 0.5`） | **16/16 步全部失配**（step 0 `max_abs` 2.44e-3） | **合法但错**的状态，只有比较能抓 |

前两个证明状态对象会对"根本没交接"大声失败（不是静默错答）；
第三个才是比较本身的功效检验：**一个元素的扰动就让 16 步全灭**，
所以 16/16 逐位不是"比了个寂寞"。

### 步 2 建立了什么、没建立什么

**建立**：真实权重、分片默认、三种层型、整块链下，
`prefill → seed_decode_payload → 静态 decode 路径`的状态交接是**逐位精确**的。

**未建立**（不得外推）：
1. **没有捕图**——本门只跑 eager decode 路径。图的正确性靠 E0sf（"合法状态下
   graph == eager"）**组合**得到，不是本门直接测的。这是刻意的：一旦失败，
   失败点不含糊。
2. **不是真实 prompt**——prefill 喂的是确定性 residual 流 + 随机 token id，
   走的是状态机制，不是 embed/head 与 prompt 语义。golden token 对拍是步 3。
3. **单 stage、无 PP、无跨机**；
4. **prefill 只有 256**，没测 4096/8192 档与 chunked prefill；
5. 未跑 `--no-attention-tp-shard` 对照臂（分片是已放行默认，故先测默认）。

### 步 2.5：捕图不是问题；但**图路径只在 ctx ≥ 2047 存在**

给门加了图臂（capture + replay）后，第一次跑（prefill 256）**硬失败**：

```
ValueError: stateful ratio-4 decode requires a saturated fixed index top-k
```

出处 `ratio4_attention.py:956`：
`minimum_candidates = (start_position + 1) // COMPRESS_RATIO`，要求 `≥ cfg.index_topk`。
Flash 的 `index_topk = 512`（config.json）、`COMPRESS_RATIO = 4`，
故 **start_position ≥ 2047**。

这正是 E1F 的 `start_position < 2047` 断言（`e1f_full_decode_bench.py:653`），
其 docstring 第 49 行也早写明"ratio-4 index saturation needs >= 2047"。
**约束本身不是新的**；新的是它对交付的含义（见下"对 Phase 1 的影响"）。

⚠️ 机制：图要求固定形状，而 top-k 的候选数在饱和前随位置变化
（非 stateful 路径用 `index_topk_count = min(cfg.index_topk, compressed_after)`
吸收了这个变化，`ratio4_attention.py:873`）。三个图族里**没有**未饱和族。

### 步 2.5 结果（prefill 2048、max_seq 3072、4 卡、分片默认）

artifact：`results/graph-attrib/rank{0..3}.json`（`results/graph-2048/` 是加归因臂前的同配置跑）。

四条臂，同一个交接后的状态：

| 比较 | 结果 |
|---|---|
| arm C（**非** stateful decode）vs arm R（prefill lane 续跑） | **16/16 逐位** |
| arm S（stateful eager）vs arm R | **12/16**，失配位 2052 / 2058 / 2060 / 2062，`max_abs` **1.46e-3** |
| arm G（**捕图** replay）vs arm R | 失配位**与 arm S 完全相同** |
| **arm G vs arm S（E0sf 判据：同实现）** | **16/16 逐位，`max_abs` 0.0** |

**归因结论：那 4 个失配与捕图无关、也与交接无关。**
arm G 和 arm S 逐位相同，说明捕图对一个 **prefill 来的**状态与对合成状态一样精确
（E0sf 的结论在新 provenance 下成立）；失配整个来自
**stateful 与非 stateful 两条 decode 实现之间**的差异——两臂吃的是同一个状态。

⚠️ **该差异的机制未归因，不要猜**。我最初猜是融合 HC 边界，
**这个猜测是错的**：本次 `--hc-backend default`，而
`resolve_hc_boundary_backend("default")` 返回 **None**（`hc_boundary_backend.py:289`），
即根本没有融合链，两臂都走逐块路径。所以差异在
`block.forward_decode_tensor` 与 `block.forward_stateful_decode_tensor` 之间，
**具体是哪个算子的求和序，尚未定位**。
量级 1.46e-3 在 bf16 上约 1 ULP（幅值 ~0.2–0.4 处），与"同数学、异求和序"相容，
但**相容不等于已证**。

**从 artifact 重读出的 family 模式（实测，纯分析、无新跑）**：4 个失配位
2052/2058/2060/2062 **全是 `normal` family**；而全部 4 个 `ratio4_boundary` 位
（2051/2055/2059/2063）**逐位相等（0.0）**；normal 位内部**间歇失配**（8 中 4）。
**读法**：ratio-4 的压缩/finalize（求和序确定）两条实现吻合，
漂移只在 normal 位、数据相关地间歇出现——**符合** §9.6"改变行序不可能逐位"
（每位稀疏 attention 累加序在两实现间可不同）。
⚠️ family 分布是**实测**；"差在稀疏 attention 累加序"当时是**相符的假设**。

### 层型隔离：差异全在 ratio-128（实测，更正了上面的假设）

用现成的 `--layers` 单层型各跑一遍（无新代码，`results/attrib-L{2,3,0}/`）：

| 隔离 | arm-S vs 参考 |
|---|---|
| 单 ratio-4（`--layers 2`） | **0/16 逐位** |
| 单 window（`--layers 0`） | **0/16 逐位** |
| **单 ratio-128（`--layers 3`）** | **11/16 失配**，`max_abs` 1.95e-3~3.91e-3 |

**整栈的 stateful≠非 stateful 全部来自 ratio-128 层**——
ratio-4 与 window 的 stateful/非 stateful 两条实现逐位一致。
⚠️ **这更正了上面按 family 做的猜测**：先前"ratio4_boundary 逐位、normal 间歇"
被读成 ratio-4 稀疏 attention 的求和序；其实 family 是**调度节拍**（ratio-4 的压缩
周期），与"哪层出错"正交。隔离把它钉在 **ratio-128**。
**又一次"先断言后被实验更正"**——正是 goal 文档要求归因在主会话做、
用隔离实验而非 family 花样去定位的理由。

⚠️ 仍未到**算子级**（ratio-128 层内是 attention-branch 还是 moe / 哪个 reduce），
但层型已定。机制属 §9.6（sparse 层每位选择/累加求和序在两实现间可不同，非 bug）。
对步 3 的意义不变。

### 一个此前没人比过的对子

据我检索，**stateful 与非 stateful decode 从未被直接对拍过**。逐个核对过的近邻：

| 门 | 它比的是什么 | 是否覆盖本对子 |
|---|---|---|
| `e0sf` 部分 (a) | `forward_decode_tensors`（superstage）vs 手工 `forward_decode_tensor` 链 | ❌ **两边都是非 stateful** |
| `e0sf` 部分 (b) | stateful graph vs stateful eager | ❌ 两边都是 stateful |
| `e0e2e_ratio4_selfcheck` | fullpos vs **非** stateful `forward_decode_tensor` | ❌ 两边都是非 stateful |
| `e0kf` 系列 | fp8-KV 配对门 | ❌ |

⚠️ `e0sf` (a) 的措辞（"superstage adds no arithmetic, so its gate is also bitwise"）
容易被读成已经覆盖——**它没有**：那是 superstage 组合 vs 手工链，
两边都走非 stateful 实现。本门是第一次把 stateful 与非 stateful 接在一起比。

**这不推翻任何冻结数字**——E1F 的 bitwise 132/132 比的正是 graph vs stateful eager，
本门在该对子上也拿到 0.0。但它有一个对 Phase 1 要紧的推论：
**D0L 的质量证据跑在非 stateful 路径上（`e0ef2e`），而 serving 要跑 stateful 图路径，
两者不是同一份代码，且已实测不逐位。**
所以步 3"真实 prompt 经图路径对 golden token"**不是走过场，是必需的**。

### 试到 11 层（真实 stage 0 形状）：本门自己装不下

想把步 2 推到生产 stage 形状（层 0-10 = 16 卡配置里的 stage 0，prefill 2048），
**OOM**：`23.18 GiB in use`，还差 384 MiB。

⚠️ **第一次跑这个配置的读数是污染的**：开跑前 4 张卡各已占 15,375 MiB
（上一轮的残留进程还在），那次 OOM 不算数。清干净后**重跑**，
precheck 读到 4 MiB 总量，仍然 OOM——所以这是真的，不是残留。
（HANDOFF §3.8 的老教训：**每次开跑前确认 1 MiB**。我打了快照却没让脚本
据此中止，所以第一次白跑了一轮；现在的命令里加了 precheck 会 `exit 9`。）

**这是本门的开销，不是 runtime 的容量结论**：门为了做对照，同时持有
prefill lane 的状态、decode 侧的状态、以及两份快照。
⚠️ **各项占比未归因**——11 层权重约 11.51 GiB（E3F）、8192 行的 MoE prefill
缓冲约 3.06 GiB（§3.8），合计约 14.6，与实测 23.18 差约 8.6 GiB。
这 8.6 里 prefill 工作区与门自己的多份状态各占多少，**没测，别猜**。

**对结论没有影响**：交接契约是**逐层**的，层数不改变它；步 2 的 4 层配置已覆盖
全部三种层型。真正需要 11 层的是步 3，而步 3 只需**一份**状态（真实 serving 形态），
不背这个门的对照开销。

## 对 Phase 1 的影响（TARGET §10）

1. **已放行的单路 39.2 tok/s 是 ctx ≥ 2048 的数字**——E1F 的
   `--start-position 2048` 是饱和阈值，不是随手选的。TARGET §2 的 M4 实测列
   记了"16 卡口径、分片默认后、p50 25.49 ms、bitwise 132/132"，**没有记上下文长度**。
2. **短 prompt 请求今天没有快路径**。ctx < 2047 时只有非 stateful 的 eager 路径，
   而 TARGET §4.3 的冻结事实是 B=1 eager **210 ms/步** vs graph 36.3 ms。
   一个 200 token prompt 生成 100 token，全部落在未饱和区。
3. 因此 §10 Phase 1 工作项表把"最小单路 serving"写成 HTTP + tokenizer + 请求循环，
   **少了一项前置**：未饱和区的图路径（或等价的快路径）。
   候选解法（均未做、未评估）：把候选集 pad 到 `index_topk`（须先证与 reference
   语义一致）；为未饱和区单独立一个图族；或接受短 prompt 走 eager（按上面的算术，
   这等于放弃短 prompt 的单路指标）。

## 步 3 尝试：撞上 prefill/decode 引擎分离，一个物理约束

**目标**：16 卡、真实 D0L prompt（≥2047）、prefill → 交接 → stateful decode，
判据是 D0L（分数不降 + 近平局包络不越）。做法是给 `e0ef2e_golden_gate.py` 加一个
默认关闭的 `--with-stateful` 臂（复用 eager 臂的 prefill，交接后跑 serving 的 decode
路径）。**基线已备**：eligible 子集（≥2047，7 条）冻结非 stateful = **427/448**，
近平局包络 **0.9558830261**（≤ §1.3 常数 0.959503）。

**结果：跑不起来，撞上一个真实的架构约束，不是 bug。**

`--with-stateful` 的 decode 臂要建 `TP4DecodeStage`，其构造器强制
**每层的 MoE slot 缓冲互不别名**（`superstage.py:330 _require_unique`），
这样捕图时各层 MoE 状态独立、不会 race。**而 golden gate 必须带
`--share-moe-buffers`**——8192 档 prefill 的 MoE 缓冲若不跨层共享就装不下
（v2chunk 冻结口径 load 后只剩 **5.61 GiB**，且每个冻结档都开了 share）。
共享让各层 MoE 缓冲**别名**，正好违反 decode stage 的唯一性要求。

**两者在一套 material 里互斥**：
- prefill 要 `share_moe_buffers=True`（大缓冲跨层共享才装得下）；
- stateful decode 要 `share_moe_buffers=False`（各层缓冲独立，图不 race）。

想建两套 material（prefill 一套 + decode 一套）共享权重也不行——
`build_physical_stage` 每次重载权重（11.5 GiB/卡），第二套装不进剩下的 5.61 GiB。

⚠️ **这不是可以绕过去的工程细节，是 serving 的真实结构**：
**prefill 与 decode 是两个共享权重的独立引擎**。当前 runtime 在**一个进程里
只支持一套 material**，无法同时表达"prefill 引擎（大共享缓冲）"与
"decode 引擎（小独立缓冲）"。这正是 §10 Phase 1 工作项"prefill → 图 decode 交接"
底下没写出来的那一层：交接的两端是两个**引擎**，不是两个 material。

### 第一次尝试怎么暴露的（操作记录）

第一跑：earth ProxyJump 在 load 后掉线，远端 torchrun 被孤儿化并在 collective 里
自旋（100% util / **~95 W @2700 MHz** = §3.8 的"自旋非计算"）。**教训**：
`ssh | tee` 前台管道一断，远端作业就孤儿化挂死。已把 launcher 改成
**setsid 脱离 + 远端日志文件 + DONE 哨兵轮询**（`run_e7f_golden_stateful.sh`），
掉线不再孤儿化，轮询按**产物**（哨兵文件）而非进程存活（§9.12）。

第二跑（robust launcher，earth 稳定）：真错误浮现——
`ValueError: super-stage MoE slot buffers must not alias across layers`，
即上面的约束。**但它以 pipeline 死锁的形式出现**：stage 0 在 `lane.forward`
之前就 raise（我的 dispatch 把 `_build_stateful_decode_stage` 放在 forward 前），
没 send，下游 stage 卡在 `pair_transfer` recv → 死锁自旋。
已加**早退护栏**（`e0ef2e_golden_gate.py`，arg 解析后、任何 collective 之前）：
`--with-stateful` 与 `--share-moe-buffers` 同开即 `SystemExit`，
四路对称退出，**再不会死锁**。

### 步 3 的正确设计（下一个 session）

需要 runtime 支持**两套共享权重的 material**（prefill 引擎 + decode 引擎），
二选一：

1. **`build_physical_stage` 支持权重复用**：先建 prefill material（share=True，
   大缓冲），再建 decode material 时**传入已加载的 resident 权重**、只新分配
   decode 的小独立缓冲（share=False，仅 decode row-shape、`slots_per_shape≥4`）。
   这是 serving 的真实形态，也是最干净的。
2. **`TP4DecodeStage` 加 `eager_only` 模式**：跳过 MoE-slot 唯一性检查
   （eager 顺序 decode 用共享缓冲是安全的——eager StageLane 就是这么跑的），
   并让 stateful plan 用 `moe_slot=0`（`graph_moe_slots=(0,0,0)`）。
   更小，但改了两处 frozen 安全检查，**且步 3 的 golden 结论会依赖这两处放松
   本身正确**——一旦放松有 subtle 错，golden 结论就不可信。故**不在无人值守下做**。

⚠️ 两条都触碰 runtime；在有人复核前不改。**`--with-stateful` 的脚手架保留**
（arg、handoff 复用、per-prompt skip、护栏都在），换上任一支持后即可直接跑，
基线 427/448 与包络 0.9558830261 已备好。

### 一个顺带落地的正结果：eager 臂逐字复现冻结基线

死在 stateful 臂之前，**eager（非 stateful）臂已在 smoke 子集上逐字复现冻结基线**
（artifact：`results/step3-blocked/`）：

| prompt | 长度 | 本次 eager | 冻结基线 |
|---|---|---|---|
| 0（原 p3） | 2048 | **63/64** | 63 ✓ |
| 1（原 p4） | 2048 | **60/64** | 60 ✓ |
| 2（原 p5） | 2048 | **60/64** | 60 ✓ |

183/192，逐字节相同。这**证明两件事**：(1) `--with-stateful` 脚手架**没有扰动
冻结路径**（改动对默认臂安全）；(2) 配置（分片默认、chunk 4096、share-moe-buffers、
tilelang prefill）是对的。所以步 3 卡住**只**卡在引擎分离，不在我的接线或配置。
这也满足了 goal 文档的"引用他人结论前先复跑"——冻结的 427/448 里的这 3 条已复跑对上。

### 仍然成立的（步 1/2/2.5 不受影响）

交接**逐位**（步 2）、捕图对 prefill 来的状态**精确**（步 2.5）、
图路径**只在 ctx ≥ 2047 存在**（§7.8）、stateful 与非 stateful **不逐位**（§7.9）——
这些都在 4 卡上独立成立，不依赖 16 卡 golden。步 3 卡住的是"把 golden 判据搬到
stateful 路径上"这**一件**事，而它卡在 runtime 的引擎分离，不在语义。
