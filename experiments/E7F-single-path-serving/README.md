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

**下一步是步 3**：16 卡、真实 D0L prompt、prefill → 交接 → **捕图** → decode，
判据是生成 token 与冻结 golden 一致。它同时了结上面第 1、2、3 条。
