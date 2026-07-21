# HANDOFF（2026-07-21，goal session 更新）

面向继任者。**权威目标定义是 [`docs/TARGET-v4-flash.md`](docs/TARGET-v4-flash.md)**，
本文不重复它已写的东西（实测值、已证伪假设、硬规则、环境坑）。本文只写三件它没有的：
**手上未完成的事**、**我原本打算的下一步**、**尚未写进任何 README 的隐性知识**。

---

## 1. 未完成的事

### 1.1 交付状态

本轮 goal session 完成 E2F（B=1 decode profile）、E3F（8 卡容量判决），并在做
E4F（decode 侧融合 indexer QAT 核）。E2F/E3F 已完整落盘并 commit；**E4F 进行中**，
见 §1.3。

E2F/E3F 的两个结论改了 TARGET：335 tok/s 单用户天花板被证伪（真值 76.2，因
DP-attention 复制 attention 权重），8 卡 TP4×PP2 装不下 43 层（M7 证伪、M4 回 16 卡）。

### 1.3 E4F 进行中的状态

- `runtime/dsv4_direct/ratio4_attention.py` 已加 `indexer_qat_mode`（默认 `ref`，
  env `DSV4_INDEXER_QAT_DECODE` 可覆盖），两个 decode 调用点走 `_indexer_qat()`。
  **默认关闭，未改变任何既有语义。**
- `runtime/e2f_decode_phase_probe.py` 已加 `--mode ab`：两条常驻 lane 背靠背
  交替重放（TARGET §9.1 口径），顺带每步逐位对拍作为层内数值门。
- E4F 微基准已完成并落盘：decode 形状下融合核**逐位相等**（max_abs_diff 0.0），
  省 **55.2 µs/次调用**，轮间离散 <1%。
- **未完成**：层内 A/B 尚未跑出数字。跑通前不得引用任何层内收益。

### 1.2 挂着的账（按我认为的紧要程度）

1. **C3F↔C2F 矛盾读数的成因未坐实**。chunked prefill 的"快 1.5–2.3×"已被 C2F 同口径
   证伪（证伪是实测），但**为何 C3F 会读到相反结果，目前仍是推断**（疑为整段臂在
   0.6 GiB 余量下的分配器压力）。补一个对照臂即可了结：C3F 的 E2E 口径 + 整段 +
   只注册 1 种形状 + 充足 free，一次运行。
   ⚠️ 本仓库已被同一形状的"分配器假说"坑过一次（第二十二竖条花整条竖条证伪它，
   真因是 NCCL 静默回退 SHM）。**在补测前不要把这个假说当已知**。
2. **chunked prefill 没有自有 golden**，因此只能作为 Phase 4 能力存在，不能默认开启。
   现状：语义已证精确（状态机逐位 27/27、索引集精确 18/18），但 chunk=512/999 在
   D0L 上越出 `top2_gap` 包络。需要一个"分段 vs 整段"的专用 golden 才能放行默认开。
3. **两个已否决杠杆的复活条件已写明但未做**：prefill HC 融合（+19.9%，卡在 vLLM
   kernel ≥1024 行数值错误，分派点见 TARGET §7.4）；tilelang 稀疏核包装层 16.1%
   搬运税（去掉不需要新 kernel，只需出口按 head chunk 分块不拼回）。
4. **`run_e0mf_titan.sh` 故意不带 `NCCL_P2P_LEVEL=SYS`**——它的冻结产物就是在无 P2P
   下产生的，加上会与冻结值对不上。若要统一，必须同时重生成其冻结产物。

---

## 2. 我原本打算的下一步

按**旧**优先级（新 TARGET 文档到达前）我的下一步是继续 prefill：稀疏核包装税 →
chunked prefill 自有 golden → serving。

**但按新 TARGET 文档我不会这么排。** 它的 §2 空白格优先级把 M4（延迟模式，差 8–14×，
且 8 卡形态从未跑过）和 M5（长上下文，零覆盖）放在 prefill 继续优化之前，这个判断
我认同，理由是：prefill 已达内部计划的 72% 且路径清晰，而 M4/M5 是**零覆盖**且
M4 直接卡对外承诺；另外 8 卡形态同时卡住 M4、M7 与所有单机口径的计划值，越晚验越贵。

TARGET §7.1 的判断也对：**M4 的第一步是 profile 而不是优化**。理由在本 session 已被
C4F 反复证实——我给 C4F 的任务书里预设的四个候选杠杆，profile 出来的头号项（indexer
QAT 链，41.27%）**一个都不在里面**，而我预设的"wo_a einsum 退化"和"投影 GEMM 有余量"
两条都被证据判死（投影已在 BF16 峰值 96–100%）。**不要跳过 profile。**

---

## 3. 隐性知识（没写进任何 README，但会决定结论真假）

### 3.0 本轮新增的方法学（E2F/E3F/E4F）

1. **图内相位计时要 `external=True`**。`torch.cuda.Event(enable_timing=True)`
   在 capture 期间能被录成 event-record 节点，但之后 `elapsed_time` 返回
   `cudaErrorInvalidValue`；`torch.cuda.Event(..., external=True)` 才可查询。
   已封装为 `dsv4_direct/phase_timer.py:GraphPhaseRecorder`。
   **标记不是免费的**：每个是图里一个真实节点，244 标记 = +14.8%，47 标记 = +2.0%
   （约 3.3–4.6 µs/标记）。**相位表只做定性分相，定量归因用 nsys。**
2. **nsys 必须带 `--cuda-graph-trace=node`**，否则一次 graph replay 在报告里是
   一个不透明 range。配合探针的 `--cuda-profiler-range`（cudaProfilerStart/Stop）
   把 load/warmup/capture 排除在报告外。
3. **被 trace 的 rank 会把自己的减速通过集合通信传给同组其他 rank**：4 个 rank
   全读到 8.23 ms（未 trace 7.62）。nsys 运行的**绝对 ms 不可用，份额可用**。
4. **不用标记也能分段**：一次 replay 内 kernel 序列固定，MoE 集合通信是天然分隔符
   （`ReduceScatter` 每层一次；层内第一个 `AllGather` 分 attention/MoE）。
   见 `experiments/E2F-*/analyze_regions.py`——**先对层数校验计数再报数**，
   否则 MoE 结构一变就会静默错分。
5. **MoE 对象在 lane 之间是共享的**（`physical_stage` 只给每条 lane 自己的
   state/attention）。所以**两条 graph lane 不能都用 slots 1-3**——第二次 capture
   会以 `slot is not clean` 失败。做双 lane A/B 必须给变体 lane 独立 slot 区间
   （探针用 5-7，`slots_per_shape=8`），**且变体 lane 的 warmup 也必须用它自己的
   slot**，否则会把基线 lane 的 slot 弄脏。这一条咬了两次。
6. **OOM 信息里的"另一个进程"可能是上一次跑挂的残留**。E2F 曾把一次残留导致的
   OOM 误读为自己的内存回归；`nvidia-smi --query-compute-apps` 先看一眼再改代码。
   清理用 `pkill -f "<pattern 带方括号>"`（§3.3.2）。
7. **env 开关经 ssh 传远端时会静默失效**。`run_*.sh` 里 `ENV_BASE` 是**单引号**
   字符串，往里插 `${VAR:-:}` 不会在本地展开——它作为字面量传到远端，而远端没有
   该变量，于是展开成 `:`（空操作）。**症状是"杠杆一点效果都没有"，与"杠杆无效"
   完全同形**。E4F 因此差点把一个实测 −3.82% 的融合读成 0%。两条要求：
   - 插入点必须让**本地**展开：`ENV_BASE='...'"; ${VAR:-:}"`；
   - **结果 JSON 里要有处理组见证**。已给 `e1f_full_decode_bench.py` 加
     `attention_modes`（记录每层实际解析出的 `indexer_qat_mode` 等），
     没生效的开关在 artifact 里看得见，而不是被读成阴性结果。
   这与 §3.0.6、TARGET §9.10 是同一类教训：**静默失效的机制比崩溃危险**。
8. `runtime/dsv4_direct/__pycache__/*.pyc` **是被 Git 跟踪的**（历史遗留，21 个）。
   改 `dsv4_direct` 下的文件后 `git add -A` 会带进 .pyc 噪声；本轮 commit 都是
   显式加路径避开的。

### 3.1 测量方法学

1. **4090 上串行 A/B 计时不可信**。同配置三轮可给出 51/219/318 µs 三个完全不同的
   差值（时钟/热漂移）。所有 graph-replay 级 A/B 必须用**成对交替重放**：两条 lane
   常驻、每步背靠背、逐步交替顺序。MTP 与 HC 的可信数字全出自这个口径。
2. **per-component `synchronize` 计时的开销随 forward 次数增长**（1024 段长的臂上
   +14.7%）。`component_walls` **不能跨 forward 次数不同的臂相减**；headline 一律用
   未插桩 p50。C4F 的分相探针是反例做法：插桩开销 0.13%、相位覆盖 100.02%，因为它
   用 CUDA event 而非 host sync。
3. **一个组件的桶会吸收整条链的跨 rank 偏斜**。C2F 的 MoE 桶曾被记成 131 ms/层，
   隔离测只有 49.8——因为 MoE 是链上唯一含集合通信处。**看到某个桶异常大时，先问
   "它是不是链上唯一的同步点"，再怀疑该组件本身。**
4. **微基准 ≠ 层内**。C4F 的 cast 提取微基准预期 0.65 ms/层，层内只兑现 0.166 ms
   （已被分配器复用/与 GEMM 重叠）。反向也成立：A5F 的 HC 微基准 2.92× 在集成态
   只回收 4.7 ms/stage。**微基准只用于判活/判死，不用于折算收益。**
5. **≥3 轮 + 报轮间离散**。已放行数字的离散都在 0.03–0.29%；超过 1% 说明口径里有
   未控变量（经验上：NCCL 传输选择、分配器状态、或插桩本身）。

### 3.2 质量门的实际判据

1. **`top2_gap` 包络比分数更硬**。D0L 基线最大近平局间隙 **0.9595**；分数 ±3 在
   "换一组 prompt 就可能翻号"的范围内，而**越出包络是质量性变化**。prefill HC 融合
   正是因越包络（1.4929）被否；chunked prefill 的 chunk=512/999 同理。
   **放行判据 = 分数不降 且 gap 不越包络。**
2. **Marlin 分组 GEMM 对 batch 组成是 1–2 ULP 敏感的**（行分组变了求和序就变）。
   这是固有性质不是 bug。凡改变行集合/行序的改动——失步投机、集合重叠的行分块、
   chunked prefill——**都不可能逐位，别浪费时间追**。
3. **跨 TP lane 的输出本来就不逐位**（reduce_scatter 求和序），规范输出取 tp_rank0。
   种子残留态下各 lane 的 argmax 会分叉、真实 prompt 下则一致——**用种子态做
   "lane 一致性"检查会得到假失败**。
4. **exact-topk 门对 E2M1 格点翻转敏感**：两条独立公式在 1 ulp 差处会翻转一个量化码，
   进而换掉一个近平局 top-k 条目。E0ff 的种子 **20260717** 是筛出来的"全部抽样落在
   格点上"的种子，**不要随手改**。
5. **HC 类数值门只在真实权重上有意义**：真实 `hc_scale` 是 0.03–0.20，会压制 GEMM
   误差向 post/comb 的传播；合成 `hc_scale=1` 放大约 4×。A5F 的"≤1e-5"就是这么来的假象。

### 3.3 集群与工具

1. **16 rank 跑挂后集合通信会楔死**，需在两台机 `pkill -f <bench>`；TORCH_NCCL 超时
   是 120 分钟，等不起。
2. **`pgrep` 自匹配**：ssh 起的 `bash -c` 里若内嵌 pgrep 的模式串，pgrep 会匹配到
   自己的父 shell，给出假的"有残留进程"。独立执行才准。
3. **Bash 工具里的 `cd` 跨调用保持**，曾导致 `git add` 路径落空。用绝对路径。
4. **长 commit message 里的引号会破坏 shell 引用**，用 `git commit -F <file>`。
5. **单卡微基准优先放 titan065**，避免与 titan064 上的 16 rank 作业抢卡。
6. 热 page cache 下 11 层 stage 加载约 11–12 s；首次加载慢很多。
7. `runtime/` 不是安装包：一律 rsync 到 `titan064:~/e0f-runtime/`（部分微基准在
   `titan065:~/a5f/`），靠 PYTHONPATH 串起来。

### 3.4 子 agent 协作（若继续用多 agent）

1. **agent 会在"等监视器"处提前停下**而不是跑完（本 session 发生 2 次）。症状是报告
   里写 "waiting for X to finish"。解法：SendMessage 明确要求**同步执行到完成，
   不要用等待/监视器方式停下**。
2. **完成后仍会持续收到滞留的监视器通知**，内容是旧结论的复述。不要据此重做工作。
3. **数字必须从拉回本地的 per-rank JSON 复核**。本 session 出现过与实际不符的完成
   通知。另：`e0ef2e` 的 `result.json` 是人工挑选的字段子集，新加字段会被静默丢弃，
   分析须以 per-rank JSON 为源。
4. **归因、复测、跨实验对账必须在主会话做**。子 agent 只看得到自己那一片，本 session
   有两次是主会话对账才发现问题：C2F 的 131 ms 归因错误、以及我自己把"推断"写成
   "实测"（由另一个 agent 复核时抓出来）。

### 3.5 门脚本命名对照

| 脚本 | 覆盖 |
|---|---|
| `e0ef` / `e0ff` / `e0wf` | ratio-128 / ratio-4 / 窗口层 attention oracle |
| `e0cf` / `e0df` / `e0sf` | TP4 MoE / 整层 block / superstage + stateful graph |
| `e0pf` / `e0qf` | 单机 TP4×PP2 / 双机跨机管线 |
| `e0ef2e` | E2E golden（短门；`--prefill-chunk` 为 D0L 长门臂） |
| `e0kf` / `e0mf` | fp8-KV ratio-4 配对门 / MTP block oracle |
| `e1f` / `e1if` | 满配吞吐 B 扫描 / 4-microbatch 交织 |
| `c2f_*` / `c4f_*` | prefill stage bench 与各类探针、分相 profile |

---

## 4. 一句话总结

语义链条完整且每一环都有真实权重的门背书。**当前最高价值的单项工作是
attention TP4 分片**，因为三件事同时压在它上面：

1. **单路速度**——E2F 实测当前形态（DP-attention 复制 attention 权重）的
   带宽天花板只有 76.2 tok/s，对外承诺的 ≥150 在这个形态下**不可达**；
2. **8 卡容量**——E3F 实测当前形态连权重都装不下（22 层 stage 载到第 19 层
   OOM），而"装得下"的那个字节账（余 1.13 GiB）**本身就是分片之后才有的**；
3. **标准版五行**——客户口径的五行不要求同时满足、每行可用各自配置，
   所以 8 卡是**五个独立的容量问题**（TARGET §7.7），而分片落地之前
   一行都无法有意义地重问。

⚠️ **注意 E3F 结论的范围**：证伪的是"8 卡跑吞吐档位"（M7 的 10–16K 聚合），
**不是"8 卡装不下模型"**。本仓库初稿曾写成后者——那会让 8 卡以"已关闭问题"
的身份进 TARGET §5、后续不再回头重问，而它实际上是个**还没被正确提问过的
问题**。已于 2026-07-21 收敛，见 E3F §5 的撤回说明。

M4 的 200–350 还需要 elementwise 尾巴折叠同时到位（E4F 已取下第一块 +2.31%，
逐位）；两处缺一不可，任一单独完成都停在 ~80 tok/s（E2F）。
