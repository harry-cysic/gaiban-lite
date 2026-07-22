# E8F — 最小单路 serving 外壳（TARGET §10 Phase 1）

**状态**：第一条 serving 通路已跑通并测出**首个框架口径单路数字**。
设计见 [`docs/design-serving-shell.md`](../../docs/design-serving-shell.md)。

## 做了什么

把两半接成一个**常驻 16 卡请求循环**（`runtime/e8f_serving_loop.py`，
launcher `run_e8f_serving.sh`）：
- **prefill + 交接**：e0ef2e 的 `StageLane` 真实 prefill（chunked、eager）+ E7F 交接
  （decode-only MoE 共享 prefill resident 权重，Blocker B）。
- **decode 引擎**：e1f 的 free-running 图闭环（stage3 head → argmax → NCCL loopback →
  stage0 重嵌 → 图 replay），**一图族一次捕获、位置无关、跨请求复用**（plan 建一次、
  每请求 `cursor.reset` 重置——e1f restore_cycle 口径，故一次一个定长 bucket）。
- 新增：真实 prefill、free-running（定长 max_new_tokens，EOS 后置）、请求循环、
  device 侧 token 累加（无逐步 host read/object broadcast）、per-request 框架计时。

## 实测（fused HC，每形状 5 个测量请求，round-0 捕获排除）

artifact：`results/b1024-fused-rank0.json`、`results/b2048-fused-rank0.json`。

| 形状 | decode ms/tok | decode tok/s | prefill ms | **首 token ms** | **框架 tok/s** | **折扣** |
|---|---:|---:|---:|---:|---:|---:|
| 1024-prompt / 48-tok | **24.80** | 40.33 | 183 | **687** | **25.92** | **34%** |
| 2048-prompt / 48-tok | **24.81** | 40.30 | 342 | **1336** | **19.18** | **51%** |

裸引擎参照：**39.2 tok/s = 25.49 ms/tok**（E1F @ctx2048）。

**生成长度摊薄曲线（1024-prompt，首 token 固定 687 ms）**：

| gen_len | 框架 tok/s | 折扣 | 占 decode 满速(40.3) |
|---:|---:|---:|---:|
| 48 | 25.92 | 34% | 64% |
| 128 | 33.33 | **15%** | 83% |

生成越长，首 token 延迟被摊得越薄，框架 tok/s 趋近 decode 满速 40.3——
折扣 → 0。**§1.2 的 20% 大致对应"短 prompt + ~128 生成"的工作点。**

**这些点合起来说清了三件事**：
- **decode 在两个 ctx 完全一致**（24.80 / 24.81 ms/tok = 40.3 tok/s），
  **与裸引擎打平/略优**——serving 图 decode 路径**满速、与 ctx 无关**。
- **首 token 延迟随 prompt 长度缩放**（687 → 1336 ms，≈2×，与 prefill 工作量一致）——
  这是框架开销的**唯一大项**（decode 无损）。
- **折扣随 prompt 长度增大**（34% → 51%，固定 48 生成 token）：首 token 延迟越大、
  摊到同样 48 个 token 上折扣越深。**折扣 = f(prompt_len, gen_len)，不是单一常数。**

**端到端连贯性（实测 detokenize，非仅 token 正确）**：req0 的 prompt 尾部是
（中文）"...请用三句话概括上文的主要内容。`<｜Assistant｜>`"，serving 续写头 8 token
detokenize = **"DeepSeek-V4系列包含两款"**——**切题、连贯的真实回答**，非乱码。
即 serving 通路不只 token 与 golden 路径一致（E7F 步 3），实际输出**读起来是对的**。

**关键读法**：
1. **serving 图 decode 与裸引擎同速**（24.8 vs 25.49 ms/tok）——A（未饱和 padded）+
   B（decode-MoE）+ fused HC 合起来没有拖慢 decode。首版曾读到 36.9 ms/tok，
   **成因是 HC backend**：serving 默认误用 eager per-block HC，改回 e1f 的 **fused HC**
   即回到 24.8。
2. **折扣几乎全部来自首 token 延迟**（686 ms 摊到 48 token = 14 ms/tok 额外），
   即 prefill 流水。**折扣是生成长度的函数**：生成越长，框架 tok/s 越接近 40；
   越短，首 token 延迟越主导。**故 §1.2 的单一 20% 不足以刻画**——见下"待办"。
3. 确定性：框架 tok/s 在 5 个请求上 spread **0.11**。

## 待办（下一步）

- **✅ 已对齐**：ctx 1024（未饱和/A 路径）与 2048（饱和）**decode 都打平裸引擎**。
  4096 bucket 可补，但两点已定性（decode 满速、折扣随 prefill 缩放）。
- **✅ 生成长度摊薄曲线已起**（48→34%, 128→15%）；256 可补，趋势已明（→ decode 满速）。
  §1.2 的最终折扣数应绑定**目标操作点的典型 (prompt_len, gen_len)**，而非单点。
- **✅ teardown 挂起已修**：write_json 后 `os._exit`（数据已落盘；destroy_process_group
  在本拓扑 ~19 个自定义 group 下会挂）。运行现在干净收尾、done 哨兵及时触发。
- **✅ EOS 已加（`--stop-on-eos`）**：真变长生成实测（1024-prompt，上限 128）——
  各 prompt 自然停在 **58 / 90 / 128** token（前两个 hit_eos，第三个撞上限），
  跨 round 确定（req2 两 round 都 58）。**成本仅 ~0.29 ms/tok**（25.09 vs 24.80，+1.2%）——
  per-step 1-elem stop-flag **tensor 广播 + host read** 很便宜（首版误用 object broadcast 曾 +11ms/tok）。
  即 serving 现在是**满速的真变长生成器**。artifact `results/b1024-eos-rank0.json`。
- **HTTP**：后置（JSONL 已够测折扣）。
- **✅ back-to-back 自检已过**（从现有 artifact）：同一 prompt 在 round 1（前面已跑 5 个
  请求）与 round 0 的 token **逐字节相同**（1024 与 2048 bucket 各 3 prompt 全过）——
  **per-request 状态重装完全复位、无跨请求残留**。故 serving 数字不受请求顺序影响。
- **首 token 延迟归因（已查）**：prefill **已用 tilelang 稀疏核**（launcher 设
  `DSV4_PREFILL_SPARSE_BACKEND=tilelang`，StageLane 从 env 解析，multi-token 走 sparse core），
  **非未优化产物**。687 ms = **4 stage 串行的单 prompt prefill**（stage0 ~183ms × ~4 ≈ 687）——
  单 prompt 一个 chunk 过 4 个串行 stage，无 stage 间重叠。**降它的杠杆是 prefill 分块流水**
  （把 1024 prompt 切小块、跨 stage 流水填泡），是真优化、非平凡；单路短 prompt 下 prefill
  本就短，优先级由目标操作点定。
- **§1.2 折扣写入**：需绑定目标操作点的典型 (prompt_len, gen_len)——**产品输入**，
  不自拟。本实验给出的是**折扣函数**（decode 满速、折扣 = 首 token 延迟 / 生成长度），
  代入操作点即得单点。

⚠️ **本数字是"一个形状的首测"，不是 §1.2 的最终折扣**。写入 §1.2 前需按 bucket +
生成长度补齐（上"待办"第一条）。裸引擎 39.2 的口径见 E1F/E6F。
