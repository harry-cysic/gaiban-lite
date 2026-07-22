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

## 首个实测（1024-prompt / 48-new-token，fused HC，5 个测量请求，round-0 捕获排除）

artifact：`results/b1024-fused-rank0.json`。

| 量 | 值 | 对比裸引擎（E1F 39.2 tok/s @ctx2048） |
|---|---:|---|
| **decode** | **24.80 ms/tok（40.33 tok/s）** | **打平/略优** 裸 25.49 ms/tok —— serving 图 decode 路径**满速** |
| prefill（stage0 视角） | 182.7 ms | 1024-token prefill + 交接安装 |
| **首 token 延迟** | **686.6 ms** | 整条 prefill 流水（4 stage）+ head + loopback |
| **框架口径** | **25.92 tok/s** | 端到端 48 token；= 裸 39.2 的 **66%** |
| **⟹ 单路 serving 折扣** | **34%**（此形状） | §1.2 推断 20%，**此短生成形状实测更大** |

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

- **⚠️ 口径对齐**：本测在 ctx **1024**（未饱和，A 路径），裸 39.2 在 ctx **2048**。
  decode 打平已说明路径满速，但**折扣要按 bucket 分列**（1024/2048/4096 各测），
  且**扫生成长度**（首 token 延迟摊薄曲线）。本表只是**一个形状的一个点**。
- **teardown 挂起**：请求循环跑完后，最终 `dist.barrier`/`destroy_process_group`
  挂住（GPU idle 非自旋），需 `pkill` 清理。结果在循环内已写盘（rank JSON），
  不影响数据，但要修（可能是某 rank 的 barrier 不匹配）。
- **EOS**：当前定长（测折扣用）；交互版需 EOS，其 per-token host-read 成本单列。
- **HTTP**：后置（JSONL 已够测折扣）。
- **首 token 延迟优化**：prefill 是 eager chunked；单路操作点是短交互轮次，
  prefill 短，但 686 ms 仍是折扣主项，值得单独 profile（prefill 流水的 stage 摊薄）。

⚠️ **本数字是"一个形状的首测"，不是 §1.2 的最终折扣**。写入 §1.2 前需按 bucket +
生成长度补齐（上"待办"第一条）。裸引擎 39.2 的口径见 E1F/E6F。
