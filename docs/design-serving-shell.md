# §10 Phase 1：最小单路 serving 外壳（design-first）

- 日期：2026-07-22 · 状态：**设计判断，实现前先读**。
- 目标（§10 Phase 1）：把单路从**裸引擎数字**变成**框架实测数字**，
  并在此过程中建成第一条 serving 通路。**第一产出是测出真实 serving 折扣**
  （现 39.2 tok/s 裸引擎 → 框架口径 = ?），不是先冲 150。
- 前提已就绪：E7F 已放行 **stateful serving 路径**（真实 prompt → prefill → 交接 →
  图 decode → token，§1.3 非逐位门），A（未饱和可捕图，无条件默认）、
  B（decode-only MoE 共享 prefill 权重）都在。

## 关键认识：引擎已经存在，外壳是把两半接起来

- **`e1f_full_decode_bench.py` 就是 serving decode 引擎**：16 卡闭环、free-running
  argmax、图 replay（stage3 head→argmax→NCCL loopback(loop_pair)→stage0 重嵌
  →图 replay→逐 stage 传递）。`pipeline_step`（:1138）即单步。**它只差两点**：
  状态来自**合成 seed**（`build_seed_payload`）、跑**固定步数**。
- **`e0ef2e_golden_gate.py` 就是真实 prefill + 交接那一半**：`StageLane` 真实 prefill、
  `_build_stateful_decode_stage`（E7F B：decode-MoE 共享 resident）、
  `tokenizer_preflight`（`encode_messages` + `AutoTokenizer`）。**它只差**：
  decode 是 teacher-forced（喂 golden）、批量跑完就退。

**⟹ 外壳 = e1f 引擎（free-running 图 decode）+ e0ef2e 的真实 prefill/交接/tokenizer
+ 一个请求循环 + EOS 停止。三块都已证，外壳是接线，不是新算法。**

## 执行模型（最小版，§10：无调度器/无槽位回收/无批处理准入）

一个**常驻** 16 rank torchrun 作业：

1. **加载一次**（PP4×TP4 全 stage），**捕图一次**（decode 三图族，位置无关，
   跨请求复用——状态每请求经交接重装进同一批 decode 缓冲，图读同一地址）。
2. **请求循环**（单请求串行）：
   - rank 0（stage0 tp_rank0）收请求 → `encode_messages`+`AutoTokenizer` →
     `prompt_tokens` → `broadcast_object_list` 给 16 rank。
   - 全 rank **prefill**（chunked，如 e0ef2e，eager；prefill 是带宽绑定，不捕图）
     → 状态到 `prompt_len`。
   - **交接**（E7F）：snapshot 状态 → 装进 decode-MoE stage 的静态状态。
   - **free-running decode**（e1f 闭环）：每步 argmax → loopback → 重嵌 → 图 replay，
     **直到 argmax==eos 或 max_tokens**。
   - rank 0 `AutoTokenizer.decode` → 返回文本 + 计时。
   - 请求间：非 rank0 阻塞在下一个 `broadcast`。

## 复用 / 新建 清单

| 块 | 复用自 | 新建 |
|---|---|---|
| 拓扑、图捕获、闭环 decode、loopback、warmup | `e1f` | — |
| 真实 prefill（StageLane）、交接、decode-MoE、tokenizer | `e0ef2e`(E7F) | — |
| 请求循环（broadcast prompt / return text） | — | **新** |
| EOS 停止（argmax==eos_token_id 即停该请求） | — | **新** |
| 变长 prefill → 定形 decode 图跨请求复用 | e1f 图位置无关 | **接线** |
| 框架口径计时（含 tokenize/detokenize、per-request tok/s） | — | **新** |
| HTTP（可后置；先 stdin/JSONL 请求文件即可测折扣） | — | 后置 |

## 折扣测量（第一产出）

- **裸引擎基线**：单路 decode **39.2 tok/s**（E1F，图口径，start_pos 2048）。
- **框架口径**：per-request 端到端 tok/s = 生成 token 数 /（prefill + decode +
  tokenize + detokenize + 传输 + 请求开销）墙钟。**含 prefill 的首 token 延迟**
  与 detokenize，即真实单路。
- **折扣 = 框架口径 / 裸引擎**。§1.2 现记单路折扣 20%（推断值，**从未实测**）——
  这个外壳第一次把它测出来。
- ⚠️ 计时纪律沿用 §9：≥3 轮报离散；独占目标 GPU；前后 nvidia-smi 快照落 artifact。

## 设计约束（为 Phase 2 预留，不重写）

- **槽位回收结构预留**：请求循环把"一条序列的状态"做成可被下一个请求填入的对象
  （decode-MoE stage 的静态状态已是"可重装"的——交接就是重装）。Phase 2 的
  聚合/并发是**扩展**这个循环（多 lane / 多 slot），不是重写。
- **无批处理准入 / 无调度器**（§10 明确不需要）——单请求串行即最小可行。
- **ctx < 2047 短请求**：A 已让它们可捕图（padded top-k），故短交互轮次也走图路径，
  不落回 210 ms/步 eager。这正是单路操作点。

## 边界与风险

- **首 token 延迟**：prefill 是 eager（chunked），长 prompt 的首 token 慢；
  单路操作点是短交互轮次，prefill 短，可接受。计时须**分列** prefill(首 token) 与
  decode(后续 token/s)，别把两者混成一个 tok/s。
- **采样**：最小版 greedy(argmax)，与 e1f 闭环一致。温度/top-p 后置。
- **图跨请求复用的正确性**：decode 图读固定 plan 缓冲 + 静态状态；每请求交接重装状态
  （E7F 已证交接逐位、图对 prefill 来的状态精确）。**须验**：连续两请求之间状态
  完全重置（无上一请求残留）——交接的 `seed_decode_payload` 是原子重装，理论上够，
  但要加一个"两请求 back-to-back、第二个结果与单独跑一致"的自检。
- **常驻作业的健壮性**：sarth ProxyJump 掉线不应杀作业（用 setsid 脱离 + 文件/socket
  通信），沿用 `run_e7f_golden_stateful.sh` 的脱离-轮询教训。

## 下一步（实现顺序）

1. **先做无 HTTP 的最小闭环**：一个常驻 torchrun，从 JSONL 读请求（prompt），
   prefill→交接→free-running decode→detokenize→打印文本 + per-request 计时。
   这一步就能测出**折扣**（第一产出），不需要 HTTP。
2. **加 back-to-back 自检**（两请求连跑 = 各自单跑，token 一致）。
3. **≥3 轮测折扣**，落 artifact，写 §1.2 单路折扣实测值（替换推断的 20%）。
4. HTTP 外壳（stdin/JSONL → HTTP）后置，不阻塞折扣测量。
