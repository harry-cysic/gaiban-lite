# DeepSeek-V4-Flash 部署目标（权威版）

- 版本：2026-07-21 · 状态：**本文是项目的唯一权威目标定义**
- 取代：立项报告 `docs/inception-topology-and-perf-estimates.md` 的性能预估部分，
  与 `docs/feasibility-v4-flash-2x8x4090.md` 的第 5 节预估带。两者保留为历史记录：
  前者含完整推导链（仍有参考价值），后者含实测修订记录（§5.5/§6）与环境附录（附录 B）。
- 维护规则：本文的"实测"列由实验结果更新，**修订必须带实验 ID**；被证伪的条目移入
  §5 而不是删除。预估与实测在本文中永远分列，不合并表述。

---

## 1. 验收标准（对外承诺，唯一的"做完"判据）

| 项 | 承诺 | 当前实测 | 口径 | 状态 |
|---|---|---|---|---|
| 单路 decode | **≥150 tok/s** | 27.5（+MTP 投影 ~38） | 16 卡 PP4、B=1、graph、bf16 | ❌ 差 4–5×，**从未针对性优化** |
| 单机聚合 decode | **≥2,000 tok/s** | 8,733（16 卡） | 16 卡 PP4、8K、FP8 KV、bl72×mb4 | ✅ 双机余量充足；**8 卡口径未验** |
| prefill | **≥15,000 input tok/s** | 25,308（16 卡） | 16 卡、whole-8192 | ✅ 双机达标；**8 卡口径未验（折算 ~12.6k）** |
| 上下文档位 | 128K / 512K / 1M | — | — | **完全未测** |

**口径错配是当前最大的交付风险**：承诺按"单机"写，验证全部在双机 16 卡完成。
8 卡 TP4×PP2 形态一次未跑（见 §7.2）。

质量验收：模型级长 prompt golden 门（D0L），当前基线 **494/512**。任何性能改动
若使该分数下降即不放行，**容差不得放宽**。

---

## 2. 部署模型：预定义模式

**不追求完全动态服务。** 交付形态是若干**预定义模式**，每个模式冻结自己的 B 与
max_seq，捕获自己的 CUDA graph 族。模式内需要的是**槽位回收**（一条序列结束后
把新请求填入该行），不是 continuous batching。切换模式是运维动作（重启/重捕），
不是每请求动作。

因此：`B 是图的编译期常量` 不是限制，是设计前提；多档位不叠加显存。

### 模式矩阵（"做完"= 每格都有冻结配置 + 实测数字 + 质量门背书）

| 模式 | 拓扑 | 冻结 B | 目标 | 实测 | 状态 |
|---|---|---|---|---|---|
| **M1 短 ctx 吞吐（2K）** | 16 卡 PP4 | bl≈72×mb4 | — | **9,656** | ✅ 已达 |
| **M2 标准吞吐（8K）** | 16 卡 PP4 | bl72×mb4 | 计划带 8–12K | **8,733** | ✅ 带内（下沿） |
| **M3 prefill 重载 / P 侧** | 16 卡 PP4 | chunk 8192 | ≥15K | **25,308** | ✅ 已达 |
| **M4 延迟 / 单用户** | **8 卡 TP4×PP2** | B=1–8 | **≥150** | 27.5（16 卡口径） | ❌ **未建，拓扑未验** |
| **M5 长上下文** | 16 卡 PP4 | 由 token-slot 预算定 | 128K/512K/1M | — | ❌ **完全未测** |
| **M6 混合单池（8K/1K）** | 16 卡 PP4 | — | — | T=2,322 | ⚠️ 裸引擎口径 |

**空白格优先级**：M4（有承诺、差 4–5×）> M5（双机方案核心卖点，零覆盖）> serving 折扣验证。

---

## 3. 定数（模型与硬件，不因目标而变）

| 项 | 值 |
|---|---|
| 权重 on-disk | 159.6 GB / 148.6 GiB（FP4 专家 + FP8 其余，无 BF16 税） |
| 结构 | 43 层 + 1 MTP；hidden 4096；vocab 129,280 |
| 层型分布 | 21×ratio-4（带 indexer, topk 512）+ 20×ratio-128 + 2×纯滑窗（L0/L1）+ MTP |
| MoE | 256 routed（MXFP4, inter 2048）+ 1 shared，top-6；**前 3 层 hash 路由** |
| Attention | 64 头 × head_dim 512（rope 64）；q_lora 1024；o_groups 8 × o_lora 1024 |
| 激活参数 | 13B/token；B=1 每 token 读 ~11 GB（**dense 侧为主，非专家**） |
| KV（bf16） | **~6.7 KB/token** 全模型（r4 21×320B + r128 20×8B）；fp8 减半 |
| KV（PP4 每卡） | ~1.7 KB/token/卡（= 全模型 × stage 层数/43） |
| PP 边界载荷 | mHC 4×4096×bf16 = **32 KB/token** |
| 每层专家权重 | 3.2 GB（FP4）；TP4 per-expert inter 切分后 0.8 GB/层/卡 |
| 权重/卡 | 16 卡 ~9.4 GiB；**8 卡 ~20 GiB**（8 卡形态的容量根因） |
| 实测容量墙 | 16 卡 bl80 撞 **22.4 GiB**（权重+KV+图池+工作区，结构性） |
| sm89 限制 | optin smem **101,376 B**；sparse attention head_chunk ≤16 |

---

## 4. 硬规则（已被实测确认，勿再挑战）

1. **TP=4，永不 TP8**：Flash 64 头 ÷8 = 8 头/卡会被 pad 到 16（50% 注意力算力白烧），
   且 86 次 allreduce 被迫跨 socket。TP4 = 16 头恰好。TP 只在 socket 内。
2. **P2P 通路必须双重确认**：驱动 patch **且** `NCCL_P2P_LEVEL=SYS`。GPU0–3 处于
   `NODE` 距离，NCCL 默认 P2P level 不含该距离，会**静默回退 SHM（4.12 GB/s vs
   23.79 GB/s）**。⚠️ **`nvidia-smi topo -p2p` 与 `cudaDeviceCanAccessPeer` 在
   NCCL 实际走 SHM 时仍全报 OK，对传输选择没有诊断力**——只有 `isAllDirectP2p`
   与实测带宽算数。每次性能测量须记录并自检（已内建于 result JSON）。
3. **CUDA graph 强制**：B=1 eager 210 ms/步 vs graph 36.3 ms（5.8×）。decode 是
   launch-bound。
4. **质量门必须用长 prompt + 真实权重**：短 prompt（10–22 token）永远不进入 prefill
   大行数路径，会给出无意义的"无损"；合成权重会掩盖 HC 数值问题（真实 hc_scale
   0.03–0.20 vs 合成 1.0）。构造长 prompt 时 **token 数必须精确命中档位**（1024/2048/…），
   差一个 token 就会绕过被测分支。
5. **reference 实现既是 oracle 也是 kernel 来源**：其 tilelang `sparse_attn` 是 prefill
   最大杠杆（6.49×）。不得因 `fp4_gemm` 慢而整体否定 reference kernel。

---

## 5. 已被实测证伪的假设（勿再据此推导）

### 5.1 出自 `feasibility-v4-flash-2x8x4090.md`

| 假设 | 结局 | 实验 |
|---|---|---|
| decode 15–25k @8K | 实测天花板 8,733（bf16 仅 6.4k） | E1F/E1IF |
| t_stage 随 B 线性 | **强次线性**：单 stage ≈ 8 ms 固定 + 0.17 ms/行 | E1F 扫描 |
| KV 容量账 | **漏算管线填充 ×mb 同驻**（B=512 工作点需 2048 in-flight，8K 下不可行） | E1IF |
| prefill 30–40k | 25,308（路径仍在，见 §7.4） | C2F |
| reference kernel 仅作 oracle | **错**：tilelang sparse_attn 6.49× | C2F |
| Marlin 大 M MFU 11.5% | **口径错误**：实为 135 TFLOPS ≈ 4090 BF16 峰值 82% | C2F 重归因 |
| gaiban 资产可直接复用 | decode 侧 fused MHC −3%、W8A16 中性、fused indexer 仅 prefill | C1F |

### 5.2 出自立项报告

| 假设 | 结局 | 实验 |
|---|---|---|
| batch-1 200–350 | 带宽天花板 335 **仍成立**；延迟栈估 3.5–5 ms，**实测 ~33 ms（差 ~10×）** | E1F |
| MTP ×1.5–1.8 | B=1 实测 1.30×（graph 投影 1.4×）；**大 B 证伪**（8K 持平、2K +13%） | MTP 竖条 |
| PP 交接固定税 2.3–2.7 ms/步 | **好于预期**：0.14–0.24 ms/32KiB | E0pf/E0qf |
| 8 卡显存模板（§0.2） | 自相矛盾（21.5+2+3 > 24）且低估图池/工作区 | 待验（§7.2） |
| dense-BF16 MoE 是 prefill 出路 | **慢 4–17×**（Marlin 融合了 swiglu+路由加权+unsort） | C2F-dense |
| 8 卡聚合 5–8K @8K | 未验，按容量实测很可能乐观 | 待验 |

### 5.3 出自本项目自身实验（自我修正）

| 假设 | 结局 |
|---|---|
| A5F：HC 融合数值 ≤1e-5、可回收 ~10 ms/stage | 合成权重假象（真实 ~1e-4）；集成态回收 4.7 ms |
| C3F：chunked prefill 快 1.5–2.3× | **证伪**：每 token 成本随每次前向行数**下降**，切段掉到更差档位（0.958–0.994×）。C3F 的观测来自分配器压力下的对照臂。**chunking 是容量杠杆，不是速度杠杆**，代价 0.6–4.2% |

---

## 6. 仍然有效的推导（可继续据此规划）

1. **单用户带宽天花板 335 tok/s**：B=1 每 token 读 ~11 GB 激活权重，PP 串行下
   任意时刻仅 4 卡工作，`5.5 GB ÷ (4×928 GB/s) ≈ 1.48 ms/级`。**至今无实测反驳。**
   当前 36.3 ms/token 距此地板 11.5×，缺口全在延迟栈（未 profile，见 §7.1）。
2. **MTP 在 B=1 的物理来源**：PP 串行下约 50% 级空转，draft/verify 错级填泡。
   这解释了为何 MTP 在 B=1 有效（1.30×）而在 94% busy 的满流水中无效。
3. **字节账**（权重、KV、PP 载荷、专家权重）全部与实测吻合，可继续用于容量规划。
4. **TP8 被否的推导**成立（见 §4.1）。

---

## 7. 开放问题（按优先级）

### 7.1 M4 延迟模式（有承诺，差 4–5×）
- **第一步是 profile，不是优化**：当前只有端到端 36.3 ms 与每 stage 8.3 ms，
  **11.5× 的开销没有任何分项归因**。候选：M=1 Marlin 异常（gaiban A1.5 记录 M=1
  48µs vs M≥2 21µs）、残留 eager torch 算子链、sinkhorn 微 kernel、86 次 allreduce。
- 该固定成本**同时限制吞吐**（bl=32 时占 stage 时间 40%，是"小 microbatch 交织
  不划算"的机制），所以 profile 对两条线都有价值。
- 延迟模式有**自己的图族、kernel 几何与拓扑**，可独立于吞吐路径特化。
- 家在 **8 卡 TP4×PP2**（报告：单用户是单机游戏，加机器不涨 batch-1）。

### 7.2 8 卡 TP4×PP2 形态未验（承诺口径）
权重 ~20 GiB/卡，留给图池+工作区+KV 的空间远小于 16 卡形态。这既是 M4 的家，
也是三项对外承诺的书面口径。**越晚发现错配，改形态越贵。**

### 7.3 M5 长上下文完全未测
128K/512K/1M 的容量、吞吐、以及"25 路 1M 会话并发"均无数据。这是双机方案的
核心卖点之一。注意 reference 侧 oracle 上限为 4096 token（见 §8），长 ctx 的
正确性验证需另设方案。

### 7.4 已知可回收的性能
- **vLLM fused HC 在 ≥1024 行数值错误**（rel_fro 2.3e-4 → 1.07e-1，公有 API 无法
  绕开；已用 MAX_ROWS=896 规避但精度仍不足）。**正确实现（自研或上游修复）可
  取回 prefill +19.9%（→30,345）**。分派点：`vllm/model_executor/kernels/mhc/
  tilelang.py:43`；通用路径 `hc_prenorm_gemm_tilelang` 已验证正确。
- MoE 集合重叠：**已结案**，天花板仅 51.8% 重叠率（NCCL P2P 在 NODE 距离 PCIe 上
  是 SM 驱动，与 Marlin 抢 SM），零开销实现也只到 +4%，不值得继续投入。

### 7.5 serving 折扣未验
立项报告明示所有计划值含 **30–40% serving 折扣**。当前所有数字都是**裸引擎数字**。
即使静态批也要付 tokenizer、HTTP、调度、KV 准入、采样、detokenize 开销。
另需验证**模式内槽位回收**（序列结束后不重捕图即可填入新请求）。

---

## 8. 运行环境要点（踩过的坑）

- `NCCL_P2P_LEVEL=SYS` 必设（见 §4.2）。
- **reference 侧 golden oracle 上限 4096 token**：`model.py:685` 的 `hc_post` 广播
  `[b,s,hc,hc,d]` fp32 = 256 KiB/token，8192 需恰好 2.00 GiB 单次分配而 OOM
  （`max_seq_len` 仅 ~72 MB，不是约束）。
- `ModelArgs.n_mtp_layers` 默认 1 且 Flash config 从不覆盖，会分配并加载一个
  **永不使用**的 MTP block（`Transformer.forward` 不触碰），去掉省 0.52 GiB。
- MoE 的 per-shape buffer 注册：**chunk 应整除 prompt 长度**（chunk=1000 注册 5 种
  形状，比 chunk=1024 多占 0.09 GiB）。
- `e0ef2e` 的 `result.json` 是人工挑选的字段子集，分析须以 per-rank JSON 为源。
- 环境事实（机器、venv、镜像、ssh）见 `feasibility-v4-flash-2x8x4090.md` 附录 B。
