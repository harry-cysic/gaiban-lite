# D0L — 长 prompt golden-token oracle:让 E2E 门进入 prefill chunk 区间,并重判两个 prefill 杠杆

第二十四竖条(2026-07-21,titan064 + titan065 实测)。

## 动机

第二十三竖条([`C2F-prefill/results/prefill-overlap`](../C2F-prefill/results/prefill-overlap/README.md)
§1.6)在推两个 prefill 杠杆时发现了**门本身的缺口**:D0 golden oracle 的 8 条
prompt 只有 10–22 token,于是 E2E 门里 prefill 每 lane 只有 10–22 行。凡是"只在
长 chunk 下才改变行为"的 prefill 改动 ——

- 融合 HC 边界的 `MAX_ROWS = 896` 行切分,以及它所规避的 vLLM
  `num_tokens >= 1024` 错误 kernel 分支;
- fused indexer 的 `fuse_min_seqlen = 1024`;
- 按行分块的 MoE 集合/计算重叠(行数须整除块数);

—— **这个门在结构上一次都触发不到**。因此杠杆 A(469/482,判 FAIL)与杠杆 B
(473/482,判 PASS)两个结论都不具判定力,两杠杆当时按"待门"处理(默认关)。

本竖条补这个缺口:建长 prompt golden oracle,把 E2E 门的 prefill 推到 C2F 的
chunk 区间,然后用它重判两个杠杆。

## 结论速览

| 项 | 结果 |
|---|---|
| 长 prompt 集 | 10 条,**精确命中** 1024 / 2048 / 4096 / 8192 token;真实仓库文档拼接,构造脚本可复现 |
| **reference 侧长度上限** | **4096 token**(8192 实测 OOM)。瓶颈已精确定位:`model.py:685` `hc_post` 的 fp32 广播中间量 `[b,s,hc,hc,d]` = **256 KiB/token**,8192 时单次分配 **2.00 GiB** |
| 冻结 golden | [`results/oracle-long.json`](results/oracle-long.json):8 条(1024×3 / 2048×3 / 4096×2),每条 64 token,合计 **512** 个比较位(短门是 482) |
| **长门基线** | **494/512 = 0.9648**(冻结配置:tilelang prefill sparse + eager HC + 顺序 MoE) |
| **杠杆 A(prefill HC 边界融合)** | **489/512(−5)→ 不放行**。且最大分歧 `top2_gap` **1.492 已超出基线包络 0.960** |
| **杠杆 B(MoE 集合重叠 blocks=2)** | **491/512(−3)→ 不放行**。短门的 +1 **未复现**;但分歧仍全部落在基线包络内(max 0.869 < 0.960) |
| **chunk 覆盖证据** | 杠杆 A:**152/152** 个 HC 边界全部走行切分(1024→2 块 / 2048→3 块 / 4096→5 块);杠杆 B:**80/80** 次 prefill MoE 调用全部走流水路径 |

**一句话**:门修好了,而且**它一修好就把两个杠杆都判否了** —— 杠杆 B 在短门上
拿到的 +1 是噪声,在真正进入 chunk 区间后变成 −3。两杠杆维持"不放行、默认关"。

---

## 1. 长 prompt 集

### 1.1 构造(`build_long_prompts.py`,可复现)

1. **语料**:按固定顺序拼接本仓库 **23 份真实文档**(英文 model card
   `reference/README.md` + 中文工程文档 `docs/feasibility-*.md`、
   `runtime/PORT-PLAN.md`、各实验 README 等),分隔符 `\n\n=====\n\n`,
   共 151,053 字符。**排除** `reference/encoding/README.md` —— 它引用了
   `<|begin_of_sentence|>` 一类特殊 token 字面量(U+FF5C `｜`),混进 user
   content 会被重新 tokenize 成控制 token;脚本对此有硬校验。
   选真实自然文本而非按词表随机采样,是因为随机 token 的统计分布模型没见过,
   MoE 的 top-k 路由行为不具代表性。
2. **切窗**:每条 prompt 取语料的一段**互不重叠**的字符窗口作为正文,
   外面套 header + 尾部指令("读下面的节选,然后回答……"),中英文交替。
   正文在句中截断,尾部指令保证 prompt 仍是良构任务。
3. **精确定长**:二分正文字符数,使 `encode_messages(thinking_mode="chat")`
   → `tokenizer.encode` 后的**整条 prompt** token 数正好等于目标。
   token 数对字符数单调不减但会**跳 2**(某个字符把一次 BPE 合并劈开),
   所以纯二分会跳过目标;补法是往正文尾部追加短 filler(`" ."` / `"\n"` …),
   一次一个单位地长到正好命中。10 条**全部精确命中**(0–2 个 filler 字符)。

**为什么必须精确**:1023 行会落在 vLLM `>= 1024` 那条错误 kernel 分支的**下方**
(即 `MAX_ROWS=896` 行切分要防的东西根本不触发),而**奇数**行数会让按行分块的
MoE 重叠回退到顺序路径 —— 两种情况下被测杠杆都会"静默地没跑",门却会报一个
毫无意义的"无损"。第一版构造出来是 1023/1023/2047/2047,正是这个坑,已修。

产物 [`long_prompts.json`](long_prompts.json) 冻结了 prompt 原文 + 出处
(每份源文件的 md5、字符数、窗口偏移),即使源文档以后被改,prompt 集不变。

### 1.2 长度分布

| 目标 | 条数 | 实际 token 数 | 每 lane prefill 行数 | MoE 全局行数(TP4) |
|---:|---:|---:|---:|---:|
| 1024 | 3 | 1024(精确) | 1024 | 4096 |
| 2048 | 3 | 2048(精确) | 2048 | 8192 |
| 4096 | 2 | 4096(精确) | 4096 | 16384 |
| 8192 | 2 | 8192(精确) | — | —(reference 侧放不下,见 §1.3) |

前 8 条(≤4096)进入 golden;8192 两条**保留在 `long_prompts.json` 里**,
等 reference 侧容量问题解决后可直接复用。

### 1.3 reference 侧长度上限 —— **4096**,依据如下

reference MP=8 单机(titan064,8×4090,23.52 GiB/卡),权重驻留后
**free 2.21 GiB**。逐条实测峰值(`torch.cuda.max_memory_allocated`):

| prompt 长度 | peak allocated | 剩余 free |
|---:|---:|---:|
| 1024 | 21.315 GiB | 1.52 GiB |
| 2048 | 21.762 GiB | 0.95 GiB |
| **4096** | **22.655 GiB** | **0.05 GiB** |
| 8192 | — | **OOM** |

8192 的失败点是**确定的**,不是笼统的"显存不够":

```
model.py:693 Block.forward -> model.py:685 hc_post
torch.OutOfMemoryError: Tried to allocate 2.00 GiB.
GPU 0 total 23.52 GiB, 1.14 GiB free, 21.72 GiB allocated by PyTorch
```

`hc_post` 是
`post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`,
其中 `comb.unsqueeze(-1) * residual.unsqueeze(-2)` 广播成
**`[b, s, hc, hc, d]` fp32** = `4 × 4 × 4096 × 4 B` = **256 KiB/token**:
4096 时 1.00 GiB、8192 时正好 **2.00 GiB**,而且每层要算两次(attn 侧 + ffn 侧)。
这就是 4096 只剩 0.05 GiB、8192 直接爆的原因。

**上限记录为 4096**。要上 8192 需要改 reference 的 HC 实现(按行分块),
那会动 oracle 的语义基准,本竖条不做。

> 顺带:`max_seq_len` 本身**不是**瓶颈 —— 8320 与 4224 的 KV + freqs_cis 差只有
> ~72 MB。真正的量是激活峰值。

### 1.4 意外发现:reference 默认驻留一个**用不到的 MTP block**

`ModelArgs.n_mtp_layers` 默认 **1**,而 V4-Flash 的 `config.json` 没有覆盖它,
所以 `Transformer.__init__` 会建并加载一整个 `MTPBlock`(attention + 32 个本地
FP4 专家)—— 但 `Transformer.forward` **从不触碰 `self.mtp`**,在 generate 路径上
它是纯粹的死驻留。

`--drop-mtp`(即 `n_mtp_layers=0`)后:**free 1.69 → 2.21 GiB(+0.52 GiB)**,
**这正是 4096 能不能跑的分水岭**(4096 只剩 0.05 GiB)。

**已验证 token 等价**:带 MTP 的那次 run 与 drop 后的 run 在 6 条重叠 prompt 上
`prompt_tokens` / `completion_tokens` **逐条完全相同**(6 identical / 0 different)。
D0 短 oracle 当时也带着这个死驻留,只是短 prompt 下不吃紧。

---

## 2. golden 冻结(`results/oracle-long.json`)

方法沿用 D0(reference 实现、MP=8、贪心 `temperature=0`/argmax),三点差异
都在 `oracle_long_generate.py` 的 docstring 里写明:

1. **输入改 JSON**:长 prompt 是真实文档节选,自带空行,D0 的 `"\n\n"` 分隔
   格式表达不了。
2. **一条一次 `generate`(batch=1)**:`generate.py` 只 prefill
   `min(prompt_lens)` 再逐 token 前进(`prompt_mask` 覆盖),混长度 batch 会把
   1024/8192 的组合变成 ~7000 次单 token forward 而不是 prefill。batch=1 还
   让每条 prompt 拿到**满长度 `start_pos=0` prefill** —— 正是 E2E 门要复现的形状
   —— 且激活峰值最小。
3. **prompt 之间显式复位状态**:reference 的 KV/compressor 是跨 `generate` 调用
   持久的 `register_buffer`。长度是 compress ratio 整数倍时 `start_pos==0` 分支
   恰好会覆盖掉后面会读到的部分,但依赖这个太脆;`reset_model_state` 把
   `kv_state`→0、`score_state`→−inf、`kv_cache`→0 复位成构造器初值,
   使每条 prompt 等价于全新进程。

环境指纹:titan064 / world_size 8 / torch 2.11.0+cu130 / RTX 4090 /
config md5 `aba1b3578f5554013ff20a422c81c9b7`(与 D0 同) /
prompts md5 `8d8ffed1f7fa0a07b3216f9734db2bd3` / `max_seq_len` 4224 /
`n_mtp_layers` 0 / 8 个 mp8 分片的文件清单与字节数。

输出质量正常(中英文摘要成篇、要点抓取正确),8 条 completion 全部跑满 64 token
(无提前 EOS),合计 **512 个比较位**。

---

## 3. E2E 门的长臂

### 3.1 门的改动(`runtime/e0ef2e_golden_gate.py`,默认行为不变)

沿用同一支门而不是另写一个,新增旗标的默认值都保持 D0 短门的原行为:

| 旗标 | 默认 | 长臂用值 | 为什么需要 |
|---|---|---|---|
| `--max-seq-len` | 256(原硬编码 `MAX_SEQ_LEN`) | 4224 | 4096 prompt + 64 解码步 |
| `--share-moe-buffers` | off | on | Marlin per-shape 缓冲 **80 KiB/全局行**;11 层各持一套,16384 行时 11×1.25 GiB **加载即 OOM** |
| `--prompt-min-tokens` / `--prompt-max-tokens` | 0 / 0 | — | 按长度分桶(每个不同长度都要注册一套 per-shape 缓冲) |
| `--max-steps` | 128 | 64 | golden 是 64 token + generate.py 追加的合成 EOS;取 64 正好把合成 EOS 排除在外 |

另外把 `prefill_evidence` / `run_evidence` 写进每条 prompt 的记录(见 §3.2)。

**runtime 侧还修了一个真 bug**(`dsv4_direct/physical_stage.py`):
`share_moe_buffers` 的 donor 是**一次性**从 stage 首层选的,而
`build_physical_layer_material` 对 route_kind 不匹配的 donor 会拒绝
(hash 路由多一个 `gathered_input_ids` 缓冲,两种 route kind 不能共享)。
`model_contract.py:101` 规定 layer 0–2 是 hash、3+ 是 learned,而
`STAGE_LAYERS[0] = range(0,11)` —— 于是 **stage 0 的 8 个 learned 层全部没共享上**,
16384 行时是 8×1.25 GiB,加载即 OOM。改成**按 route_kind 各选一个 donor**。
stage 1–3 全 learned,原本就正常;C2F 单 stage bench 用的是 L11–21(全 learned),
所以从没撞上。修完实测:stage 0 载入后 free **4.23 GiB**、stage 1 **7.30 GiB**、
stage 3 **5.61 GiB**。

### 3.2 chunked prefill 确实被覆盖 —— 证据

先把口径说清楚,这一点很容易含混:

- **runtime 的 prefill 形态是"整段一次 forward"**,不是增量分块。
  `Ratio4FullPositionAttention.__call__`(`ratio4_fullpos.py:550-553`)对
  `seqlen > 1` **强制要求 `start_pos == 0`**,`start_pos > 0` 只接受单 token。
  即**当前 runtime 不支持 `start_pos>0` 的多 token prefill**,增量 chunked
  prefill 今天写不出来。
- 所以 C2F 全系列里 **"chunk" = 一次 `start_pos=0` 整段 prefill 的序列长**
  (C2F README 口径原文)。本竖条的"进入 chunk 区间"= **让那一次整段 prefill 的
  行数落到 1024/2048/4096**,而不是把 prompt 切成多段喂。

按这个口径,每条 prompt 的 prefill 是 **1 次 forward**,长度 = prompt 长度。
门里逐条记录(`prefill_evidence`):

| prompt | seqlen | 每 lane 行数 | MoE 全局行数 | prefill forward 数 | 端到端 prefill 墙钟 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1024 | 1024 | 4096 | 1 | 17,014 ms(含 JIT 冷启) |
| 1–2 | 1024 | 1024 | 4096 | 1 | 730 / 733 ms |
| 3–5 | 2048 | 2048 | 8192 | 1 | 1477 / 1478 / 1478 ms |
| 6–7 | 4096 | 4096 | 16384 | 1 | 2965 / 2961 ms |

(墙钟取自 rank 12 = stage3 tp0,含 PP 上游等待,即整条流水的 prefill 时延;
对长度基本线性,和"整段一次 forward"自洽。)

**真正的"分块"发生在杠杆 A 内部**,而这正是要验的东西 ——
`FusedTilelangHCBoundaryBackend` 对 >896 行按行切块。门里读的是 backend 自己的
计数器(纯观测,不改行为):

| prompt 长度 | HC 边界调用数 | 走行切分的 | kernel launch 数 | 每次切几块 |
|---:|---:|---:|---:|---:|
| 1024 | 19 | **19** | 38 | 2 × 512 |
| 2048 | 19 | **19** | 57 | 3 × 683 |
| 4096 | 19 | **19** | 95 | 5 × 820 |
| **合计(8 条)** | **152** | **152(100%)** | — | — |

(19 = stage 3 的 10 层 × 层内边界 + 9 个层间边界。)
`row_histogram` 逐条记录了实际行数(`{"1024": 19}` / `{"2048": 19}` / `{"4096": 19}`),
**全部 > 896**,即 §1.2 那条"必须精确定长"的要求确实兑现了。

杠杆 B 同理,`TP4MoE.overlap_stats` 记录实际走的路径:

| 臂 | prefill MoE 调用:流水路径 | 顺序路径 |
|---|---:|---:|
| 基线 / 杠杆 A | 0 | **80** |
| **杠杆 B(blocks=2)** | **80(100%)** | 0 |

对照第二十三竖条:短门 8 条 prompt 里有 2 条长度 13(奇数)会自动回退顺序路径,
**这次 80/80 全部走流水**。

---

## 4. 两杠杆重判

**验收准则**(沿用第二十三竖条):同口径下**不劣化于基线**。三臂唯一差异就是杠杆
本身 —— 同一份 golden、同一 `--max-seq-len` / `--max-steps` / `--share-moe-buffers`、
同一 `DSV4_PREFILL_SPARSE_BACKEND=tilelang`、同一双机 16 rank 拓扑。

### 4.1 总分

| 臂 | 配置 | 合计 | match rate | mismatch 数 | `top2_gap` max / median | 判定 |
|---|---|---:|---:|---:|---:|---|
| **基线** | eager HC,顺序 MoE | **494/512** | 0.9648 | 18 | 0.9595 / 0.3458 | — |
| **杠杆 A** | prefill HC fused(decode 保持 eager) | **489/512** | 0.9551 | 23 | **1.4920** / 0.2127 | **不放行(−5)** |
| **杠杆 B** | MoE 重叠 blocks=2,HC 全 eager | **491/512** | 0.9590 | 21 | 0.8687 / 0.3521 | **不放行(−3)** |

逐 prompt:

| prompt 长度 | 基线 | 杠杆 A | Δ | 杠杆 B | Δ |
|---:|---:|---:|---:|---:|---:|
| 1024 | 62 | 61 | −1 | 62 | 0 |
| 1024 | 62 | 60 | −2 | 61 | −1 |
| 1024 | 63 | 62 | −1 | 63 | 0 |
| 2048 | 63 | 63 | 0 | 64 | **+1** |
| 2048 | 61 | 60 | −1 | 61 | 0 |
| 2048 | 61 | 60 | −1 | 58 | **−3** |
| 4096 | 59 | 60 | **+1** | 59 | 0 |
| 4096 | 63 | 63 | 0 | 63 | 0 |
| **合计** | **494** | **489** | **−5** | **491** | **−3** |

### 4.2 杠杆 A — prefill HC 边界融合:**不放行**

- **−5**,不满足"不劣化"。8 条里 5 条掉、1 条涨、2 条平 —— **单向漂移**,
  不是对称噪声。
- 更关键的是**质量**而不只是数量:杠杆 A 的最大分歧 `top2_gap` = **1.4920**,
  **超出基线自身的近平局包络(max 0.9595)**。也就是说它至少产生了一次
  "按基线标准算不上近平局"的判决翻转。第二十三竖条在短 prompt 下看不到这一点
  (那时它的 max 是 0.8425,在基线 0.5621 附近)。
- **这次的判定是有覆盖力的**:152/152 个边界全部走了 `MAX_ROWS=896` 行切分
  (短门下这条路径**一次都没触发过**)。所以结论从"在小行数下掉 3 个 token"
  升级为"**在真实 chunk 行数下、行切分全程生效时,仍然掉 5 个 token,且有一次
  分歧超出基线包络**"。
- 第二十三竖条留的问题("是这条链本来就对 bf16 敏感,还是融合边界确实更差")
  **现在有答案了:在 chunk 区间它确实更差**,不是门看不见的假象。

### 4.3 杠杆 B — MoE 集合/计算重叠(blocks=2):**不放行**

- **−3**,不满足"不劣化"。**短门的 +1(473/482 vs 472)未复现** —— 那 +1 是
  噪声,不是证据。这正是第二十三竖条自己提醒过的("+1 不是'更准',
  与 −3 同性质")。
- 但要如实区分:**杠杆 B 的分歧在质量上与基线不可区分** —— `top2_gap` max
  **0.8687 < 基线 0.9595**,全部落在基线包络内。8 条里只有 3 条变化
  (+1 / −1 / **−3**),其余 5 条逐 token 持平;**净损失几乎全部来自单独一条
  prompt**(#5,2048 token,61→58)。所以杠杆 B 是"没通过计数准则",
  而不是"出现了新的错误类型"——与杠杆 A 的性质不同。
- **这次同样是有覆盖力的**:80/80 次 prefill MoE 调用全部走流水路径,
  而短门里有 2 条 prompt 因奇数行数自动回退。所以这是**流水路径第一次被真正
  按规模检验**。
- 结合第二十三竖条已量化的收益(+1.5%,受 NCCL/计算并发 51.8% 硬件上限封顶):
  **收益本来就在天花板上、现在质量侧又拿不到"不劣化",不建议再投入。**

### 4.4 关于统计力,如实说明

- 门是**确定性**的(teacher-forced、无采样、runtime 决定性)。**已实测坐实**:
  基线原样复跑一次(`base-r2`,独立进程、重新加载权重),
  **494/512 逐 prompt 完全相同**(62/62/63/63/61/61/59/63)、
  `top2_gap` 统计量完全相同、8 条的 first-mismatch step 也完全相同
  (32/27/36/11/30/5/18/3)。所以 −5 / −3 **不是** run-to-run 抖动,
  是配置差异的确定性后果。
- 但 512 个比较位、基线本身就有 18 处分歧,**−3 落在"换一组 prompt 可能变号"
  的量级里**;−5 且伴随包络外分歧的杠杆 A 结论更硬。
- 两个杠杆都是**语义变更**(A:kernel 数学不同;B:行置换改变 Marlin 的 M 分组),
  本来就不该期望逐位。判定按既定准则给,不放宽,也不加严。

---

## 5. 收尾检查

- **门的确定性**:基线复跑 `base-r2` 与首跑**逐 prompt、逐统计量完全一致**
  (§4.4)。产物 [`results/e2e-long-base-r2.json`](results/e2e-long-base-r2.json)。
- **短门未被破坏(已实测回归)**:长臂是在同一支 `e0ef2e_golden_gate.py` 上加
  旗标,新旗标默认值保持 D0 短门原行为。改完后用 D0 短 golden
  (`oracle-mp8.json`)按第二十一竖条的原配置(`run_e0e2e_tilelang_arm.sh`)
  复跑一次:

  | 臂 | 合计 | 逐 prompt |
  |---|---:|---|
  | 冻结的第二十一竖条 tilelang 臂 | 472/482 | 2, 29, 127, 124, 12, 22, 32, 124 |
  | **本竖条改动后复跑** | **472/482** | **2, 29, 127, 124, 12, 22, 32, 124** |

  **合计、逐 prompt、`top2_gap` 与 `golden_deficit` 统计量全部逐字相同**,
  `overall: PASS`。产物
  [`results/e2e-short-regression.json`](results/e2e-short-regression.json)。
- **两机显存**:三臂 run 前后 `nvidia-smi` 均记录在 launcher 日志里,
  **run 后 titan064 / titan065 共 16 张卡全部回到 1 MiB**。
- **run 内峰值**(`torch.cuda.max_memory_allocated`,rank0):
  基线 **20.671 GiB**、杠杆 A **20.734 GiB**、杠杆 B **20.675 GiB**
  (peak reserved 22.123 / 22.143 / 22.125 GiB,容量 23.52 GiB;
  `base-r2` 与基线逐字节相同 20.671 / 22.123)。
  三臂高水位实质不变;`--share-moe-buffers` + donor 修复是能装下的前提。
- **数字复核**:三份 `result.json` 已拉回本地
  (`results/e2e-long-{base,leverA,leverB}.json`),表里所有分数由
  [`summarize.py`](summarize.py) 从本地 JSON 重新聚合得出,不取日志行。

---

## 6. 意外发现(原样记录)

1. **reference 默认驻留一个用不到的 MTP block**(§1.4)。`n_mtp_layers` 默认 1、
   config 不覆盖、forward 从不用。0.52 GiB,**恰好是 4096 能否跑通的分水岭**。
   已验证 drop 后 token 逐条相同。D0 短 oracle 同样带着它。
2. **reference 长 prompt 的容量瓶颈是 `hc_post` 的 fp32 广播中间量**
   `[b,s,hc,hc,d]` = **256 KiB/token**(§1.3),不是 KV、不是 `max_seq_len`。
   8192 时单次 2.00 GiB。这也解释了为什么 runtime 侧要专门做 HC 边界融合。
3. **`build_physical_stage(share_moe_buffers=True)` 的 donor 选择有 bug**(§3.1):
   donor 只选一次,route_kind 不匹配就静默不共享,导致 **stage 0 的 8 个 learned
   层各自分配**。C2F 单 stage bench 用全 learned 的 L11–21,所以一直没暴露。已修。
4. **杠杆 B 在短门上的 +1 没有复现**,长门给 −3。**短门 ±1 量级的差异不具证据力** ——
   这条对以后所有"用 482 分制判 ±1~2"的结论都适用。
5. **杠杆 A 的最大分歧首次跑出基线包络**(1.4920 vs 0.9595)。短 prompt 下它的
   分歧还都在近平局带内,进入 chunk 区间后不再是。
6. **`lane_argmax_agreement = False` 是既有状态,不是长 prompt 引入的**:
   已核对 `out-e0e2e`(468)、`out-e0e2e-tl-tilelang`(472)、
   `out-e0e2e-hcf-hcfused`(469)、`out-e0e2e-hcf-moeovl`(473)四次历史 run,
   **全部都是 False**。它不在 `accepted` 的判据里。
7. **runtime 今天不支持增量 chunked prefill**(§3.2):
   `Ratio4FullPositionAttention` 对多 token 输入强制 `start_pos == 0`。
   "chunked prefill 覆盖"只能按 C2F 的口径理解为"整段 prefill 的行数进入 chunk
   区间"。若将来要做真正的分段 prefill(P/D 分离、长 context 分块),
   **这是一个还没写的能力**,不是配置项。

---

## 7. 产物

| 文件 | 内容 |
|---|---|
| [`build_long_prompts.py`](build_long_prompts.py) | 长 prompt 构造(语料清单、切窗、精确定长、出处 md5) |
| [`long_prompts.json`](long_prompts.json) | 冻结的 10 条 prompt(含 8192 两条)+ 出处 |
| [`oracle_long_generate.py`](oracle_long_generate.py) | reference MP=8 长 prompt golden 生成(batch=1、状态复位、`--drop-mtp`、逐条落盘) |
| [`results/oracle-long.json`](results/oracle-long.json) | **冻结 golden**:8 条 × 64 token = 512 比较位 + 环境指纹 + 逐条峰值显存 |
| [`results/e2e-long-base.json`](results/e2e-long-base.json) | 长门基线 494/512 |
| [`results/e2e-long-leverA.json`](results/e2e-long-leverA.json) | 杠杆 A 489/512 + HC 行切分证据 |
| [`results/e2e-long-leverB.json`](results/e2e-long-leverB.json) | 杠杆 B 491/512 + MoE 流水证据 |
| [`results/e2e-long-base-r2.json`](results/e2e-long-base-r2.json) | 基线复跑,与首跑逐 prompt 完全一致(门的确定性证据) |
| [`results/e2e-short-regression.json`](results/e2e-short-regression.json) | 改动后的 D0 短门回归,472/482 逐字不变 |
| [`summarize.py`](summarize.py) | 从本地 JSON 聚合三臂对比与覆盖证据 |

runtime 侧:
[`runtime/e0ef2e_golden_gate.py`](../../runtime/e0ef2e_golden_gate.py)
(`--max-seq-len` / `--share-moe-buffers` / `--prompt-{min,max}-tokens` /
prefill 证据记录)、
[`runtime/dsv4_direct/physical_stage.py`](../../runtime/dsv4_direct/physical_stage.py)
(per-route-kind buffer donor 修复)、
[`runtime/dsv4_direct/hc_boundary_backend.py`](../../runtime/dsv4_direct/hc_boundary_backend.py)
与 [`runtime/dsv4_direct/moe_runtime.py`](../../runtime/dsv4_direct/moe_runtime.py)
(纯观测计数器)、
launcher [`runtime/run_e0l2e_long_arm.sh`](../../runtime/run_e0l2e_long_arm.sh)。

titan064 运行现场:`~/flash-oracle-long/`(prompt 集、golden、日志)、
`~/flash-oracle/reference/inference/oracle_long_generate.py`。
