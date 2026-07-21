# tilelang `sparse_attn` 接入 direct runtime 的 prefill 路径(2026-07-21)

第二十一竖条。重归因(`../reattribution/`)定位 prefill 的 58% 在 attention,
而 runtime 一直跑 torch masked-einsum 正确性核;微基准判活给出 6.49×。本竖条把
reference 的 tilelang `sparse_attn` 作为**可选 prefill backend** 接入,过数值门、
E2E golden 门,并按 C2F 同口径重测吞吐。

> **状态(2026-07-21 更新)**:本页吞吐的端到端数字是在 `NCCL_P2P_LEVEL` 缺失、
> MoE 退到 SHM 的环境下测的(§4.3 有更正块)。修复后的同口径实测是
> **torch 17,022 → tilelang 25,308 input tok/s/stage**,见 `../moe-alloc/`。
> 本页的 **attention 分量、全部数值门与 E2E 结论不受影响**。

**结论:接入成立。** 单算子门 rel_fro ≤ 2.06e-3(限 1e-2)全 PASS;层级 oracle
(e0ef/e0wf)**按原容差不放宽** PASS;E2E golden **472/482**(冻结基线 468/482,
不劣化且略好);prefill attention 桶 **1.147 s → 0.520 s(2.20×)**,与微基准
预测的 ~611 ms/pass 节省相差 **+2.6%**(实测 626.6 ms)。

## 1. 实现要点

新增 `runtime/dsv4_direct/ops/tilelang_sparse.py`,提供与
`attention.torch_sparse_attention` **完全同签名**的
`tilelang_sparse_attention(query, latent_kv, attn_sink, topk_indices, softmax_scale, head_chunk=None)`。

### 1.1 `-1` padding 语义对齐(两处真实差异,均已在包装里对齐)

先读 torch 版(`attention.py:594`)确认语义,再对拍 kernel(`kernel.py:294-352`):

| 情形 | torch 版 | reference kernel 原样 | 包装处理 |
|---|---|---|---|
| 任意负数索引 | `valid = topk >= 0`,**任何负数**都屏蔽 | 只测 `!= -1`;`-2` 会当**有效**去 gather `kv[b,-2]`(回绕读) | 检出后归一化为 `-1` |
| 整行全 padding | row max 回落到 sink,分子 0 分母 1 → **输出 0** | running max 停在 `-inf`,`exp(-inf-(-inf))` → **NaN 污染整行** | 检出后把该行清零 |
| 有效索引越界 | 显式 `.any()` 校验后抛错 | 无校验 | 同 torch,抛错 |

两个差异都**实测坐实**,不是纸面推演(`c2f-tilelang-op-gate.json` 的 `edge_cases`):

- 全 padding 行:raw kernel 输出 **32768 个非有限值**(= 64 heads × 512 dim,整行),
  torch 输出全 0,包装后全 0。
- `-2` padding:raw kernel 对 torch 的 rel_fro **0.2988** / max_abs 2.78e-2,
  包装后 **2.01e-3**(回到正常 bf16 量级)。

**在现网三个 prefill 调用点上这两条都不会触发**:`window_topk_indices`、
`compressed_topk_indices` 的 padding 都是**恰好 `-1`**,ratio-4 走
`-1 - offset` 再 `+ offset` 也**恰好 `-1`**;而三处的 topk 都拼了因果 window 部分,
对角元恒有效,故不存在全 padding 行(op gate 里 `first_row_valid` 逐例 = 1)。
包装仍做校验,让两个 backend **无条件**等价而不是依赖该不变量。

第三处差异**无法对齐,只能记录**:torch 用 `M = max(row_max, sink)` 稳定化,
kernel 用 `m = row_max` 再补 `exp(sink - m)`。两者代数恒等,只差舍入;kernel 形式
在 `sink - row_max > ~88` 时 FP32 溢出。gate 实测该余量 **最大 3.04**(阈值 88),
留 29× 余量。

### 1.2 head-loop(sm89 smem 墙)

`sparse_attn_kernel` 的 shared 用量 = `q[h,d] + kv[64,d] + o[h,d] + acc_s_cast[h,64]`。
`h=64,d=512` 需 141312 B > sm89 的 101376 B(A4F 结论)。包装按 `head_chunk`(默认
16)循环 head 轴;heads 在该 kernel 里完全独立(逐 head softmax、逐 head sink),
故切分是**精确分解**。gate 实测 `head_chunk` 16 与 8 的结果**逐位一致**,
32 / 64 分别报 `Failed to set the allowed dynamic shared memory size to 104448 / 141312`
——**16 是 sm89 上的上限**,坐实 A4F。

输出用预分配 + `copy_` 写入 head 切片(而非探针那样 `torch.cat` 四块),峰值省一份输出。

### 1.3 注入点(三处 prefill,decode 一律不动)

| 文件 | 位置 | 开关 |
|---|---|---|
| `attention.py` | `Ratio128TorchAttention.__call__`(ratio-128 prefill) | env `DSV4_PREFILL_SPARSE_BACKEND` |
| `window_attention.py` | `WindowTorchAttention.__call__`(滑窗 prefill) | 同上 |
| `ratio4_fullpos.py` | `Ratio4FullPositionAttention.__call__`(C2F prefill 用) | 构造参数 `prefill_sparse_backend`,缺省回落到同一 env |

风格与既有 `--index-score-mode` / `--kv-dtype` / `DSV4_PREFILL_SPARSE_ROW_BLOCK` 一致,
**默认 `torch`,语义零变化**。三处都只在 `start_pos == 0` 分支切核;decode 与
plan/stateful 路径(含"禁止注入 backend"断言的那些)一个字节没动——注意本竖条用的是
**新的 prefill 开关**,不是既有的 `sparse_attention_backend` 注入协议(那个是 decode 侧的)。

tilelang 臂下 **row blocking 自动关闭**:`sparse_row_block` /
`DSV4_PREFILL_SPARSE_ROW_BLOCK` 存在的唯一理由是给 torch 核的 FP32 gather 工作区
(8192×640×512 fp32 ≈ 10.7 GB)封顶,而 kernel 根本不物化它;行独立,两种形式同解。

reference `kernel.py` 定位顺序:`DSV4_TILELANG_KERNEL`(文件)→
`DSV4_TILELANG_REFERENCE_DIR`(目录)→ **沿本文件向上找 `reference/inference/kernel.py`**
→ `~/a5f/`、`~/flash-oracle/reference/inference/` 等 home 候选。用
`importlib.util.spec_from_file_location` 以私有模块名导入,避免与其他 `kernel` 模块互相遮蔽。
三台机上的 `kernel.py` md5 一致(`e4d8e272f13515b899ef8b145b736001`);launcher 把仓库
自带的那份 rsync 到 `~/e0f-runtime/reference/inference/`,实测 E2E 两节点都命中它
(日志 `WARM /home/cysic/e0f-runtime/reference/inference/kernel.py`),即**臂钉在仓库版本**上。

## 2. 数值门

### 2.1 单算子门 — PASS(`c2f-tilelang-op-gate.json`,titan065 单卡)

11 个配置 × 2 个 head-chunk 宽度,索引几何**按三个调用点各自的真实构造方式**生成
(window-only / window+compressed / window+top-k),覆盖 padding 密集的早位置。
门限 rel_fro ≤ 1e-2、max_abs ≤ 5e-3(冻结值,未放宽)。

| 例 | 候选数 | padding 占比 | rel_fro | max_abs |
|---|---:|---:|---:|---:|
| window-96 | 96 | 0.495 | 1.947e-3 | 4.88e-4 |
| window-128 | 128 | 0.496 | 1.935e-3 | 4.88e-4 |
| window-200 | 128 | 0.318 | 1.956e-3 | 4.88e-4 |
| window-2048 | 128 | 0.031 | 2.046e-3 | 4.88e-4 |
| ratio128-128 | 129 | 0.500 | 1.966e-3 | 4.88e-4 |
| ratio128-512 | 132 | 0.139 | 1.981e-3 | 4.88e-4 |
| ratio128-2048 | 144 | 0.087 | 2.046e-3 | 4.88e-4 |
| ratio128-8192 | 192 | 0.174 | **2.056e-3** | 4.88e-4 |
| ratio4-512 | 256 | 0.313 | 1.989e-3 | 4.88e-4 |
| ratio4-2048 | 640 | 0.407 | 2.018e-3 | 4.88e-4 |
| ratio4-8192 | 640 | 0.102 | 2.034e-3 | 4.88e-4 |

max_abs 4.88e-4 = 输出量级(0.12–0.19)下的**一个 bf16 ULP**,即差异就是 bf16 输出
精度本身。与探针的 rel_fro 1.93e-3 同量级。head_chunk 16 与 8 全部逐位相同。
非有限值 0。

### 2.2 层级门 — PASS,容差沿用原值未放宽

`e0ef`(ratio-128)与 `e0wf`(滑窗)单层 oracle 直接加臂
(`DSV4_PREFILL_SPARSE_BACKEND=tilelang`),4 rank 全部 `accepted=true`、0 errors。
`e0ff` 无 prefill phase(冻结在 start_pos ≥ 8192 的 decode),不适用——ratio-4 的
prefill 层级门另做(§2.3)。

**先验证重构本身无副作用**:torch 臂对冻结基线 `out-e0ef` / `out-e0wf`
**706 / 5280 个指标逐个相等(0 处差异)**——默认路径逐位不变。

tilelang 臂只有 prefill 阶段的指标移动(e0ef 34/706、e0wf 46/5280,**decode 指标 0 处移动**):

| oracle / case | 指标 | 基线 | tilelang | 限 | 用量 |
|---|---|---:|---:|---:|---:|
| e0ef prefill128 | `sparse_control` | 2.47e-5 | **1.718e-3** | 3e-3 | 57% |
| e0ef prefill127 | `sparse_control` | 2.62e-5 | 1.711e-3 | 3e-3 | 57% |
| e0wf prefill128 | `sparse_control` | 1.48e-5 | **1.761e-3** | 3e-3 | 59% |
| e0wf prefill200 | `sparse_control` | 1.25e-5 | 1.732e-3 | 3e-3 | 58% |
| e0wf prefill96 | `sparse_control` | 7.73e-6 | 1.790e-3 | 3e-3 | 60% |
| e0ef prefill128 | `sparse_output` | 1.175e-2 | 1.179e-2 | 3e-2 | 39% |
| e0ef prefill128 | `branch` | 1.250e-2 | 1.254e-2 | 4e-2 | 31% |
| e0ef prefill128 | `output_lora` | 1.272e-2 | 1.277e-2 | 3.5e-2 | 36% |

`sparse_control` 是**直接对拍稀疏核**的那一项,故它是唯一被显著推高的指标
(1e-5 → 1.7e-3),吃掉约 57–60% 的冻结预算,仍留 1.7× 余量。下游各级
(`sparse_output` / `output_lora` / `branch`)几乎不动——bf16 级扰动在后续投影里
不放大。**这是本次最紧的一处,记录在案:若将来再叠别的 prefill 语义变更,
`sparse_control` 的余量要重新算。**

### 2.3 ratio-4 prefill 层级门 — PASS(真实权重,逐层锁步,chunk 8192)

`e0ff` 没有 prefill phase,所以 ratio-4 的 prefill 稀疏核此前只有 E2E 背书。这里
在 C2F bench 里补一个层级门(`--gate-sparse`,`sparse-layer-gate.json`):建两条
lane(torch / tilelang),**逐层锁步**——同一个 hidden 分别喂两条 lane 的 attention,
记录该层 branch 的差,然后**两条链都用 torch 的 branch 前进**,保证每层的差都是
**该层的局部核误差**而不是逐层累积的偏离。

| 层 | 型 | rel_fro | max_abs | \|ref\|max | 非有限 |
|---:|---|---:|---:|---:|---:|
| 0 | ratio128 | 3.501e-3 | 3.13e-2 | 6.50 | 0 |
| 1 | ratio4 | 2.920e-3 | 6.25e-2 | 12.75 | 0 |
| 2 | ratio128 | 3.520e-3 | 6.25e-2 | 10.38 | 0 |
| 3 | ratio4 | 3.688e-3 | 6.25e-2 | 23.38 | 0 |
| 4 | ratio128 | 4.297e-3 | 1.25e-1 | 19.00 | 0 |
| 5 | ratio4 | 4.122e-3 | 2.50e-1 | 48.00 | 0 |
| 6 | ratio128 | **4.677e-3** | 6.25e-2 | 15.31 | 0 |
| 7 | ratio4 | 3.578e-3 | 1.09e-1 | 18.63 | 0 |
| 8 | ratio128 | 4.263e-3 | 1.25e-1 | 30.63 | 0 |
| 9 | ratio4 | **4.231e-3** | 1.88e-1 | 20.88 | 0 |
| 10 | ratio128 | 4.611e-3 | 9.38e-2 | 24.00 | 0 |

两型最差:ratio-4 **4.23e-3**、ratio-128 **4.68e-3**,全部无非有限值。这既补上了
ratio-4 的层级门,也把 ratio-128 的检查从 oracle 的 seqlen ≤ 200 延伸到 **prefill
实际形状 8192**(oracle 覆盖不到的那一段)。branch 级 3–5e-3 与稀疏核级 ~2e-3
相符——输出投影不放大。

> 首版 gate 写错过一次并已修正,记录以免复用时踩坑:ratio-128 / 滑窗层是从**进程级
> env** 读 backend 的,只把 `sparse_backend` 传给 `Ratio4FullPositionAttention`
> 会让"torch 臂"的 ratio-128 层仍跑 tilelang(表现为第 0 层 rel_fro 恰好 0.0,
> 其余层是**累积**而非局部误差,末层能到 2.1e-1)。修正后每层调用前显式翻转 env。

## 3. E2E golden 门 — PASS,472/482(基线 468/482)

`e0ef2e_golden_gate.py` 加 tilelang-prefill 臂(prefill 用 kernel,decode 不变),
16 rank / 双机 / eager HC,8 条 prompt 全量对 D0 golden tokens。

| 臂 | 匹配 | 逐 prompt |
|---|---:|---|
| 冻结基线(eager) | 468/482 | 2, 27, 127, 124, 11, 22, 32, 123 |
| C2F fused indexer | 468/482 | 2, 27, 127, 124, 11, 22, 32, 123 |
| C2F W4A8 | 468/482 | 2, 29, 127, 123, 12, 22, 32, 121 |
| FP8 KV | 467/482 | 2, 28, 128, 124, 12, 22, 31, 120 |
| **tilelang prefill** | **472/482** | 2, **29**, 127, 124, **12**, 22, 32, **124** |

**不劣化,且是目前所有臂里最高的一档**:三条 prompt 各 +1/+2,无一条下降。
10 处分歧全为近平局(top2_gap 中位 0.325、最大 0.562;基线最大 0.712),与冻结的
分歧类型一致。`accepted: true`。

> 注意这**不表示 kernel"更准"**——golden tokens 由 reference MP=8 产生,而 reference
> 本身就用这个 tilelang kernel;prefill 换回 reference 的核,等于在 prefill 侧减少了
> 一处与 oracle 的实现差异。方向合理,但 4 token 的量级仍在近平局翻转的噪声内,
> 不应当作精度结论。

## 4. 吞吐(C2F 同口径重测)

titan064 4 卡、11 层 L11–L21、all-on(W4A8 Marlin MoE + D0b fused indexer)、
iters 5 / warmup 2、host wall + barrier、eager 无 graph。

### 4.1 chunk 8192(每臂 ≥2 轮)

| 臂 | 轮 | input tok/s/stage | attn ratio-4 | attn ratio-128 | attn 合计 | moe | hc | norm | total_instr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| torch(原 launcher 对照) | – | 11,415 | 0.8189 | 0.3288 | 1.1477 | 1.3723 | 0.3135 | 0.0322 | 2.8667 |
| torch | r1 | 11,442 | 0.8161 | 0.3284 | 1.1445 | 1.3620 | 0.3132 | 0.0322 | 2.8528 |
| torch | r2 | 11,415 | 0.8193 | 0.3288 | 1.1481 | 1.3782 | 0.3132 | 0.0322 | 2.8727 |
| **tilelang** | r1 | **14,645** | 0.3613 | 0.1588 | 0.5200 | 1.3706 | 0.3132 | 0.0322 | 2.2371 |
| **tilelang** | r2 | **14,615** | 0.3614 | 0.1588 | 0.5202 | 1.3713 | 0.3133 | 0.0321 | 2.2380 |
| tilelang | gate | 14,601 | 0.3611 | 0.1590 | 0.5201 | 1.3782 | 0.3133 | 0.0322 | 2.2447 |

轮间离散 < 0.3%。**component_walls 对照**(torch = r1/r2/对照 3 轮均值,tilelang = r1/r2 2 轮均值;
`--gate-sparse` 那轮 14,601 作一致性旁证,不入均值):

| 分项 | torch | tilelang | 变化 |
|---|---:|---:|---|
| attention ratio-4(5 层) | 0.8181 | 0.3614 | **2.264×**,−91.3 ms/层 |
| attention ratio-128(6 层) | 0.3287 | 0.1588 | **2.070×**,−28.3 ms/层 |
| **attention 合计** | **1.1468** | **0.5201** | **2.205×**,**−626.6 ms/pass** |
| moe | 1.3708 | 1.3709 | 1.000(未动,如期) |
| hc | 0.3133 | 0.3132 | 1.000 |
| norm | 0.0322 | 0.0322 | 1.000 |
| total_instrumented | 2.8641 | 2.2375 | −626.6 ms |
| **吞吐** | **11,424** | **14,630** | **1.281×** |

### 4.2 chunk 扫点

| chunk | torch tok/s | tilelang tok/s | 比 | attn torch | attn tilelang | attn 比 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 11,260 | 14,404 | 1.279× | 0.2798 | 0.1333 | 2.10× |
| 4096 | 11,415 | 14,468 | 1.267× | 0.5570 | 0.2590 | 2.15× |
| 8192 | 11,424 | 14,630 | 1.281× | 1.1468 | 0.5201 | 2.20× |

attention 加速随 chunk 单调上升(候选数与 O(s²) 项随 chunk 增长,kernel 的相对优势
变大);端到端比值基本平坦,因为分母被 MoE 主导(见 §4.3)。

### 4.3 重要口径警告:本轮 MoE 落在"慢分支",不能直接对 16.6k

`../reattribution/` 记录过 MoE 的**双模行为**(同一 kernel/几何,49.8 ms/层 vs
131 ms/层,归因为分配器状态,列为鲁棒性缺陷)。**本轮所有运行都落在慢分支**:
moe 1.362–1.378 s(125 ms/层),而重归因那 4 轮是 0.483–0.485 s(44 ms/层)。

这不是本竖条造成的:

- attention 分量与重归因**逐项吻合**(ratio-4 0.8181 vs 0.8201、ratio-128 0.3287 vs
  0.3294,差 < 0.5%);
- 用**完全未改动的原 launcher**(`run_c2f_prefill_titan.sh 8192 w4a8 fused`,不带
  `--sparse-backend`)复跑,同样是 11,415 tok/s / moe 1.3723——慢分支与本次改动无关;
- 两个臂的 moe 桶彼此相差 0.1%,故**臂间 A/B 不受影响**。

> **[2026-07-21 更正,由第二十二竖条 `../moe-alloc/` 证伪并定案]**
> 本节下面那条"分配器线索"**是错的**,原文保留以便追溯。第二十二竖条实测:慢分支
> 22 次 MoE 调用的分配器计数器(`num_alloc_retries` / `device_alloc` / `device_free`
> / `ooms`)**全部恰好为 0**,`max_memory_reserved` 只反映
> `PYTORCH_CUDA_ALLOC_CONF` 有没有设,与快慢无关(有快指纹却慢、慢指纹却快的反例)。
> **真因是 NCCL 传输选择**:GPU0–3 处于 NODE 距离,NCCL 默认 P2P level 不覆盖,
> TP4 MoE 的两个集合通信退到经主机中转的 `SHM/direct`(4.12 GB/s);
> `NCCL_P2P_LEVEL=SYS` 令 `isAllDirectP2p` 翻为 1 → 23.79 GB/s
> (all-gather 48.84→8.46 ms、reduce-scatter 46.99→8.65 ms),双模之差 100% 落在这里。
> **本竖条写的 4 个 C2F launcher 正是 runtime/ 里唯一漏掉该 export 的**(我从
> `run_c2f_prefill_titan.sh` 抄的 ENV_BASE 就缺它;而 `run_e0e2e_tilelang_arm.sh`
> 抄自 E2E launcher,带了完整 NCCL env,故 E2E 臂不受影响)。四个 launcher 均已补上。
>
> 修完后同口径**实测**:torch 臂 **17,022**、tilelang 臂 **25,308** input tok/s/stage
> (三轮离散 0.03%),即下面"折算 24,268"**兑现为实测 25,308(+4.3%)**。
> 关键佐证:修复后 attention 桶 **0.5197–0.5202**,与本竖条测的 **0.5201 完全一致**
> ——§4.1/§4.2 的 attention 分量与 §5 的预测偏差结论**不受影响**,受影响的只有
> 被 MoE 污染的端到端分母。

分配器侧的观察(**已证伪,见上**):本轮
`max_memory_reserved` **20.50 GiB**、`max_memory_allocated` 20.41 GiB(reserved 只比
peak 高 0.09 GiB);重归因那轮 reserved **23.98 GiB**、allocated 20.42 GiB(高 3.57 GiB)。
即当时 allocator 把 expandable segment 撑到接近整卡并留住了,MoE 每层数 GB 的 fp32
combine 临时量能从缓存里拿;本轮没撑起来,于是每次都要重新映射。

**按快分支 MoE 折算**(把重归因的 0.4837 s MoE 桶代入本轮的 pass 组成):

| 臂 | 本轮 pass | 代入快 MoE | 折算 tok/s |
|---|---:|---:|---:|
| torch | 2.8641 | 1.977 | **16,575** |
| tilelang | 2.2375 | 1.350 | **24,268** |

torch 侧折算出 16,575 ≈ 冻结基线 **16,602**(差 0.2%),说明这个代入是自洽的;
据此 tilelang 的同口径值 **≈ 24.3k input tok/s/stage(1.46×)**。这是**算术折算,
不是实测**——直接实测值是 §4.1 的 11,424 → 14,630。

> **[更正]** 该折算已由第二十二竖条兑现为实测:修掉 `NCCL_P2P_LEVEL` 后同口径
> **torch 17,022 / tilelang 25,308**(见 `../moe-alloc/`),折算 24,268 兑现 +4.3%。

## 5. 与 6.49× 微基准预测的偏差与归因

微基准预测:ratio-4 每层省 ~90 ms × 5 + ratio-128 每层省 ~27 ms × 6 = **~611 ms/pass**。

**实测省 626.6 ms/pass — 偏差 +2.6%,预测基本命中。**

| 项 | 预测 | 实测 |
|---|---:|---:|
| ratio-4 每层节省 | ~90 ms | **91.3 ms** |
| ratio-128 每层节省 | ~27 ms | **28.3 ms** |
| pass 总节省 | ~611 ms | **626.6 ms** |

**"为什么桶只有 2.20× 而不是 6.49×"**——这不是偏差,是口径差:6.49× 是**稀疏核本身**
的比值,而 `attention_*` 桶装的是整层 attention:wq_a/wq_b/wkv/wo_a/wo_b 投影、
逐头 RMS、RoPE 与逆 RoPE、FP8 QDQ、lightning indexer(fused)+ top-k、overlap
compressor、ring 写入。只有稀疏核被替换。反推:

- ratio-4 桶 torch 163.6 ms/层,减去探针的 torch 核 106.07 ms → **非稀疏部分 57.5 ms/层**;
- tilelang 桶 72.3 ms/层 − 57.5 = **反推稀疏核 14.8 ms/层**,对探针的 16.36 ms。

即接入后的稀疏核**比探针臂还略快**:探针用 `torch.cat` 拼 4 个 head 块,包装改成
预分配 + `copy_`;这点省出的量,足以盖过包装新增的那一次校验归约(对
[1,8192,640] int32 做几遍扫描,~0.1 ms 量级)。**校验不是热点,可以一直开着。**

按 Amdahl:attention 从 58% 降到本轮口径下的 23.2%(0.5201/2.2375),prefill 的头号
瓶颈**已经换人**——现在是 MoE(慢分支下 61%,快分支下也仍是最大单项)。

## 6. 意外发现(原样记录)

1. **reference kernel 在全 padding 行上产 NaN**(§1.1),torch 版产 0。现网 prefill
   三个调用点都拼了因果 window,不会触发;但任何未来的"纯 compressed 候选"路径
   (例如去掉 window 部分的实验)会直接踩上。包装已兜底。
2. **reference kernel 只认 `-1` 作 padding**,`-2` 会被当有效索引做回绕 gather
   (实测 rel_fro 0.30)。现网产生的 padding 恰好都是 `-1`,属于**未被文档化的隐式契约**。
3. **sm89 上 head_chunk 32 也过不去**(需 104448 B,仅比 101376 B 多 3 KB)。
   A4F 只测了 h=64 撞墙,这里补上边界:16 是上限,再往上一档就失败。
4. **e0wf/e0ef 的 `sparse_control` 冻结限 3e-3 是本次最紧的门**(用掉 57–60%)。
   其余各级都有 ≥1.7× 到 3× 余量。
5. **MoE 双模缺陷今天落在慢分支**(§4.3),使端到端分母被污染。
   ~~且与 `max_memory_reserved` 是否撑到接近整卡高度相关~~ —— **该归因已被第二十二
   竖条证伪**(`../moe-alloc/`):分配器计数器全 0,真因是 NCCL 退到 SHM(4.12 GB/s),
   `NCCL_P2P_LEVEL=SYS` 修复后 tilelang 臂**实测 25,308 tok/s**。**本竖条的 4 个
   C2F launcher 漏了该 export,是这次踩坑的直接原因**——新写 launcher 时应从
   runtime/ 里带完整 NCCL env 的那份抄,而不是从 `run_c2f_prefill_titan.sh` 抄。
6. **e0ff 不能加 prefill 臂**:它冻结在 start_pos ≥ 8192 的 decode plan 路径,
   根本不经 `torch_sparse_attention`。ratio-4 的 prefill 从来只有 E2E 背书,
   本竖条补了一个真实权重的逐层锁步门(§2.3,两型最差 4.23e-3 / 4.68e-3)。

## 7. 产物

| 文件 | 内容 |
|---|---|
| `c2f-tilelang-op-gate.json` | 单算子门(11 例 × 2 head-chunk + 边界例 + smem 墙探测) |
| `oracle-e0ef-tilelang-summary.json` | ratio-128 单层 oracle,tilelang 臂,4 rank |
| `oracle-e0wf-tilelang-summary.json` | 滑窗单层 oracle,tilelang 臂,4 rank |
| `e2e-tilelang-prefill.json` | E2E golden 门 472/482 |
| `c2f-tl-chunk*-{torch,tilelang}-*.json` | 吞吐各臂(含 component_walls) |
| `sparse-layer-gate.json` | ratio-4/ratio-128 真实权重逐层锁步 A/B |

代码:`runtime/dsv4_direct/ops/tilelang_sparse.py`、三个注入点、
`runtime/c2f_tilelang_sparse_gate.py`(单算子门)、
`runtime/c2f_prefill_stage_bench.py`(`--sparse-backend` / `--sparse-head-chunk` /
`--gate-sparse`)、launcher `run_c2f_tilelang_gate.sh` / `run_c2f_tilelang_oracles.sh` /
`run_e0e2e_tilelang_arm.sh` / `run_c2f_tilelang_bench.sh`。

收尾:两机 GPU 均已回到 1 MiB(见各 launcher 日志尾部 nvidia-smi 快照)。
