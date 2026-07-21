# E5F — decode 侧融合 KV-latent FP8 QAT 核

第三十一竖条。接着 E4F 往 elementwise 尾巴里走第二块。

E2F §5b 的待攻清单把 `raw_kv_done` 排在第三（30.1 µs/层非 GEMV），但它有一条
E4F 的 indexer 链没有的性质：**它在每种层型上都跑**（ratio-4、ratio-128、
纯滑窗层读数都是 30.1 µs/层），因为三者共用同一条
`rms_norm → rope → fp8_quant_dequant` 的 KV latent 链。E4F 那条链只在 21 个
ratio-4 层上有；这条在全部 43 层上都有。

---

## 1. 目标与边界

本竖条只融合链里的 **`fp8_quant_dequant`** 一段（约 13 个 eager 核）：

```
grouped = value.float().reshape(..., -1, 64)
amax    = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4)
scale   = exp2(ceil(log2(amax / 448)))
q       = (grouped / scale).clamp(-448, 448).to(float8_e4m3fn)
out     = (q.float() * scale).to(bf16)
```

**不碰 `rms_norm` 与 rope**：把它们一起融需要 norm 权重与 rope 表，逐位难度
高一个量级，是另一个问题。先取确定能拿的那块——与 E4F 同样的分级做法。

### 1.1 只接 decode，不接 prefill——这是查出来的，不是省事

prefill 路径（`ratio4_fullpos.py`）有同样的写法，而且 C4F 的分相 profile
早就单独给这条链打了标记。查冻结产物即可定案：

| 链 | prefill（C4F，8192 行/层 72.07 ms） | decode（E2F/E5F，B=1） |
|---|---:|---:|
| `kv_fp8_qdq`（本竖条） | 0.1004 ms = **0.14%** | 15.1 µs/层 = **2.25%** |
| `idx_hadamard_fp4`（E4F 那条） | 29.75 ms = **41.3%** | ~66 µs/层 = 4.3% |

**同一段代码，两个 regime 的量级关系完全相反。**prefill 下 8192 行让这条链
变成真正的带宽工作，占比可以忽略；decode 下 448 个值让它退化成十几次
kernel 启动，于是显著。所以本竖条**只接 decode 调用点**，prefill 保持原状。

这也是 E2F"尾巴按核数收费"机制的一个反向验证：**launch-floor 绑定是
decode 现象**，同样的代码在 prefill 尺度上没有问题。

## 2. 实现要点

### 2.1 按"组"而不是按"行"分块

decode 调用点量化的是 `raw_latent[..., :-rope_dim]`，宽 **448 = 7 × 64**。
7 不是 2 的幂，`(ROWS, GROUPS, GROUP)` 的 Triton block shape **不存在**。
把张量摊成 `(-1, 64)`、让每个 Triton row 恰好是一个量化组，约束就消失了，
kernel 也不再关心一行里有几个组。

### 2.2 就地写 strided 前缀，而不是 gather/scatter

调用点的写法是 `latent[..., :448] = qat(latent[..., :448])`——在一个 512 宽的
行上操作一个 strided 前缀。若 kernel 只支持连续张量，就要先 `.contiguous()`
再写回，**多出的两个 elementwise 核正是这次融合想去掉的东西**。所以主入口是
`fused_kv_fp8_qat_prefix_(tensor, width)`：带 `ROW_STRIDE` 直接读写前缀，
整条链保持一个 kernel，且与 eager 的就地语义一致。

读写同一块内存无冒险：每个组由单一 Triton row 处理，组内 amax 在任何 store
之前就已从寄存器算出。

### 2.3 逐位

沿用 C4F/E4F 的纪律：目标是**逐位相等**，不是接近。

- `clamp_min(1e-4)`、`/448.0`、`clamp(±448)` 用同样的字面量；
- `exp2(ceil(log2(·)))` 用 ATen 下降到的同一组 libdevice 函数；
- **E4M3 往返是唯一"按构造"而非"按代数"对齐的一步**——Triton 的
  `float8e4nv` cast 与 ATen 的 `.to(torch.float8_e4m3fn)` 必须在
  round-to-nearest-even 上一致。这正是 `bitwise_selfcheck` 每次都要在真实
  数值范围上重验它、而不是信任它的原因。

**自检结果：18/18 逐位相等**（9 例 out-of-place + 9 例 prefix 就地），
`max_abs_diff` 全为 0.0，前缀之外的尾部逐位不变。覆盖三种形状
（decode `[1,1,448]`、prefill `[1,8192,448]`、非整行 `[3,17,64]`）
× 三个输入尺度（×1、×0.02、×50），后者是为了让每组的量级跨若干个指数、
真正压到 E4M3 的舍入边界上。

## 3. 接入

`Ratio4TorchAttention` / `Ratio128TorchAttention` / `WindowTorchAttention`
各加 `kv_qat_mode`（默认 `"ref"`，env `DSV4_KV_FP8_QAT=fused` 或构造参数切换）。
调度函数 `attention.kv_fp8_qat_prefix` / `kv_fp8_qat` 在**任何 kernel 不支持的
情况下回落 eager**（组大小不符、非连续、宽度不是 64 的倍数），
所以这个开关不可能"静默不生效地改变行为"——要么走融合核，要么走原路。

覆盖的 decode 调用点：ratio-4 的两条 decode 路径 + ratio-4 compressor、
ratio-128 与滑窗层的 `_nope_control`。

## 4. 结果

### 4.1 层内成对交替 A/B

口径与 E4F 同：两条常驻 lane、每步背靠背、逐步交换先后顺序，每步顺带
`torch.equal` 对拍作为层内数值门。titan065，stage 0（11 层，三种层型齐全），
B=1，max_seq 3328，3 轮 ×160 步。

| 项 | 值 |
|---|---:|
| 基线 lane p50 | 7.3625 ms |
| 变体 lane p50 | **7.1968 ms** |
| 差 | **−0.1657 ms（−2.25%）** |
| 每层 | **−15.1 µs** |
| 轮间离散 | base 0.359% / variant 0.362%（<1%，§9.5） |
| **层内逐位** | **480/480 步，max_abs_diff 0.0** |
| 作用层 | L0–L10 **全部 11 层**（三种层型都有这条链）✓ |

**注意基线口径**：E4F 的融合 indexer QAT 已是默认，所以这里的基线 lane
本身就带 E4F 的收益，**−2.25% 是叠加在其上的增量**。两个杠杆可叠加。

每层 15.1 µs 与预期一致：`raw_kv_done` 的 30.1 µs 非 GEMV 里，本次只融了
`fp8_quant_dequant` 那约一半的核，`rms_norm`/rope 仍是 eager。

### 4.2 全模型闭环（16 卡 PP4，E1F 冻结脚本）

基线是 **E4F 翻默认之后**的 28.402 tok/s（即已含 E4F 收益），加
`DSV4_KV_FP8_QAT=fused`：

| 项 | 基线（E4F 默认） | +E5F | 差 |
|---|---:|---:|---:|
| **单路 decode** | 28.402 tok/s | **29.055 tok/s** | **+2.30%** |
| stage 0 replay | 7.9913 ms | 7.7618 | −0.2295 |
| stage 1 replay | 7.9957 | 7.8162 | −0.1795 |
| stage 2 replay | 8.1396 | 8.0025 | −0.1371 |
| stage 3 replay（10 层） | 7.2616 | 7.0984 | −0.1632 |
| 四级合计 | 31.3882 | 30.6790 | **−0.7092** |

每层 **−16.5 µs**（709 µs ÷ 43 层），与层内 A/B 的 −15.1 µs 一致。

**累计（E4F + E5F）：27.740 → 29.055 tok/s，+4.74%**，两步都是逐位的。

⚠️ 同 E4F §3.3 的口径声明：这两次是独立的 16 卡作业（跑间离散约 1.0%），
**因果结论建立在 §4.1 的成对交替 A/B 上**，本节只确认它在闭环里兑现。

### 4.3 D0L 长门

与 E4F 同：改动是逐位的，但"逐位所以门必过"是推理不是实测，仍跑一次冻结长门。

| 项 | 冻结基线 | +E5F |
|---|---:|---:|
| 命中 | 494/512 | **494/512** |
| match_rate | 0.964844 | **0.964844** |
| max `top2_gap` | 0.959503173828125 | **0.959503173828125** |
| median gap | 0.3458061218261719 | **同** |
| 八条 prompt 的首个失配记录 | — | **逐条逐字段相同** |
| accepted | True | **True** |

### 4.4 prefill 侧：默认翻转前必须先问的一件事

`_nope_control` 由 ratio-128 与滑窗层的 `__call__` 共用，而
`c2f_prefill_stage_bench.py` 的**非 ratio-4 层正是走 `new_attention()` +
`__call__`**——也就是说，把 `kv_qat_mode` 默认翻成 `fused` **会同时改到
prefill**，不只是 decode。这一点是查调用图查出来的，不是想当然。

两个理由说明方向上不会变差，但都不能替代实测：

1. C4F 冻结的分相 profile 给出该链在 prefill 只占 0.14%（§1.1）；
2. 融合核对同样的字节做**一读一写**，而 eager 链要材料化多个 FP32 临时量
   ——在带宽绑定的 prefill 尺度上只可能更省，不可能更费。

⚠️ 但注意：**这个效应量（≲0.14%）低于该 bench 的跑间离散**，所以它在 prefill
上"不可能被 A/B 分辨"——这既意味着不可能是有意义的回归，也意味着**任何声称在
prefill 上测到收益的说法都不可信**。故只做回归确认，不报 prefill 收益。

**实测（单 stage prefill bench，chunk 8192 / w4a16 / fused indexer，两臂各自
独立 out-dir）**：

| 臂 | 见证 `resolved_kv_qat_mode` | stage p50 | input tok/s (dp4) | ratio-128 分量 |
|---|---|---:|---:|---:|
| ref | `ref` | 1.9897 s | 16,468.6 | 0.3288 s |
| fused | `fused` | 1.9922 s | 16,448.0 | **0.3289 s** |

差 +0.13%，在噪声内；**最直接的证据是 ratio-128 分量墙逐位量级不变
（0.3288 vs 0.3289）——受影响的层正好在这个桶里**。判定：prefill 无可测影响。

另有一个精度补充，由见证读出：prefill 的 ratio-4 层走
`Ratio4FullPositionAttention`（只有 `indexer_qat_mode`、没有 `kv_qat_mode`），
**所以本改动在 prefill 侧只触及 ratio-128 与滑窗层**，比"会改到 prefill"
这个粗判更窄。

### 4.5 两次踩坑（都已写入 TARGET §9）

1. **两个臂写了同一个文件**。`run_c2f_prefill_titan.sh` 的输出名只由
   `chunk/moe/indexer` 决定，两个只在 env 上不同的臂**互相覆盖**，最后两份
   "结果"字节完全相同——与"处理无效"同形，差点据此误判。
   顺带还覆盖了 C2F 的**冻结产物** `out-c2f/c2f-chunk8192-w4a16-*.json`
   与同名 log（已 `git restore` 还原）。已给该脚本加 `C2F_OUT_DIR` 覆盖。
2. **处理组见证**。给 `c2f_prefill_stage_bench.py` 加了
   `dsv4_kv_fp8_qat_env` + `resolved_attention_modes`——与该文件里第 22 竖条
   为同一原因加的 allocator 记录是同一条教训，隔了九个竖条又踩一次。

### 4.6 已设为默认

证据链完整且 prefill 无可测影响，故**默认改为 `fused`**；
`DSV4_KV_FP8_QAT=ref` 可回退。翻默认后不带 env 再跑一次 E1F：

| 项 | 值 |
|---|---:|
| tok/s p50（无 env） | **28.992** |
| 对照：带 env | 29.055（差 0.2%，噪声内） |
| 对照：E4F-only 基线 | 28.402 |
| accepted | True |

⚠️ **见证本身有个洞，已补**：E4F 给 `e1f_full_decode_bench.py` 加的
`attention_modes` 只记了 `indexer_qat_mode` / `index_score_mode` 两个 key，
**没记 `kv_qat_mode`**——所以本次的处理组在 E1F artifact 里看不见。
本次是靠"28.992 明显高于 E4F-only 的 28.402"间接确认的，够用但不该如此。
已把 key 列表扩到含 `kv_qat_mode` / `nope_quant_mode`。
教训：**见证也会有覆盖不全的问题；加一个开关时要同时把它加进见证列表。**

## 5. Artifact

| 路径 | 内容 |
|---|---|
| `../../runtime/dsv4_direct/ops/kv_fp8_qat.py` | 融合核（含 prefix 就地变体与逐位自检） |
| `../../runtime/dsv4_direct/attention.py` | `kv_fp8_qat` / `kv_fp8_qat_prefix` / `resolve_kv_qat_mode` |
| `../../runtime/e2f_decode_phase_probe.py` | `--ab-variant kv_fp8_fused` |
| `results/micro/bitwise_selfcheck.json` | 18/18 逐位自检 |
| `../E2F-decode-latency-profile/results/out-e2f-kvqat-ab/` | 层内成对交替 A/B（480 步逐位） |
| `results/e1f-bl1-kvfused/` · `results/e1f-bl1-defaultkv/` | 闭环（带 env / 翻默认后无 env） |
| `results/d0l-kvfused/` | D0L 长门（与冻结基线逐位一致） |
| `results/prefill-regression/{ref,fused}/` | prefill 回归两臂（含处理组见证） |
