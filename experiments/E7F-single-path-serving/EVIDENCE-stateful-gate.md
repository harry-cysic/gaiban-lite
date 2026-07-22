# E7F 步 3 质量证据包：stateful serving 路径的 −4 是噪声还是系统性回归？

- 日期：2026-07-22 · 状态：**证据，非裁决**。
- **本文只汇总证据。§1.3 门的放行/迁移、默认翻转，均是人的决定，本 session 不作、
  不迁门、不改默认、不放宽任何容差。** 与 E6F 同一纪律。
- 待判命题：v2 全集上 stateful **610/640** vs eager 614（**−4**）——
  是 serving 路径的系统性回归，还是集中在 v2 短 prompt 的抽样噪声（如 E6F 的 −3）？
- **载荷性纪律**：单路 serving 操作点 = 短/未饱和区（Blocker A 解锁的那个）。
  v2 上 flips 在短 prompt、4096/8192 不变，**长短混合的聚合净值会把短区回归
  用不变的长区平均掉**。故**每条证据按 regime 分列**（unsat/serving vs sat）。
  regime 定义：decode 位 < 2047（index 未饱和）= unsat；v2 里 1024 prompt 全程 unsat，
  2048+ 全程 sat。

reuse-before-cite：**eager 臂逐字复现冻结 v2 基线 614/640**（已记于 step3-full）。

---

## 证据 1：新 stateful 路径是确定性的（3 轮，spread 0）

冻结门有过"复跑逐 prompt 一致"的背书；这条新代码路径此前没有。
3 轮同配置（det1/det2/det3，v2 的 1024/2048/4096 子集，8 prompt）：

| 臂 | 3 轮 predicted_tokens 逐字一致 | 分数 | spread |
|---|---|---|---|
| eager | **是** | 491/491/491 | **0** |
| **stateful** | **是** | 487/487/487 | **0** |
| perturbed | **是** | 488/488/488 | **0** |

另：det1 与 step3-full 在重叠 prompt(0–7) 上 stateful 预测**逐字相同**（两次独立 16 卡跑）。
**⟹ −4 是真实可复现的，不是测量噪声。**（§9.5：spread 0。）

---

## 证据 2：独立重抽集（与 v2 不相交），按 regime 的符号检验 —— **符号反转**

oracle-indep（15 prompt / 927 位，独立重抽，6×1024 + 6×2048 + 3×4096，与 v2 实测不相交）：

| 臂 | 总分 | unsat(1024, serving) | sat(2048/4096) |
|---|---|---|---|
| eager | 892/927 | 334/351 | 558/576 |
| **stateful** | **900/927（+8）** | **338/351（+4）** | **562/576（+4）** |
| perturbed | 894/927（+2） | 336/351（+2） | 558/576（0） |

**在独立集上 stateful 反而比 eager 高 +8，两个 regime 都是正的（+4/+4）。**
即 **v2 的 −4 与 indep 的 +8 符号相反** —— 典型抽样噪声特征（非单向系统性）。

符号检验（回归 = eager 对、该臂错；两侧）：

| 对照 | regime | 该臂-worse | better | n | sign-test p |
|---|---|---|---|---|---|
| eager vs stateful | **unsat(serving)** | 1 | 5 | 6 | **0.219** |
| eager vs stateful | sat | 2 | 6 | 8 | **0.289** |
| eager vs perturbed | unsat | 2 | 4 | 6 | 0.688 |
| eager vs perturbed | sat | 6 | 6 | 12 | 1.000 |

**全部不显著**（p ≥ 0.219，含决定 serving 质量的 unsat 区）。差异与"平衡"不可区分。
独立集包络：**eager / stateful / perturbed 三臂全部 1.178104（逐位相同）**——
stateful 在独立集上**不抬高包络**。

---

## 证据 3：v2 包络越界的穷举分类 —— 唯一一处，且是共享失配

对 v2 全集 stateful 每一个 gap > 0.9595 的失配，问：是 stateful 引入的
（eager 在包络内、stateful 越界 = 真质量回归），还是共享的（两臂都失配、
stateful 只是把 gap 加宽 = 基线属性被求和序放大）？

| regime | >0.9595 越界数 | 其中 stateful-引入 |
|---|---|---|
| unsat(serving) | **1** | **0** |
| sat | 0 | 0 |

**全集只有 1 处越界**：p2(1024) step36，gap 1.127——**共享失配**：
eager 与 stateful **预测同一个错 token 8842**（都漏 golden 59819），
eager gap 0.841、stateful 1.127（top1 36.96→37.23，ULP 经全栈放大）。
**stateful-引入的越界 = 0。** 独立集上包络三臂逐位相同（证据 2），同结论。

---

## 证据 4：1-ULP 内在敏感度对照（E0hf 法）—— serving 区打平

对照臂 `perturbed`：把 prefill 后的状态**每个 bf16 元素挪 1 ULP 远离零**
（E0hf `view(int16).add_(1 where≠0)`），再走**非 stateful** decode。
它给出"一次 ~1 ULP 扰动应当翻多少近平局"的基准。3 轮一致：

| regime | eager | stateful(Δ) | **perturbed 1-ULP 对照(Δ)** |
|---|---|---|---|
| **unsat(serving)** /192 | 187 | 185 (**−2**) | 185 (**−2**) |
| sat /320 | 304 | 302 (−2) | 303 (−1) |

**决定 serving 质量的 unsat 区：stateful 翻的近平局数 = 1-ULP 对照，一模一样（各 −2）。**
sat 区 stateful −2 vs 对照 −1（多 1 个；对照是单方向单次实现）。
独立集上符号检验里 perturbed 也双向、不显著（证据 2）。
**⟹ stateful 的近平局翻转落在 1-ULP 求和序敏感度之内，是 reorder 不是新 error。**

---

## 汇总（证据，不含裁决）

四条独立证据同向：

1. **确定性**：3 轮逐字一致、spread 0——差异真实可复现，非测量噪声。
2. **独立集符号反转**：v2 −4 → indep **+8**（serving 区 +4），符号检验全不显著。
3. **包络**：v2 唯一越界是**共享失配**、stateful-引入 = 0；独立集包络三臂逐位相同。
4. **1-ULP 对照**：serving 区 stateful 翻转数 = 1-ULP 扰动，打平。

**指向**：stateful serving 路径对 eager 的偏差是**抽样噪声级的 ULP 求和序**
（§7.9 ratio-128 + A 的 padding），**不是 serving 区的系统性质量回归**——
性质与结论都与 E6F 的 −3 同。

⚠️ **但这是证据，不是放行。** 是否按 §1.3 放行 serving 路径、是否像 E6F 那样
迁移门基准、是否翻默认——**都是人的决定**。本 session **未迁门、未改默认、未放宽容差**。
artifact：`results/{step3-full,evidence-det1,evidence-det2,evidence-det3,evidence-indep}/`。
