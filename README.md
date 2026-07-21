# gaiban-lite

DeepSeek-V4-Flash（284B/13B）在 2×8×RTX 4090 上的推理系统。官方推理参考实现在
[`reference/`](reference/)；可行性评估、实施方案与性能预估见
[`docs/feasibility-v4-flash-2x8x4090.md`](docs/feasibility-v4-flash-2x8x4090.md)。
实验方法与 kernel 资产来源：`../gaiban`（DeepSeek-V4-Pro on 64×4090）。
长期目标与无人值守约束见 [`CLAUDE_GOAL.md`](CLAUDE_GOAL.md)。

## 当前状态（2026-07-21）

**权威目标定义见 [`docs/TARGET-v4-flash.md`](docs/TARGET-v4-flash.md)**（两级验收 +
模式矩阵 + 已证伪假设清单）。本段只给一句话现状与索引；数字的权威副本在该文档的
"实测"列，推导与证据在各实验 README 与 git history。

### 已放行的实测（裸引擎，均过 D0L 长门 494/512，容差从未放宽）

| 项 | 实测 | 口径 | 实验 |
|---|---:|---|---|
| 聚合 decode @8K | **8,733** tok/s | 16 卡 PP4、FP8 KV、bl72×mb4、graph | E1F/E1IF |
| 聚合 decode @2K | **9,656** tok/s | 同上，bl128×mb2 | E1IF/MTP |
| prefill | **28,622** input tok/s | 16 卡、whole-8192、tilelang 稀疏核 + 融合 QAT 核 | C2F/C4F |
| 混合单池 8K/1K | T=**2,538** tok/s | 裸引擎折算 | — |
| 单路 decode | 27.5 tok/s（+MTP ~38） | 16 卡、B=1、graph | E1F/MTP |

语义:满配 43 层 PP4×TP4 双机 E2E 对 golden tokens 97.1%（分歧全为近平局）;
每个语义变更（fused HC、FP8 KV、MTP、tilelang attention、融合 QAT 核）均过冻结门。

### 阶段

- **Phase 0–3 完成**:环境/标定/oracle → kernel regear → dsv4_direct 全量移植
  （契约、加载、三层型 attention、MoE、整层、superstage+stateful graph、单机 PP2、
  双机 PP4、满配 E2E）→ 吞吐与容量前沿。
- **Phase 4 进行中**:prefill 杠杆（已放行 tilelang 稀疏核、融合 QAT 核；已否决
  HC 融合与集合重叠）、chunked prefill（能力已具，容量杠杆而非速度杠杆）。
- **最大空白**（按 TARGET §2 优先级）:M4 延迟模式（差 8–14×，8 卡形态从未跑过）、
  M5 长上下文（零覆盖）、serving 折扣验证。

### 实验索引

`experiments/` 下每个目录一个实验，README 记动机/方法/结论/artifact:
B1·B2 标定 | A0 契约 | D0·D0L golden oracle | A3F·A4F·A5F·A6F kernel |
C1F 集成 block | E1F 吞吐与容量前沿 | C2F·C3F·C4F prefill |
`runtime/` 是 direct runtime 与全部门脚本（非安装包，靠 rsync 到 titan 运行）。
