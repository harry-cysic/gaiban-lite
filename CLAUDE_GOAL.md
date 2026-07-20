# Claude 长期目标

在 titan064/titan065（2 台 8×RTX 4090，`ssh 10.234.1.64 / 10.234.1.65`）上实现
DeepSeek-V4-Flash（284B/13B，FP4+FP8）的高吞吐推理系统，达到
`docs/feasibility-v4-flash-2x8x4090.md` 中的预估性能区间（decode-only roofline
~15–25k output tok/s @8K；8K/1K 单池 aggregate ~3.2–4.2k output tok/s）。若指标所依赖的
容量或性能假设确实不成立，必须用可复现、可归因且通过验收的实验结果证明并修正容量模型，
不能仅因暂时未达到而降低目标。

## 长期原则

- **系统形态**：PP4 × TP4（每 socket 一个 TP4 super-stage）、decode/prefill 均
  DP-attention、routed experts 走 Marlin MXFP4（decode W4A16 / prefill W4A8-FP8）、
  per-expert intermediate-TP 摆放、direct runtime（零 SGLang 运行时依赖）。目标不是
  通用推理框架或生产 API。形态变更须先用实验证伪当前选择。
- **语义对齐**：经常逐段对照 `reference/inference/model.py` 的完整数据流与数值语义，
  尤其 Compressor/Indexer/稀疏 Attention、hash+noaux_tc Gate/MoE、Hyper-Connections、
  head 与 MTP，以及 Flash 特有几何（43+1 层、256 experts、64 heads、topk 512、
  L0/L1 纯滑窗层、前 3 层 hash 路由）。布局、kernel 和执行顺序可以改，语义变化必须
  显式记录并由独立 oracle（reference 实现 golden-token 对拍）验证。
- **复用优先**：`../gaiban` 的 kernel、runtime 骨架（`experiments/E0-direct-runtime/
  dsv4_direct/`）、实验方法论与校准数据是本项目的基础资产；换几何复用优先于重写，
  且对已被 gaiban 判死的路线（tilelang fp4_gemm 调优、attention activation 原生 FP8、
  FlashInfer sparse MLA on sm89）不再重复投入。
- **分级推进**：operator → 单 TP4 stage → 单机 TP4×PP2 → 双机 PP4 → serving/benchmark；
  每级先过 correctness、状态生命周期和资源恢复 gate，再接受性能结果。数值质量沿用
  冻结质量门方法论：attention 保持 BF16 计算 + weight-only FP8；FP8 KV 仅作容量选项，
  启用前须在 Flash 几何上重新验证速度与质量。
- **性能归因**：同条件、可重复的 causal A/B。结果必须绑定 source、kernel、checkpoint、
  环境、拓扑和 workload；失败、不完整或无法独立重建的 artifact 不产生性能结论。
  严格区分 roofline/proxy/open-loop 与完整模型 closed-loop E2E，表述不得混用。
- **实验组织**：沿用 gaiban 惯例——每个实验一个 `experiments/<ID>-<name>/` 目录，
  README 记录动机、方法、结论与 artifact 路径；有效进展及时更新根 `README.md` 顶部
  状态段并做范围清晰的 Git commit；大型结果 artifact 不进 Git。机械执行/收集与
  文档同步类子任务优先派给降档 agent 类型 `runner`/`scribe`（定义见
  `.claude/agents/`）；产出结论（实验解读、归因、去留判断）的子任务不降档。

## 无人值守硬约束

- **不得删除或改动 Pro（gaiban）资产**：titan 两台 `~/Workspace/` 下的
  `dsv4-checkpoint-stages`、`dsv4-runtime-package`、`dsv4-sglang-*` 等仍在使用；
  磁盘余量足够，禁止为腾盘清理它们。
- **不得破坏 venv 钉住版本**：`~/Workspace/venvs/sglang` 中 tilelang==0.1.8、
  flashinfer==0.6.12、tokenspeed-mla==0.1.6、llguidance<0.8 为 sglang 钉住版本；
  vllm 0.22.1 仅作 Marlin kernel 库使用，pip 对此的依赖警告是预期内的，不要"修复"它。
  需要新依赖时优先另建 venv。
- **机器边界**：实验只在 titan064/065 上跑；earth（权重源）只读；不触碰 dsv4exp
  （titan052，Pro 实验机）。权重本地副本在 `~/Workspace/DeepSeek-V4-Flash/`。
- **环境事实**以 `docs/feasibility-v4-flash-2x8x4090.md` 附录 B 为准（pip 走
  huaweicloud 镜像、titan 未装 ping、GitHub 经本地工作站中转等），遇到与之矛盾的现象
  先核实再改文档，不要在 goal 文档里累积临时结论。

持续从当前根 `README.md`、`docs/feasibility-v4-flash-2x8x4090.md`（含 §7 路线图与
附录）和最新实验目录的 handoff 获取阶段状态与下一步；不要把某个临时故障、单次实验
方案或近期日程固化进本文档。
