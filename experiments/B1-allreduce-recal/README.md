# B1-allreduce-recal：titan064/065 机内 TP allreduce 标定复跑

日期 2026-07-20。Phase 0 前置项。动机：验证 titan064/065 机内 NCCL allreduce 与 gaiban B1（dsv4exp 实测）锚点一致，并建立 Flash 几何（hidden=4096）口径数字。

方法：gaiban B1 `b1_allreduce.py` 加 `B1_DIM` 参数化（7168 = Pro 锚点对齐口径，4096 = Flash hidden），batch 扫到 512；摆位 TP4 socket0 / TP4 socket1 / TP4 cross-socket / TP8，外加 P2P-disabled 基线；两台各跑一遍（`run_b1_titan.sh`，venv `~/Workspace/venvs/sglang`）。

结论：
1. **Pro 锚点复现**：DIM=7168 bf16 TP4-socket0：B128 = 159.4 / 159.6 µs（064/065，锚点 161 µs），B256 = 265.8 / 265.7 µs（锚点 266 µs）。两台与 dsv4exp 行为一致，P2P 补丁生效。
2. **Flash 口径（DIM=4096 bf16，titan064）**：TP4-socket0 B128 112.1 µs / B256 176.8 µs / B512 299.5 µs；TP4-socket1 B128 105.9 / B256 170.4 / B512 342.0 µs；TP4-cross B512 332.4 µs；TP8 B512 444.1 µs。可行性文档 §5.2 的 DP 通信估计（11 层 × [512,4096] ≈ 4.4 ms，即 ~400 µs/层）与实测相比偏保守，实测更好（~300–342 µs/次）。
3. **P2P-disabled 基线**：DIM=4096 B512 1486 µs vs 300 µs，慢约 5×，确认 P2P 补丁收益。
4. 两台机器数字一致（差异 <5%）。

Artifacts：`results/titan064.log`、`results/titan065.log`（完整 9 段 CONFIG 原始输出）。
