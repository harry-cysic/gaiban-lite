# B2-ib-recal：titan064↔065 双机 IB NCCL 标定复跑

日期 2026-07-20。Phase 0 前置项。动机：固件统一（16.28.1002）后确认双机 IB 现状，复核可行性文档 §5.1 机间锚点，并给出 Flash PP payload 关键点的实测。

方法：复用 gaiban `rdma-nccl-qualification` 的 `nccl_pair_bench.cu`（2-rank allreduce 1MiB–256MiB）与 `nccl_pp_sendrecv_bench.cu`（定向 send/recv，sizes 448KiB–64MiB），`nvcc -arch=sm_89 -lnccl` 现场编译；no-GDR 与 GDR（`LD_LIBRARY_PATH=/home/cysic/libcuda-onebyte-patch` + `NCCL_NET_GDR_LEVEL=SYS`）各一轮；驱动脚本 `run_b2.sh`（从工作站经 ssh 编排，GPU0↔GPU0）。

结论：
1. **no-GDR 与锚点一致**：pair allreduce busbw 4.047 GB/s@256MiB（历史 4.192）；PP send/recv 单向 4.202 GB/s@16.8MB、4.219 GB/s@64MiB（锚点 ~4.0–4.2）。
2. **GDR 生效且与锚点一致**：transport 确认 `GDRDMA` / `GDR 1`；pair allreduce busbw 11.1 GB/s@≥8MiB；PP send/recv 9.234 GB/s@64MiB（fleet 升级后历史 9.236）。libcuda-onebyte-patch 的 LD_LIBRARY_PATH opt-in 在两台仍有效。
3. **Flash PP payload 关键点（B=512 跳 payload 16.8MB）**：GDR 1.837 ms/跳（9.135 GB/s）；no-GDR 3.993 ms/跳（4.202 GB/s）。可行性模型 §5.2 的 PP handoff 摊销 ~2 ms 与 GDR 路径吻合；no-GDR 下不成立。
4. 小包（448KiB）GDR 4.136 GB/s、512KiB 6.437 GB/s——小 payload 端仍受每包开销限制，与 gaiban 历史形态一致。

Artifacts：`results/driver.log`、`results/20260720-142648/{pair,pp}_{gdr,no-gdr}_rank{0,1}.log`。
