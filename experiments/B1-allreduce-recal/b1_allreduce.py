"""B1-recal: NCCL all_reduce calibration on titan064/065 (Flash geometry).

Derived from gaiban B1 b1_allreduce.py; changes:
  - DIM overridable via B1_DIM (7168 = Pro parity anchor, 4096 = Flash hidden)
  - batch sweep extended to 512 (Flash B_micro=512 global)

Measures all_reduce of [B, DIM] across a TP group, swept over decode batch
sizes and dtype. Run via torchrun; CUDA_VISIBLE_DEVICES selects physical GPUs
(0-3 socket0, 4-7 socket1, cross => xGMI).

busbw = algbw * 2(n-1)/n is the standard ring-allreduce bus bandwidth.
"""
import os, torch, torch.distributed as dist

DIM = int(os.environ.get("B1_DIM", "7168"))
BS = [1, 8, 16, 32, 64, 128, 256, 512]


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    tag = os.environ.get("B1_TAG", "")

    if rank == 0:
        print(f"\n##### {tag} | world={world} | all_reduce [B,{DIM}] #####")
        print(f"{'B':>5} {'dtype':>7} {'MB':>7} {'us':>9} {'algbw':>9} {'busbw':>9}")
    for dtype in (torch.bfloat16, torch.float32):
        for B in BS:
            x = torch.randn(B, DIM, device=dev, dtype=dtype)
            nbytes = x.numel() * x.element_size()
            for _ in range(15):
                dist.all_reduce(x)
            torch.cuda.synchronize(); dist.barrier()
            iters = 50
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(iters):
                dist.all_reduce(x)
            e1.record(); torch.cuda.synchronize()
            us = e0.elapsed_time(e1) / iters * 1e3
            algbw = nbytes / (us * 1e-6) / 1e9
            busbw = algbw * 2 * (world - 1) / world
            if rank == 0:
                print(f"{B:>5} {str(dtype).replace('torch.',''):>7} {nbytes/2**20:>7.2f}"
                      f" {us:>9.1f} {algbw:>9.1f} {busbw:>9.1f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
