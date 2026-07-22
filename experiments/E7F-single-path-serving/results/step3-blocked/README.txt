E7F step 3 — blocked-run evidence (2026-07-22)
================================================

node0-error-excerpt.txt holds the two things the smoke run established before
the stateful arm hit the architectural block:

1. The eager (non-stateful) arm reproduced the frozen v2 baseline EXACTLY on
   the three 2048-token prompts (the smoke subset, --prompt-max-tokens 2048):
     prompt 0 (orig p3, len 2048): 63/64   -- frozen p3 = 63  ✓
     prompt 1 (orig p4, len 2048): 60/64   -- frozen p4 = 60  ✓
     prompt 2 (orig p5, len 2048): 60/64   -- frozen p5 = 60  ✓
   183/192, byte-for-byte the frozen non-stateful numbers.  This proves the
   --with-stateful scaffolding did NOT perturb the frozen path, and that the
   config (sharded default, chunk 4096, share-moe-buffers, tilelang prefill)
   is the right one.

2. The stateful arm then failed at TP4DecodeStage construction:
     ValueError: super-stage MoE slot buffers must not alias across layers
   because --share-moe-buffers (needed for the 8192 prefill) aliases the MoE
   slot buffers across layers, which the decode superstage forbids.  See the
   E7F README section "步 3 尝试" and TARGET 7.10.

A guard now makes --with-stateful + --share-moe-buffers a clean SystemExit
before any collective, so this can never deadlock again.
