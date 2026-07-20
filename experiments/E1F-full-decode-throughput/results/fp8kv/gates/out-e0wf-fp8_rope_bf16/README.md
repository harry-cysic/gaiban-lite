# E0wf V4-Flash TP4 pure sliding-window attention oracle

Experiment: `E0wf-window-attention-oracle`

Status: **PASS**

This is a real-checkpoint semantic correctness gate, not a performance run.
It compares the direct BF16 window-attention control (layer 0, compress
ratio 0) against an independent raw-checkpoint FP32 projection, RoPE
(no-YaRN, base theta 10000), QDQ, sparse-softmax, and output oracle.

Exact checks cover window top-k indices and next position for every
prefill/decode phase, including ring-boundary crossing and wrap.

Checkpoint: `1ae890a2f545e4be6f8f2d52fe34169848a2fc0f9f2c03dca7c2218b005f025a`
Implementation: `81ccd827d1c0fc9076a2945c0f121e7e644c0237dfc3671d0b6808c4a6352e11`

- `prefill96_decode34_ring_cross`: PASS; worst rms_rel `prefill.output_lora=0.0126113` (limit `0.09`)
- `prefill128_decode4`: PASS; worst rms_rel `prefill.output_lora=0.012502` (limit `0.09`)
- `prefill200_decode4`: PASS; worst rms_rel `prefill.output_lora=0.0132201` (limit `0.09`)
