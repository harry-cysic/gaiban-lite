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
Implementation: `ec749f40b1304855510df099fbb02767bf5055c087c693863bfe465053243c85`

- `prefill96_decode34_ring_cross`: PASS; worst rms_rel `prefill.output_lora=0.0126104` (limit `0.035`)
- `prefill128_decode4`: PASS; worst rms_rel `prefill.output_lora=0.0125028` (limit `0.035`)
- `prefill200_decode4`: PASS; worst rms_rel `prefill.output_lora=0.0132205` (limit `0.035`)
