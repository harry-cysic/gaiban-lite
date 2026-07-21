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
Implementation: `c756da8ff94c69264585588bad0cffdea463f7c407722b204797c1c37d4ea578`

- `prefill96_decode34_ring_cross`: PASS; worst rms_rel `prefill.output_lora=0.0126786` (limit `0.035`)
- `prefill128_decode4`: PASS; worst rms_rel `prefill.output_lora=0.0125588` (limit `0.035`)
- `prefill200_decode4`: PASS; worst rms_rel `prefill.output_lora=0.0132773` (limit `0.035`)
