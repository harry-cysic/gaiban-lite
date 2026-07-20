# E0ef V4-Flash TP4 ratio-128 attention oracle

Experiment: `E0ef-ratio128-attention-oracle`

Status: **PASS**

This is a real-checkpoint semantic correctness gate, not a performance run.
It compares the direct BF16 control against an independent raw-checkpoint
FP32 projection, compressor, RoPE, QDQ, sparse-softmax, and output oracle.

NoPE semantics are fixed to intended E4M3/UE8M0 QAT. Exact checks cover
top-k indices, next position, compression presence, and compressed-row count.

Checkpoint: `e33d9526298d9d5c5bc5ffa563fd0b6b84f724c79fa29877b103541750ae95b4`
Implementation: `eabf29b6ef5fb43a61f1974d4fd4d01430be8f58cf260575992dc84512f2a3f1`

- `prefill128_decode128`: PASS; worst rms_rel `prefill.output_lora=0.0162472` (limit `0.09`)
- `prefill127_decode127_boundary`: PASS; worst rms_rel `prefill.output_lora=0.0162079` (limit `0.09`)
