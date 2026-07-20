# E0ff V4-Flash TP4 layer-2 ratio-4 semantic gate

Status: **PASS**

This is a ratio-4 transition-component semantic diagnostic for the
DeepSeek-V4-Flash geometry (hidden 4096, 64 heads, index_topk 512),
not a latency run.
Attention uses B=1 per rank over positions 8192..8195 from a nonzero
independent QAT-valid state. Hash routing separately uses B=60 per rank
and a selected-six-only FP64 oracle over 240 deterministically scanned
checkpoint token IDs (route_scale 1.5, 256 experts).
Each attention phase is teacher-forced from the independently prepared
BF16-operand control state; that control alone supplies the acceptance gate.
The control has independent state/ratio-4 math but shares the torch CUDA
BF16 GEMM backend with the candidate; it is not an independent GEMM kernel.
A raw-FP32 dequantized-checkpoint lane starts from the same state and is
retained as non-gating attribution, including complete score/route witness.
Raw-FP32 prior limits all passed: `True`; top-k classifications: `{'exact': 16, 'ordering_only': 0, 'set_change': 0}`; minimum overlap: `512/512`.
The nonzero QAT-valid seed is a random algebraic state, not proof that it
is reachable from an actual 8192-token prompt history.

No autonomous rollout, prompt, full-sequence, full-layer, pipeline, cluster
end-to-end, performance, or checkpoint-native FP8 GEMM claim is made.

Checkpoint: `b3d44d9f14e59b1be0f108fd91e4a9a171650795e03a48dc8c47957cf40d22d9`
Implementation: `8245546f095fdaf9307eadcac92c24bf7e0309268e7a664760a82225579a04a9`
