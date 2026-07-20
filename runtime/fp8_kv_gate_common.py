"""Shared helpers for the FP8 KV quality gates (fifteenth vertical).

Two measurement utilities used by the modified E0wf/E0ef oracle gates and the
new E0kf ratio-4 paired gate:

- ``latent_amplitude_stats``: real-weight latent row amplitude distribution,
  split nope/rope, against the e4m3 dynamic range (max 448, min normal 2^-6,
  min subnormal 2^-9).  This decides the A6F open point "constant-scale direct
  cast vs written-side scale": if ``frac_above_448`` is zero and the mass
  below the subnormal floor is negligible, scale 1.0 direct cast is safe.
- ``fp8_qdq_error_stats``: rms relative error of one clamp+e4m3 round trip,
  split nope/rope, quantifying what FP8 storage does to these exact rows.

FP8 gate limits: the frozen E0wf/E0ef limits compare a BF16 control against
an FP32 oracle; FP8 KV is a *semantic* change (quantized cache reads), so the
task methodology (E0hf form) keeps every non-cache stage at its frozen limit
and relaxes only cache-derived stages, recording observed magnitudes.  The
final accept/reject remains the E2E golden gate (D0 mismatch-rate vs the
468/482 eager-bf16 baseline).
"""

from __future__ import annotations

import torch


E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0**-6
E4M3_MIN_SUBNORMAL = 2.0**-9


def _segment_stats(magnitude: torch.Tensor) -> dict:
    flat = magnitude.float().flatten()
    nonzero = flat[flat > 0]
    quantiles = torch.quantile(
        flat, torch.tensor([0.5, 0.99, 0.999], device=flat.device)
    )
    return {
        "amax": float(flat.max().item()) if flat.numel() else 0.0,
        "amin_nonzero": float(nonzero.min().item()) if nonzero.numel() else 0.0,
        "p50": float(quantiles[0].item()),
        "p99": float(quantiles[1].item()),
        "p999": float(quantiles[2].item()),
        "frac_zero": float((flat == 0).float().mean().item()),
        "frac_below_e4m3_min_normal": float(
            ((flat > 0) & (flat < E4M3_MIN_NORMAL)).float().mean().item()
        ),
        "frac_below_e4m3_min_subnormal": float(
            ((flat > 0) & (flat < E4M3_MIN_SUBNORMAL / 2)).float().mean().item()
        ),
        "frac_above_e4m3_max": float((flat > E4M3_MAX).float().mean().item()),
    }


def latent_amplitude_stats(value: torch.Tensor, *, rope_dim: int = 64) -> dict:
    """Amplitude distribution of latent rows, split nope/rope vs e4m3 range."""

    v = value.detach().float()
    return {
        "shape": list(value.shape),
        "nope": _segment_stats(v[..., :-rope_dim].abs()),
        "rope": _segment_stats(v[..., -rope_dim:].abs()),
    }


def _rms_rel(observed: torch.Tensor, expected: torch.Tensor) -> float:
    difference = observed - expected
    rms_abs = float(torch.sqrt(torch.mean(difference.square())).item())
    reference = float(torch.sqrt(torch.mean(expected.square())).item())
    return rms_abs / max(reference, 1e-12)


def fp8_qdq_error_stats(value: torch.Tensor, *, rope_dim: int = 64) -> dict:
    """RMS relative error of one clamp+e4m3 write/read round trip."""

    v = value.detach().to(torch.bfloat16).float()
    q = (
        value.detach()
        .to(torch.bfloat16)
        .clamp(-E4M3_MAX, E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    return {
        "rms_rel_full": _rms_rel(q, v),
        "rms_rel_nope": _rms_rel(q[..., :-rope_dim], v[..., :-rope_dim]),
        "rms_rel_rope": _rms_rel(q[..., -rope_dim:], v[..., -rope_dim:]),
        "max_abs_err": float((q - v).abs().max().item()),
    }


# Cache-derived stages whose limits relax under FP8 KV (magnitude-recording
# ceilings, ~2.5x the expected e4m3 quantization scale; non-cache stages keep
# their frozen limits).  Applies to both e0wf and e0ef limit tables; keys not
# present in a gate's table are ignored.
FP8_STAGE_RMS_REL_OVERRIDES = {
    "attention_kv": 0.08,
    "compression_finalized": 0.08,
    "sparse_output": 0.08,
    "inverse_rotated": 0.08,
    "output_lora": 0.09,
    "branch": 0.10,
    "state.raw": 0.08,
    "state.compressed": 0.08,
}


__all__ = [
    "E4M3_MAX",
    "E4M3_MIN_NORMAL",
    "E4M3_MIN_SUBNORMAL",
    "FP8_STAGE_RMS_REL_OVERRIDES",
    "fp8_qdq_error_stats",
    "latent_amplitude_stats",
]
