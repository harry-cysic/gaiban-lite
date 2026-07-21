"""Reference tilelang ``sparse_attn`` as an optional prefill sparse core.

Twenty-first vertical.  The C2F re-attribution
(``experiments/C2F-prefill/results/reattribution/``) showed prefill is 58%
attention (ratio-4 41% + ratio-128 17%) while the runtime still runs the
torch masked-einsum correctness core (``attention.torch_sparse_attention``);
the reference tilelang kernel was never wired in.  The micro-probe
(``runtime/c2f_attention_kernel_probe.py``) measured 6.49x at prefill shape
(106.07 -> 16.36 ms, -55% peak memory) with bf16-level agreement.

This module supplies a drop-in replacement with the **same signature** as
``torch_sparse_attention`` so the three prefill call sites can switch
backends without touching any surrounding operator.

sm89 head loop
--------------
``sparse_attn_kernel`` keeps ``q_shared[h, d]``, ``kv_shared[block, d]``,
``o_shared[h, d]`` and ``acc_s_cast[h, block]`` in shared memory.  At
``h = 64, d = 512`` that is 141312 B against the 101376 B sm89 limit (A4F),
so the wrapper loops the head axis in chunks of ``head_chunk`` (16 by
default: 16*512*2*2 + 64*512*2 + 16*64*2 = 98304 B).  Heads are fully
independent in this kernel (per-head softmax, per-head sink), so chunking is
an exact decomposition -- not an approximation.

``-1`` padding semantics (aligned here; see the report in
``experiments/C2F-prefill/results/tilelang-attn/``)
--------------------------------------------------
``torch_sparse_attention`` treats **any negative** index as padding
(``valid = topk_indices >= 0``) and validates that valid indices are inside
the KV capacity.  The tilelang kernel tests **exactly ``!= -1``**
(kernel.py:325-327), so a ``-2`` would be gathered as ``kv[b, -2]`` (a
wrap-around read) instead of being masked.  Every runtime producer emits
exactly ``-1`` (``window_topk_indices``, ``compressed_topk_indices``, and
ratio-4's ``-1 - offset`` then ``+ offset``), so the two agree on the shipped
paths, but the wrapper normalises negatives to ``-1`` when it sees any other
negative value rather than relying on that invariant.

Second difference: an **all-padding row**.  torch yields zeros (row max
falls back to the sink, numerator 0, denominator 1); the kernel's running
max stays ``-inf``, so ``exp(-inf - (-inf))`` is NaN and the row is poisoned.
No prefill call site can produce such a row -- all three concatenate the
causal window part, whose diagonal entry is always valid -- but the wrapper
detects the case in its single validation reduction and zero-fills those
rows so the two backends agree unconditionally.

Third difference (numerics only, not aligned because it cannot be):
torch stabilises with ``M = max(row_max, sink)``; the kernel stabilises with
``m = row_max`` and folds the sink in afterwards as ``exp(sink - m)``.  The
two are algebraically identical; they differ only in rounding, and the
kernel's form would overflow if ``sink - row_max`` exceeded ~88.  The gate
records the observed ``sink - row_max`` margin.

Configuration (same style as ``--index-score-mode`` / ``--kv-dtype``):

- ``DSV4_PREFILL_SPARSE_BACKEND``: ``torch`` (default) | ``tilelang``.
- ``DSV4_PREFILL_SPARSE_HEAD_CHUNK``: head-loop width, default 16.
- ``DSV4_TILELANG_KERNEL`` / ``DSV4_TILELANG_REFERENCE_DIR``: explicit
  reference ``kernel.py`` (file / directory).  Otherwise auto-detected.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch


class TilelangSparseAttentionError(RuntimeError):
    """Raised when the tilelang sparse core cannot be located or applied."""


DEFAULT_HEAD_CHUNK = 16
_BACKENDS = ("torch", "tilelang")

# Directories that have held the reference tree across the machines this
# runtime is exercised on.  ``kernel.py`` is byte-identical in all of them
# (md5 e4d8e272f13515b899ef8b145b736001, 2026-07-21).
_HOME_CANDIDATES = (
    "a5f/kernel.py",
    "flash-oracle/reference/inference/kernel.py",
    "reference/inference/kernel.py",
    "e0f-runtime/reference/inference/kernel.py",
)

_kernel_module = None
_kernel_path: str | None = None


def resolve_prefill_sparse_backend(explicit: str | None = None) -> str:
    """``explicit`` if given, else ``DSV4_PREFILL_SPARSE_BACKEND``, else torch."""

    if explicit is not None:
        value = explicit
    else:
        value = os.environ.get("DSV4_PREFILL_SPARSE_BACKEND", "").strip() or "torch"
    if value not in _BACKENDS:
        raise ValueError(
            f"prefill sparse backend must be one of {_BACKENDS}, got {value!r}"
        )
    return value


def resolve_head_chunk(explicit: int | None = None) -> int:
    """``explicit`` if given, else ``DSV4_PREFILL_SPARSE_HEAD_CHUNK``, else 16."""

    if explicit is not None:
        value = explicit
    else:
        raw = os.environ.get("DSV4_PREFILL_SPARSE_HEAD_CHUNK", "").strip()
        if not raw:
            return DEFAULT_HEAD_CHUNK
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(
                f"DSV4_PREFILL_SPARSE_HEAD_CHUNK must be an integer, got {raw!r}"
            ) from error
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("head chunk must be a positive integer")
    return value


def find_reference_kernel() -> Path:
    """Locate the reference ``kernel.py`` (explicit env, repo tree, then home)."""

    explicit_file = os.environ.get("DSV4_TILELANG_KERNEL", "").strip()
    if explicit_file:
        path = Path(explicit_file).expanduser()
        if not path.is_file():
            raise TilelangSparseAttentionError(
                f"DSV4_TILELANG_KERNEL={explicit_file!r} is not a file"
            )
        return path.resolve()

    explicit_dir = os.environ.get("DSV4_TILELANG_REFERENCE_DIR", "").strip()
    if explicit_dir:
        path = Path(explicit_dir).expanduser() / "kernel.py"
        if not path.is_file():
            raise TilelangSparseAttentionError(
                f"DSV4_TILELANG_REFERENCE_DIR={explicit_dir!r} holds no kernel.py"
            )
        return path.resolve()

    # Repo / deployment tree: walk up from dsv4_direct/ops/ looking for the
    # sibling reference checkout (works both in-repo and under ~/e0f-runtime
    # when the launcher rsyncs reference/inference/kernel.py alongside).
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "reference" / "inference" / "kernel.py"
        if candidate.is_file():
            return candidate

    home = Path(os.path.expanduser("~"))
    for relative in _HOME_CANDIDATES:
        candidate = home / relative
        if candidate.is_file():
            return candidate

    raise TilelangSparseAttentionError(
        "reference kernel.py not found; set DSV4_TILELANG_KERNEL or "
        "DSV4_TILELANG_REFERENCE_DIR (looked in the repo tree and "
        + ", ".join(f"~/{relative}" for relative in _HOME_CANDIDATES)
    )


def load_reference_kernel_module():
    """Import the reference ``kernel.py`` under a private module name.

    ``spec_from_file_location`` is used instead of ``sys.path`` insertion so a
    generically named ``kernel`` module elsewhere on the path cannot shadow (or
    be shadowed by) the reference one.
    """

    global _kernel_module, _kernel_path
    if _kernel_module is not None:
        return _kernel_module
    path = find_reference_kernel()
    name = "dsv4_direct_reference_kernel"
    existing = sys.modules.get(name)
    if existing is not None:
        _kernel_module, _kernel_path = existing, str(path)
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TilelangSparseAttentionError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - environment dependent
        sys.modules.pop(name, None)
        raise TilelangSparseAttentionError(
            f"importing the reference kernel at {path} failed: {error!r}"
        ) from error
    if not callable(getattr(module, "sparse_attn", None)):
        sys.modules.pop(name, None)
        raise TilelangSparseAttentionError(
            f"{path} does not expose a callable sparse_attn"
        )
    _kernel_module, _kernel_path = module, str(path)
    return module


def reference_kernel_path() -> str | None:
    """Path of the loaded reference kernel (``None`` before the first load)."""

    return _kernel_path


def tilelang_sparse_attention(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    head_chunk: int | None = None,
) -> torch.Tensor:
    """Reference tilelang sparse MLA, signature-compatible with the torch core.

    Same contract as ``attention.torch_sparse_attention``: ``query``
    ``[b, s, h, d]``, ``latent_kv`` ``[b, n, d]``, ``attn_sink`` ``[h]``,
    ``topk_indices`` ``[b, s, k]`` with ``-1`` padding, returning
    ``[b, s, h, d]`` in ``query.dtype``.
    """

    # --- shape/device validation, mirroring the torch core exactly ---------
    if query.ndim != 4 or latent_kv.ndim != 3 or topk_indices.ndim != 3:
        raise ValueError("query/latent_kv/topk_indices must have ranks 4/3/3")
    batch, seqlen, heads, head_dim = query.shape
    if latent_kv.shape[0] != batch or latent_kv.shape[2] != head_dim:
        raise ValueError("query and latent KV shapes are incompatible")
    if tuple(topk_indices.shape[:2]) != (batch, seqlen):
        raise ValueError("top-k batch/sequence shape mismatch")
    if tuple(attn_sink.shape) != (heads,):
        raise ValueError(f"attn_sink shape {tuple(attn_sink.shape)} != ({heads},)")
    if query.device != latent_kv.device or query.device != topk_indices.device:
        raise ValueError("attention tensors must share one device")

    # --- dtype contract of the kernel -------------------------------------
    if query.dtype != torch.bfloat16 or latent_kv.dtype != torch.bfloat16:
        raise TilelangSparseAttentionError(
            "the tilelang sparse core requires BF16 query/latent_kv, got "
            f"{query.dtype}/{latent_kv.dtype}"
        )
    chunk = resolve_head_chunk(head_chunk)
    sparse_attn = load_reference_kernel_module().sparse_attn

    rows = latent_kv.shape[1]
    indices = topk_indices
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    # --- one fused validation reduction (parity with the torch core's own
    # capacity check, plus the two kernel-specific padding hazards) --------
    negative = indices < 0
    stats = torch.stack(
        (
            # valid index outside the KV capacity -> torch raises
            (indices.ge(rows)).sum(),
            # padding that is negative but not exactly -1 -> the kernel would
            # gather kv[b, idx] instead of masking
            (negative & indices.ne(-1)).sum(),
            # rows with no valid candidate -> the kernel's running max stays
            # -inf and the row becomes NaN; torch returns zeros
            (~negative).sum(dim=-1).eq(0).sum(),
        )
    ).tolist()
    over_capacity, stray_negative, empty_rows = (int(value) for value in stats)
    if over_capacity:
        raise ValueError("top-k index exceeds latent KV capacity")
    if stray_negative:
        # Align with the torch core, which masks *any* negative.
        indices = torch.where(negative, indices.new_full((), -1), indices)

    if not indices.is_contiguous():
        indices = indices.contiguous()
    if attn_sink.dtype != torch.float32:
        attn_sink = attn_sink.to(torch.float32)
    latent = latent_kv if latent_kv.is_contiguous() else latent_kv.contiguous()

    # --- head loop (exact decomposition; heads are independent) -----------
    output = torch.empty_like(query)
    for start in range(0, heads, chunk):
        stop = min(start + chunk, heads)
        piece = sparse_attn(
            query[:, :, start:stop].contiguous(),
            latent,
            attn_sink[start:stop].contiguous(),
            indices,
            float(softmax_scale),
        )
        output[:, :, start:stop].copy_(piece)
        del piece

    if empty_rows:
        # torch returns zeros for an all-padding row; the kernel returns NaN.
        empty = (~negative).sum(dim=-1).eq(0)
        output[empty] = 0
    return output.to(query.dtype)


def prefill_sparse_core(backend: str):
    """Return the sparse core callable for ``backend`` (torch | tilelang)."""

    if backend == "torch":
        from ..attention import torch_sparse_attention

        return torch_sparse_attention
    if backend == "tilelang":
        return tilelang_sparse_attention
    raise ValueError(f"unknown prefill sparse backend {backend!r}")


__all__ = [
    "DEFAULT_HEAD_CHUNK",
    "TilelangSparseAttentionError",
    "find_reference_kernel",
    "load_reference_kernel_module",
    "prefill_sparse_core",
    "reference_kernel_path",
    "resolve_head_chunk",
    "resolve_prefill_sparse_backend",
    "tilelang_sparse_attention",
]
