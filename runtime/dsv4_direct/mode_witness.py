"""Treatment witness: what the built layers actually resolved to (E5F).

An env-gated variant that fails to reach the process is **indistinguishable
from a variant that does nothing**.  This repo has now been bitten by that
three times in three disguises:

- 22nd vertical: the fast/slow MoE runs were indistinguishable in the recorded
  metadata, which is why ``c2f_prefill_stage_bench.py`` began recording the
  allocator configuration;
- E4F: ``${E1F_EXTRA_ENV:-:}`` inside a single-quoted ``ENV_BASE`` reached the
  remote shell literally and expanded to a no-op there, so a measured -3.82%
  fusion read as 0%;
- E5F: the witness added in E4F listed ``indexer_qat_mode`` but not
  ``kv_qat_mode``, so the next vertical's treatment was invisible in its own
  artifact.

The first two were fixed case by case.  The third is what makes the case for a
convention instead: **a hand-maintained key list is itself a thing that can go
stale**, and it goes stale exactly when someone adds the switch they are about
to measure.

So this module discovers the modes rather than enumerating them.  Any attribute
whose name ends in ``_mode`` or ``_backend`` is recorded, which means a new
switch is witnessed the moment it exists, without anyone remembering to come
back here.  The cost of over-collecting is a few extra strings in a JSON; the
cost of under-collecting is a false negative that looks like a real result.
"""

from __future__ import annotations

from typing import Any, Iterable


_SUFFIXES = ("_mode", "_backend")


def describe_modes(obj: Any) -> dict[str, Any]:
    """Every ``*_mode`` / ``*_backend`` attribute of one object, as strings."""

    if obj is None:
        return {}
    found: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_") or not name.endswith(_SUFFIXES):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001 - a property that raises is not a mode
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            found[name] = value
        else:
            # backends are objects; their identity is the useful part
            found[name] = type(value).__name__
    return found


def collect_attention_modes(
    layers: Iterable[tuple[Any, Any]] | None = None,
    *,
    layer_ids: Iterable[Any] | None = None,
    attentions: Iterable[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-layer witness, from either ``(material, attention)`` pairs or two
    parallel iterables of ids and attention objects."""

    if layers is not None:
        pairs = [
            (getattr(material, "layer_id", index), attention)
            for index, (material, attention) in enumerate(layers)
        ]
    elif layer_ids is not None and attentions is not None:
        pairs = list(zip(layer_ids, attentions, strict=True))
    else:
        raise ValueError("pass either layers=, or both layer_ids= and attentions=")
    return {str(layer_id): describe_modes(attention) for layer_id, attention in pairs}


__all__ = ["collect_attention_modes", "describe_modes"]
