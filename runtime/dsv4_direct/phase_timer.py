"""Opt-in CUDA-event phase timing (C4F, 27th vertical).

Diagnostic only.  ``PhaseRecorder.mark`` issues one ``cudaEventRecord`` on the
current stream and never synchronizes, so the instrumented code keeps its own
kernel order and no device barrier is inserted mid-pass -- unlike the C2F
component walls, whose per-component ``synchronize`` costs up to +14.7% on
short forwards.  The single synchronize happens in ``end()``, after the pass.

With no recorder attached (the default everywhere) the cost of a mark site is
one attribute load plus a branch.

Span naming: a mark labels the span that *ended* at it, so a pass starts with
``mark("enter")`` and every later mark carries the name of the phase it just
closed.
"""

from __future__ import annotations

import statistics

import torch


class PhaseRecorder:
    """Reusable pool of timing events for repeated instrumented passes."""

    def __init__(self, device: torch.device | str, *, capacity: int = 64) -> None:
        self.device = torch.device(device)
        self._pool: list[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(capacity)
        ]
        self._names: list[str] = []
        self._used = 0
        self.passes: list[list[tuple[str, float]]] = []

    def begin(self) -> None:
        self._names = []
        self._used = 0

    def mark(self, name: str) -> None:
        if self._used >= len(self._pool):
            self._pool.append(torch.cuda.Event(enable_timing=True))
        event = self._pool[self._used]
        event.record()
        self._names.append(name)
        self._used += 1

    def end(self) -> list[tuple[str, float]]:
        torch.cuda.synchronize(self.device)
        spans = [
            (
                self._names[index],
                float(self._pool[index - 1].elapsed_time(self._pool[index])),
            )
            for index in range(1, self._used)
        ]
        self.passes.append(spans)
        return spans

    # ------------------------------------------------------------------

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-phase p50/mean/total over the recorded passes (ms)."""

        collected: dict[str, list[float]] = {}
        order: list[str] = []
        for spans in self.passes:
            per_pass: dict[str, float] = {}
            for name, value in spans:
                if name not in per_pass:
                    per_pass[name] = 0.0
                    if name not in collected:
                        collected[name] = []
                        order.append(name)
                per_pass[name] += value
            for name, value in per_pass.items():
                collected[name].append(value)
        result: dict[str, dict[str, float]] = {}
        for name in order:
            samples = collected[name]
            result[name] = {
                "p50_ms": float(statistics.median(samples)),
                "mean_ms": float(statistics.fmean(samples)),
                "min_ms": float(min(samples)),
                "max_ms": float(max(samples)),
                "calls_per_pass": len(
                    [1 for spans in self.passes[:1] for n, _ in spans if n == name]
                ),
            }
        return result

    def pass_totals_ms(self) -> list[float]:
        return [float(sum(value for _, value in spans)) for spans in self.passes]


__all__ = ["PhaseRecorder"]
