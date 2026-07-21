#!/usr/bin/env python3
"""D0L: deterministic construction of the long-prompt set for the chunk-regime
golden-token oracle.

Motivation (C2F 23rd vertical, prefill-overlap/README.md section 1.6): the D0
oracle's eight prompts are 10-22 tokens, so the E2E golden gate's prefill never
leaves the "short" regime.  Every prefill-only code path that changes behaviour
at scale -- the fused HC boundary's ``MAX_ROWS = 896`` row split (and the vLLM
``num_tokens >= 1024`` wrong-kernel branch it guards), the fused indexer's
``fuse_min_seqlen``, the row-blocked MoE collective overlap -- is invisible to
that gate.  This script builds prompts whose *prompt token count* lands exactly
on 1024 / 2048 / 4096 / 8192 so a single ``start_pos = 0`` prefill forward puts
the runtime squarely in the C2F "chunk" regime (C2F defines chunk = the seqlen
of one whole-sequence prefill at start_pos 0).

Construction (fully reproducible):

1. A corpus is formed by concatenating a fixed, ordered list of real
   documents from this repository (English model card + Chinese engineering
   docs) with a fixed separator.  Real natural text is used deliberately --
   random vocabulary sampling would produce token statistics the model never
   sees, and routing/top-k behaviour under MoE would not be representative.
2. Each prompt takes a disjoint character window of that corpus as its body,
   wrapped in a header + trailing instruction so the prompt is a well-posed
   task ("read this excerpt, then answer").  The body is truncated
   mid-sentence; the trailing instruction keeps the prompt well formed.
3. The body's character length is binary-searched so that the *templated*
   prompt (``encode_messages(..., thinking_mode="chat")`` then
   ``tokenizer.encode``) has exactly the target token count.

Outputs ``long_prompts.json``: the frozen prompt texts plus provenance (source
file md5s, window offsets, achieved token counts).  The JSON is the artifact of
record -- the golden oracle and the E2E gate read it, so the prompt set stays
fixed even if the source documents are later edited.

Run (needs the checkpoint tokenizer + the reference encoding module):
  python3 build_long_prompts.py \
      --repo-root /path/to/gaiban-lite \
      --tokenizer /path/to/DeepSeek-V4-Flash-mp8 \
      --encoding-dir /path/to/reference/encoding \
      --out long_prompts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Ordered corpus.  Chosen for (a) being real prose this repository already
# contains, (b) a deliberate English/Chinese mix, (c) no DeepSeek special-token
# characters -- reference/encoding/README.md is excluded precisely because it
# quotes "<|begin_of_sentence|>"-style markers, which would re-tokenize as
# control tokens inside user content.
SOURCES: tuple[str, ...] = (
    "reference/README.md",
    "docs/feasibility-v4-flash-2x8x4090.md",
    "README.md",
    "runtime/PORT-PLAN.md",
    "CLAUDE_GOAL.md",
    "experiments/C2F-prefill/README.md",
    "experiments/C2F-prefill/results/prefill-overlap/README.md",
    "experiments/C2F-prefill/results/tilelang-attn/README.md",
    "experiments/C2F-prefill/results/moe-alloc/README.md",
    "experiments/E1F-full-decode-throughput/README.md",
    "experiments/E1F-full-decode-throughput/results/mtp/README.md",
    "experiments/E1F-full-decode-throughput/results/mtp-largeb/README.md",
    "experiments/A6F-fp8-kv-speed/README.md",
    "experiments/C1F-integrated-block/README.md",
    "experiments/A3F-marlin-moe-flash/README.md",
    "experiments/A0-flash-checkpoint-contract/README.md",
    "experiments/A4F-attention-flash/README.md",
    "experiments/D0-reference-oracle/README.md",
    "experiments/A5F-hc-boundary-fusion/README.md",
    "experiments/B2-ib-recal/README.md",
    "experiments/B1-allreduce-recal/README.md",
    "experiments/C2F-prefill/results/reattribution/README.md",
    "experiments/C2F-prefill/results/dense/README.md",
)

CORPUS_SEPARATOR = "\n\n=====\n\n"

# Half-width of the cut-point scan used to land exactly on a target length.
WINDOW = 96

# Filler units tried, in order, when the body cut lands short of the target.
# Each is expected to cost exactly one token per repetition; several are listed
# because that depends on the surrounding BPE context.
FILLERS: tuple[str, ...] = (" .", "\n", " x", " 、", " ,")

# (target prompt tokens, header, trailing instruction).  Ten prompts covering
# the four chunk sizes the C2F prefill work benchmarks (1024/2048/4096/8192),
# with the cheap/safe lengths sampled more often than the 8192 tail.
# Instructions alternate Chinese/English so the decode side is not a single
# repeated task.
SPECS: tuple[tuple[int, str, str], ...] = (
    (
        1024,
        "以下是一份工程文档的节选(可能在中途截断)。\n\n---\n\n",
        "\n\n---\n\n请用三句话概括上文的主要内容。",
    ),
    (
        1024,
        "Below is an excerpt from a technical document. It may be truncated "
        "mid-sentence.\n\n---\n\n",
        "\n\n---\n\nList the three most important facts stated above.",
    ),
    (
        1024,
        "阅读下面的材料,然后回答问题。\n\n---\n\n",
        "\n\n---\n\n上文讨论的核心技术问题是什么?请简要说明。",
    ),
    (
        2048,
        "以下是一份工程实验报告的节选(可能在中途截断)。\n\n---\n\n",
        "\n\n---\n\n请概括上文中给出的主要实验结论。",
    ),
    (
        2048,
        "Read the following excerpt from an engineering report.\n\n---\n\n",
        "\n\n---\n\nSummarize the main measurement result reported above.",
    ),
    (
        2048,
        "下面是一段技术文档。\n\n---\n\n",
        "\n\n---\n\n请指出上文中提到的一个性能瓶颈,并解释原因。",
    ),
    (
        4096,
        "以下是一份较长的技术文档节选(可能在中途截断)。\n\n---\n\n",
        "\n\n---\n\n请用一段话总结上文的主要内容。",
    ),
    (
        4096,
        "The following is a long excerpt from a set of engineering notes.\n\n"
        "---\n\n",
        "\n\n---\n\nWhat is the single most important conclusion above?",
    ),
    (
        8192,
        "以下是一份很长的技术文档节选(可能在中途截断)。\n\n---\n\n",
        "\n\n---\n\n请用三句话概括上文的主要内容。",
    ),
    (
        8192,
        "The following is a very long excerpt from a set of engineering "
        "notes.\n\n---\n\n",
        "\n\n---\n\nSummarize the above in three sentences.",
    ),
)


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_corpus(repo_root: Path) -> tuple[str, list[dict]]:
    """Concatenate the source documents in the fixed order."""

    parts: list[str] = []
    provenance: list[dict] = []
    cursor = 0
    for relative in SOURCES:
        path = repo_root / relative
        text = path.read_text(encoding="utf-8")
        if "｜" in text:
            raise ValueError(
                f"{relative} contains the DeepSeek special-token character "
                "U+FF5C; it must not enter the corpus"
            )
        provenance.append(
            {
                "path": relative,
                "md5": file_md5(path),
                "chars": len(text),
                "corpus_offset": cursor,
            }
        )
        parts.append(text)
        cursor += len(text) + len(CORPUS_SEPARATOR)
    return CORPUS_SEPARATOR.join(parts), provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--encoding-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.encoding_dir.expanduser().resolve()))
    from encoding_dsv4 import encode_messages  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.expanduser()))
    corpus, provenance = build_corpus(args.repo_root.expanduser().resolve())

    def token_length(prompt: str) -> int:
        return len(
            tokenizer.encode(
                encode_messages(
                    [{"role": "user", "content": prompt}], thinking_mode="chat"
                )
            )
        )

    records: list[dict] = []
    cursor = 0
    for index, (target, header, footer) in enumerate(SPECS):
        # Binary search the largest body character count whose templated prompt
        # is still <= target tokens.
        low, high = 0, min(len(corpus) - cursor, target * 8)
        if token_length(header + corpus[cursor : cursor + high] + footer) < target:
            raise ValueError(
                f"corpus exhausted for prompt {index} (target {target}); "
                "add more source documents"
            )
        while low < high:
            middle = (low + high + 1) // 2
            body = corpus[cursor : cursor + middle]
            if token_length(header + body + footer) <= target:
                low = middle
            else:
                high = middle - 1

        # The prompt token count must land *exactly* on the target.  1023 rows
        # would sit below the vLLM ``>= 1024`` kernel branch that the fused HC
        # backend's MAX_ROWS=896 split exists to avoid, and an odd row count
        # makes the row-blocked MoE overlap fall back to its sequential path --
        # either way the lever under test would silently not be exercised and
        # the gate would report a meaningless "no harm".
        #
        # Cutting the body cannot always reach the target: token length is
        # non-decreasing in the body length but can step by 2 when the added
        # character splits a BPE merge, so the target is skipped.  Pad instead:
        # append a short filler to the *body* (the body is already truncated
        # mid-sentence, and the prompt still ends with its clean instruction)
        # and grow it one unit at a time until the count matches exactly.
        body = corpus[cursor : cursor + low]
        prompt = header + body + footer
        length = token_length(prompt)
        if length < target:
            for filler in FILLERS:
                padded, guard = body, 0
                while token_length(header + padded + footer) < target and guard < 4 * WINDOW:
                    padded += filler
                    guard += 1
                if token_length(header + padded + footer) == target:
                    body = padded
                    prompt = header + body + footer
                    break
        tokens = tokenizer.encode(
            encode_messages(
                [{"role": "user", "content": prompt}], thinking_mode="chat"
            )
        )
        records.append(
            {
                "index": index,
                "target_tokens": target,
                "prompt_tokens_len": len(tokens),
                "exact": len(tokens) == target,
                "body_chars": len(body),
                "corpus_chars": low,
                "filler_chars": len(body) - low,
                "corpus_window": [cursor, cursor + low],
                "prompt": prompt,
            }
        )
        print(
            f"prompt {index}: target {target} -> {len(tokens)} tokens "
            f"({low} corpus chars + {len(body) - low} filler, "
            f"corpus [{cursor}, {cursor + low}))",
            flush=True,
        )
        # Disjoint windows: each prompt reads fresh text.
        cursor += low

    payload = {
        "experiment": "D0L-long-prompt-oracle",
        "purpose": (
            "long prompts whose single start_pos=0 prefill lands in the C2F "
            "chunk regime (1024/2048/4096/8192 rows per lane)"
        ),
        "corpus": {
            "separator": CORPUS_SEPARATOR,
            "total_chars": len(corpus),
            "consumed_chars": cursor,
            "sources": provenance,
        },
        "tokenizer": str(args.tokenizer),
        "chat_template": "encode_messages(thinking_mode='chat')",
        "length_histogram": {
            str(target): sum(1 for r in records if r["target_tokens"] == target)
            for target in sorted({spec[0] for spec in SPECS})
        },
        "prompts": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"WROTE {args.out} ({len(records)} prompts)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
