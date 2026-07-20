#!/usr/bin/env python3
"""E0e2e pre-gate: certify the full-position ratio-4 attention path.

``dsv4_direct/ratio4_fullpos.py`` supplies the ratio-4 prefill and
unrestricted-position decode that the E2E golden gate needs (the verified
plan paths are frozen to positions >= 128).  This harness backs it with two
real-weight checks on one GPU (layer 2, TP rank-0 replicated weights):

1. **Decode mirror (bitwise gate).**  The full-position lane runs
   prefill(16) + teacher-forced decode up to a saturated phase-0 position;
   its state is installed into a ``StaticRatio4KV`` via
   ``seed_decode_payload`` and the E0ff-verified
   ``Ratio4TorchAttention.forward_decode_tensor`` plan path runs the next
   16 positions side by side on identical hiddens.  Branch outputs must be
   bitwise identical at every step -- this pins the new decode step to the
   verified candidate on the shared domain.
2. **Prefill/incremental consistency (tolerance gate).**  Two lanes consume
   the identical per-position hidden stream, one entering via prefill(12)
   and one via prefill(20).  From position 20 onward their branch outputs
   must agree within a BF16 kernel-shape tolerance (full-sequence GEMMs vs
   single-row GEMMs legitimately differ in low bits), certifying that the
   reference's prefill and incremental compressor/indexer state updates are
   consistent in this implementation.  This window (positions 12-40) also
   exercises the padded window branch and the first compression boundaries
   from the empty state.

Run (titan064):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ~/Workspace/venvs/sglang/bin/python e0e2e_ratio4_selfcheck.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0e2e-selfcheck
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    Ratio4TorchAttention,
    prepare_ratio4_attention_weights,
)
from dsv4_direct.ratio4_fullpos import Ratio4FullPositionAttention
from dsv4_direct.static_ratio4_kv import StaticRatio4KV


LAYER_ID = 2
MAX_SEQ_LEN = 256
MIRROR_SEED_POSITION = 192
MIRROR_STEPS = 16
CONSISTENCY_PREFILLS = (12, 20)
CONSISTENCY_STOP = 40
CONSISTENCY_RMS_REL_LIMIT = 0.02  # E0wf/E0ef "branch" scale (BF16 lane pair)


def deterministic_hidden(seed: int, seqlen: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(1, seqlen, 4096, generator=generator, dtype=torch.float32)
    return (value * 0.02).to(torch.bfloat16).to(device)


def rms_rel(observed: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (observed.float() - expected.float())
    rms_abs = float(difference.square().mean().sqrt().item())
    reference = float(expected.float().square().mean().sqrt().item())
    return rms_abs / max(reference, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", 0)
    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0e2e-ratio4-fullpos-selfcheck",
        "measurement_class": "semantic_correctness_gate",
        "layer_id": LAYER_ID,
        "checkpoint_id": None,
        "decode_mirror": None,
        "prefill_consistency": None,
        "accepted": False,
        "errors": [],
    }
    started = time.perf_counter()
    try:
        checkpoint = inspect_stage_checkpoint(stage_root, [LAYER_ID], 4)
        if not checkpoint["ok"]:
            raise ValueError(f"checkpoint contract failed: {checkpoint['errors'][:3]}")
        result["checkpoint_id"] = checkpoint["checkpoint_id"]
        config_payload = json.loads(
            (stage_root / "config.json").read_text(encoding="utf-8")
        )
        config = Ratio4AttentionConfig.from_model_config(
            config_payload, layer_id=LAYER_ID, max_seq_len=MAX_SEQ_LEN
        )
        raw_block = load_replicated_block_weights(
            stage_root=stage_root,
            rank=0,
            world_size=4,
            layer_id=LAYER_ID,
            device=device,
            checkpoint_id=result["checkpoint_id"],
        )
        prepared = prepare_ratio4_attention_weights(
            raw_block.attention,
            layer_id=LAYER_ID,
            rank=0,
            world_size=4,
            checkpoint_id=result["checkpoint_id"],
        )

        # ------------------------------------------------------------------
        # 1. decode mirror vs the E0ff-verified plan path
        lane = Ratio4FullPositionAttention(
            config, prepared, batch_size=1, device=device
        )
        prefill_len = 16
        lane(deterministic_hidden(args.seed, prefill_len, device), start_pos=0)
        for position in range(prefill_len, MIRROR_SEED_POSITION):
            lane(
                deterministic_hidden(args.seed + 7919 * position, 1, device),
                start_pos=position,
            )
        candidate_state = StaticRatio4KV(
            num_local_sequences=1,
            max_seq_len=MAX_SEQ_LEN,
            layer_id=LAYER_ID,
            device=device,
        )
        candidate_state.seed_decode_payload(
            MIRROR_SEED_POSITION,
            raw=lane.raw.clone(),
            compressed=lane.compressed.clone(),
            indexer_kv=lane.indexer_kv.clone(),
            main_kv_state=lane.main_kv_state.clone(),
            main_score_state=lane.main_score_state.clone(),
            index_kv_state=lane.index_kv_state.clone(),
            index_score_state=lane.index_score_state.clone(),
        )
        candidate = Ratio4TorchAttention(config, prepared, candidate_state)
        mirror_steps = []
        for position in range(
            MIRROR_SEED_POSITION, MIRROR_SEED_POSITION + MIRROR_STEPS
        ):
            hidden = deterministic_hidden(args.seed + 7919 * position, 1, device)
            plan = candidate.prepare_decode_plan(
                position, advance_overlap_state=True
            )
            candidate_branch = candidate.forward_decode_tensor(
                hidden.clone(), start_pos=position, plan=plan
            )
            fullpos_branch = lane(hidden.clone(), start_pos=position)
            mirror_steps.append(
                {
                    "position": position,
                    "bitwise": bool(torch.equal(candidate_branch, fullpos_branch)),
                    "max_abs": float(
                        (candidate_branch.float() - fullpos_branch.float())
                        .abs()
                        .max()
                        .item()
                    ),
                }
            )
        state_bitwise = {
            "raw": bool(torch.equal(lane.raw, candidate_state.raw)),
            "compressed": bool(
                torch.equal(lane.compressed, candidate_state.compressed)
            ),
            "indexer_kv": bool(
                torch.equal(lane.indexer_kv, candidate_state.indexer_kv)
            ),
        }
        result["decode_mirror"] = {
            "seed_position": MIRROR_SEED_POSITION,
            "steps": mirror_steps,
            "state_bitwise": state_bitwise,
            "accepted": all(step["bitwise"] for step in mirror_steps)
            and all(state_bitwise.values()),
        }

        # ------------------------------------------------------------------
        # 2. prefill vs incremental consistency on one hidden stream
        def position_hidden(position: int) -> torch.Tensor:
            return deterministic_hidden(args.seed + 104729 * position, 1, device)

        lanes: dict[int, Ratio4FullPositionAttention] = {}
        outputs: dict[int, dict[int, torch.Tensor]] = {}
        for prefill in CONSISTENCY_PREFILLS:
            lane = Ratio4FullPositionAttention(
                config, prepared, batch_size=1, device=device
            )
            stream = torch.cat(
                [position_hidden(position) for position in range(prefill)], dim=1
            )
            lane(stream, start_pos=0)
            outputs[prefill] = {}
            for position in range(prefill, CONSISTENCY_STOP):
                outputs[prefill][position] = lane(
                    position_hidden(position), start_pos=position
                )
            lanes[prefill] = lane
        short, long = CONSISTENCY_PREFILLS
        consistency_steps = []
        for position in range(long, CONSISTENCY_STOP):
            left = outputs[short][position]
            right = outputs[long][position]
            consistency_steps.append(
                {
                    "position": position,
                    "bitwise": bool(torch.equal(left, right)),
                    "rms_rel": rms_rel(left, right),
                }
            )
        state_rms = {
            "raw": rms_rel(lanes[short].raw, lanes[long].raw),
            "compressed": rms_rel(
                lanes[short].compressed, lanes[long].compressed
            ),
            "indexer_kv": rms_rel(
                lanes[short].indexer_kv, lanes[long].indexer_kv
            ),
        }
        result["prefill_consistency"] = {
            "prefills": list(CONSISTENCY_PREFILLS),
            "stop": CONSISTENCY_STOP,
            "steps": consistency_steps,
            "state_rms_rel": state_rms,
            "rms_rel_limit": CONSISTENCY_RMS_REL_LIMIT,
            "accepted": all(
                step["rms_rel"] <= CONSISTENCY_RMS_REL_LIMIT
                for step in consistency_steps
            )
            and all(
                value <= CONSISTENCY_RMS_REL_LIMIT for value in state_rms.values()
            ),
        }

        result["accepted"] = bool(
            result["decode_mirror"]["accepted"]
            and result["prefill_consistency"]["accepted"]
        )
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
    result["seconds"] = time.perf_counter() - started

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selfcheck.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[ratio4-selfcheck] decode_mirror="
        f"{result['decode_mirror'] and result['decode_mirror']['accepted']} "
        f"prefill_consistency="
        f"{result['prefill_consistency'] and result['prefill_consistency']['accepted']} "
        f"overall={'PASS' if result['accepted'] else 'FAIL'}",
        flush=True,
    )
    if result["errors"]:
        print(result["errors"][0], flush=True)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
