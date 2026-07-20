#!/usr/bin/env python3
"""E0f: validate Flash checkpoint metadata and TP4 intermediate slices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dsv4_direct.checkpoint import CheckpointContractError, inspect_stage_checkpoint
from dsv4_direct.model_contract import MTP_LAYER_ID


def parse_layer_ids(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        # "mtp" selects the mtp.0 block (frozen pseudo-id 43).
        result.append(MTP_LAYER_ID if item.lower() == "mtp" else int(item))
    if not result:
        raise argparse.ArgumentTypeError("at least one layer id is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument(
        "--layers", type=parse_layer_ids, default=parse_layer_ids("0,1,2,3,4,42,mtp")
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = inspect_stage_checkpoint(args.stage_root, args.layers, args.tp_size)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, CheckpointContractError) as exc:
        result = {
            "schema_version": 1,
            "experiment": "e0f-flash-checkpoint-contract",
            "ok": False,
            "stage_root": str(args.stage_root.expanduser()),
            "errors": [f"{type(exc).__name__}: {exc}"],
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
