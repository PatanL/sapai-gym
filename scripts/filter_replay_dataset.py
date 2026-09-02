#!/usr/bin/env python3
"""Filter validated replay JSONL rows into a curriculum dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sapai.turtle_pack import SIMULATOR_REVISION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-turn", type=int, default=1)
    parser.add_argument("--max-turn", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_turn < 1:
        raise SystemExit("--min-turn must be positive")
    if args.max_turn is not None and args.max_turn < args.min_turn:
        raise SystemExit("--max-turn must be >= --min-turn")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    read = written = 0
    replay_ids = set()
    with args.output.open("w") as output:
        for path in args.inputs:
            with path.open() as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    read += 1
                    row = json.loads(line)
                    revision = row.get("validation", {}).get(
                        "simulatorRevision"
                    )
                    if revision != SIMULATOR_REVISION:
                        raise SystemExit(
                            f"{path}:{line_number}: simulator revision "
                            f"{revision!r} != {SIMULATOR_REVISION!r}"
                        )
                    turn = int(row["turn"])
                    if turn < args.min_turn:
                        continue
                    if args.max_turn is not None and turn > args.max_turn:
                        continue
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    replay_ids.add(row["replay_id"])
                    written += 1
    print(
        f"read={read} written={written} replays={len(replay_ids)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
