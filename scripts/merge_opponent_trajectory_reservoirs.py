#!/usr/bin/env python3
"""Merge corrected-runtime trajectory reservoirs from diverse frozen policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sapai_gym1.rl.opponent_reservoir import RESERVOIR_SCHEMA_VERSION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def merge_reservoirs(inputs):
    inputs = [Path(path) for path in inputs]
    payloads = [json.loads(path.read_text()) for path in inputs]
    if not payloads:
        raise SystemExit("at least one reservoir is required")
    if any(
        payload.get("schemaVersion") != RESERVOIR_SCHEMA_VERSION
        for payload in payloads
    ):
        raise SystemExit("incompatible opponent-reservoir schema")
    revisions = {
        payload.get("metadata", {}).get("simulatorRevision")
        for payload in payloads
    }
    if len(revisions) != 1 or None in revisions:
        raise SystemExit(f"incompatible simulator revisions: {revisions}")

    trajectories = []
    components = []
    for source_index, (path, payload) in enumerate(zip(inputs, payloads)):
        metadata = payload.get("metadata", {})
        component_trajectories = payload.get("trajectories", [])
        prefix = f"source-{source_index:02d}-"
        for trajectory in component_trajectories:
            trajectories.append(
                {
                    **trajectory,
                    "id": prefix + str(trajectory.get("id", len(trajectories))),
                    "sourceIndex": source_index,
                }
            )
        components.append(
            {
                "sourceIndex": source_index,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "checkpoint": metadata.get("checkpoint"),
                "checkpointSha256": metadata.get("checkpointSha256"),
                "checkpointTrainingRevision": metadata.get(
                    "checkpointTrainingRevision"
                ),
                "games": metadata.get("games"),
                "trajectories": len(component_trajectories),
                "seed": metadata.get("seed"),
            }
        )
    return {
        "schemaVersion": RESERVOIR_SCHEMA_VERSION,
        "metadata": {
            "simulatorRevision": revisions.pop(),
            "games": sum(int(row.get("games") or 0) for row in components),
            "trajectories": len(trajectories),
            "mixture": "diverse-frozen-policy-corrected-runtime",
            "components": components,
        },
        "trajectories": trajectories,
    }
def main():
    args = parse_args()
    merged = merge_reservoirs(args.inputs)
    trajectories = merged["trajectories"]
    components = merged["metadata"]["components"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "simulatorRevision": merged["metadata"]["simulatorRevision"],
                "games": merged["metadata"]["games"],
                "trajectories": len(trajectories),
                "components": len(components),
                "sha256": hashlib.sha256(
                    args.output.read_bytes()
                ).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
