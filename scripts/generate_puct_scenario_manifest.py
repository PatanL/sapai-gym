#!/usr/bin/env python3
"""Build deterministic scenario-state manifests for search-only exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sapai_gym1.rl.curriculum import (
    PUCT_SCENARIO_NAMES,
    PUCT_SCENARIO_VERSION,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario", action="append", choices=PUCT_SCENARIO_NAMES
    )
    parser.add_argument("--seeds-per-scenario", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2853800)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seeds_per_scenario <= 0:
        raise SystemExit("--seeds-per-scenario must be positive")
    scenarios = list(args.scenario or PUCT_SCENARIO_NAMES)
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        for example in range(args.seeds_per_scenario):
            rows.append(
                {
                    "scenario": scenario,
                    "seed": args.seed + scenario_index * 100_000 + example,
                    "selectionOnly": True,
                }
            )
    payload = {
        "scenarioVersion": PUCT_SCENARIO_VERSION,
        "selectionPolicy": (
            "Scenario builders select public starting states only. Their "
            "reference actions are metadata and are not training labels."
        ),
        "scenarios": scenarios,
        "seedsPerScenario": args.seeds_per_scenario,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"states": len(rows), "scenarios": len(scenarios)}))


if __name__ == "__main__":
    main()
