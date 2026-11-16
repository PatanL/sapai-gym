#!/usr/bin/env python3
"""Create a mergeable planner report from completed JSON rows on stdin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from benchmark_planner_depth import aggregate
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.rollout_search import SEARCH_VERSION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timed-out-states", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for line in sys.stdin:
        if not line.strip():
            continue
        payload = json.loads(line)
        if "publicState" in payload and "planners" in payload:
            rows.append(payload)
    if not rows:
        raise SystemExit("no completed planner rows on stdin")
    planner_names = ("root", "nested1", "nested2")
    summaries = {
        planner: aggregate(rows, planner) for planner in planner_names
    }
    nested2_gain = (
        summaries["nested2"]["meanOuterHorizonBattleScoreAdvantageVsRaw"]
        - summaries["root"]["meanOuterHorizonBattleScoreAdvantageVsRaw"]
    )
    compute_ratio = (
        summaries["nested2"]["meanContinuationStates"]
        / max(1.0, summaries["root"]["meanContinuationStates"])
    )
    payload = {
        "checkpoint": args.checkpoint,
        "searchVersion": SEARCH_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "plannerConfig": {
            "rolloutsPerAction": 4,
            "confirmationRollouts": 16,
            "validationRollouts": 32,
            "horizonBattles": 5,
            "leafValueWeight": 0.5,
            "opponentCandidates": 32,
            "canonicalizeEarlyMoves": True,
            "pruneAffordableFreezes": False,
        },
        "states": len(rows),
        "samplingAttempts": len(rows) + args.timed_out_states,
        "timedOutStates": args.timed_out_states,
        "summaries": summaries,
        "nested2MinusRootHorizonBattleScore": nested2_gain,
        "nested2ComputeRatio": compute_ratio,
        "recommendation": "reject-naive-nesting-unbounded-cost",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "states": len(rows),
        "timedOutStates": args.timed_out_states,
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
