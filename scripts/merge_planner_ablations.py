#!/usr/bin/env python3
"""Merge matched raw/root/nested planner-depth ablation shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PLANNERS = ("root", "nested1", "nested2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None, None
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return (
        mean,
        standard_error,
        mean - 1.96 * standard_error,
        mean + 1.96 * standard_error,
    )


def aggregate(rows, planner):
    selected = [row["planners"][planner] for row in rows]
    return {
        "states": len(selected),
        "meanOuterReturnAdvantageVsRaw": float(
            np.mean(
                [item["outerReturnAdvantageVsRaw"] for item in selected]
            )
        ),
        "meanOuterFirstBattleScoreAdvantageVsRaw": float(
            np.mean(
                [
                    item["outerFirstBattleScoreAdvantageVsRaw"]
                    for item in selected
                ]
            )
        ),
        "meanOuterHorizonBattleScoreAdvantageVsRaw": float(
            np.mean(
                [
                    item["outerHorizonBattleScoreAdvantageVsRaw"]
                    for item in selected
                ]
            )
        ),
        "meanWallSeconds": float(
            np.mean([item["wallSeconds"] for item in selected])
        ),
        "meanContinuationStates": float(
            np.mean(
                [item["compute"]["continuationStates"] for item in selected]
            )
        ),
    }


def paired_difference(rows, field, left="nested2", right="root"):
    values = [
        row["planners"][left][field]
        - row["planners"][right][field]
        for row in rows
    ]
    mean, standard_error, lower, upper = mean_ci(values)
    return {
        "states": len(values),
        "mean": mean,
        "standardError": standard_error,
        "lowerBound95": lower,
        "upperBound95": upper,
    }


def main():
    args = parse_args()
    if not args.inputs:
        raise SystemExit("at least one input report is required")
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    checkpoints = {payload["checkpoint"] for payload in payloads}
    revisions = {payload["simulatorRevision"] for payload in payloads}
    search_versions = {payload["searchVersion"] for payload in payloads}
    planner_configs = {
        json.dumps(payload["plannerConfig"], sort_keys=True)
        for payload in payloads
    }
    if (
        len(checkpoints) != 1
        or len(revisions) != 1
        or len(search_versions) != 1
        or len(planner_configs) != 1
    ):
        raise SystemExit(
            "incompatible shards: "
            f"checkpoints={checkpoints}, revisions={revisions}, "
            f"search_versions={search_versions}"
        )
    rows = []
    for shard, payload in enumerate(payloads):
        for row in payload["rows"]:
            row = dict(row)
            row["localState"] = row["state"]
            row["state"] = len(rows)
            row["shard"] = shard
            rows.append(row)
    summaries = {planner: aggregate(rows, planner) for planner in PLANNERS}
    horizon_difference = paired_difference(
        rows, "outerHorizonBattleScoreAdvantageVsRaw"
    )
    return_difference = paired_difference(
        rows, "outerReturnAdvantageVsRaw"
    )
    compute_ratio = (
        summaries["nested2"]["meanContinuationStates"]
        / max(1.0, summaries["root"]["meanContinuationStates"])
    )
    lower = horizon_difference["lowerBound95"]
    upper = horizon_difference["upperBound95"]
    if lower is not None and lower >= 0.03 and compute_ratio <= 5.0:
        recommendation = "prototype-puct"
    elif upper is not None and upper <= 0.01:
        recommendation = "keep-root-search"
    else:
        recommendation = "expand-planner-ablation"
    merged = {
        "checkpoint": checkpoints.pop(),
        "searchVersion": search_versions.pop(),
        "plannerConfig": json.loads(planner_configs.pop()),
        "simulatorRevision": revisions.pop(),
        "shards": len(payloads),
        "states": len(rows),
        "samplingAttempts": sum(
            int(payload["samplingAttempts"]) for payload in payloads
        ),
        "summaries": summaries,
        "nested2MinusRootHorizonBattleScore": horizon_difference,
        "nested2MinusRootReturn": return_difference,
        "nested2ComputeRatio": compute_ratio,
        "recommendation": recommendation,
        "sourceReports": [str(path) for path in args.inputs],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: merged[k] for k in (
        "states",
        "summaries",
        "nested2MinusRootHorizonBattleScore",
        "nested2MinusRootReturn",
        "nested2ComputeRatio",
        "recommendation",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
