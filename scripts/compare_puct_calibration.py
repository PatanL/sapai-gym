#!/usr/bin/env python3
"""Compare matched raw-critic and calibrated-critic PUCT benchmarks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("calibrated", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values):
    values = list(map(float, values))
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standardError": None,
            "lowerBound95": None,
            "upperBound95": None,
        }
    mean = statistics.mean(values)
    if len(values) < 2:
        return {
            "count": len(values),
            "mean": mean,
            "standardError": None,
            "lowerBound95": None,
            "upperBound95": None,
        }
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": mean,
        "standardError": standard_error,
        "lowerBound95": mean - 1.96 * standard_error,
        "upperBound95": mean + 1.96 * standard_error,
    }


def validate_matched(raw, calibrated):
    for field in ("checkpoint", "simulatorRevision", "stateMode", "states"):
        if raw.get(field) != calibrated.get(field):
            raise ValueError(f"Mismatched {field}")
    if len(raw["rows"]) != len(calibrated["rows"]):
        raise ValueError("Mismatched row count")
    for index, (left, right) in enumerate(
        zip(raw["rows"], calibrated["rows"])
    ):
        identity_fields = (
            "turn",
            "side",
            "stateMode",
            "opportunityReasons",
            "publicState",
        )
        for field in identity_fields:
            if left.get(field) != right.get(field):
                raise ValueError(f"Row {index} mismatched {field}")


def report_metrics(payload):
    rows = payload["rows"]
    return {
        "states": len(rows),
        "rootAccepted": sum(row["root"]["accepted"] for row in rows),
        "puctAccepted": sum(row["puct"]["accepted"] for row in rows),
        "puctIndependentlyConfirmed": sum(
            row["puct"].get("independentlyConfirmed", False)
            for row in rows
        ),
        "rootProposedVsRaw": mean_ci(
            row["root"]["independentProposedReturnAdvantageVsRaw"]
            for row in rows
        ),
        "puctProposedVsRaw": mean_ci(
            row["puct"]["independentProposedReturnAdvantageVsRaw"]
            for row in rows
        ),
        "puctProposedRootOnlyVsRaw": mean_ci(
            row["puct"][
                "independentProposedRootOnlyReturnAdvantageVsRaw"
            ]
            for row in rows
        ),
        "puctPlanningContinuationLift": mean_ci(
            row["puct"]["independentPlanningContinuationLift"]
            for row in rows
        ),
        "puctMinusRootProposed": mean_ci(
            row["puctMinusRootProposedReturn"] for row in rows
        ),
        "meanValidationTreeActions": statistics.mean(
            row["puct"]["validationTreeActionsMean"] for row in rows
        ),
        "meanPuctWallSeconds": statistics.mean(
            row["puct"]["wallSeconds"] for row in rows
        ),
    }


def reason_metrics(rows):
    values = defaultdict(list)
    for row in rows:
        reasons = row.get("opportunityReasons") or ["general"]
        for reason in set(reasons):
            values[reason].append(
                row["puct"]["independentProposedReturnAdvantageVsRaw"]
            )
    return {reason: mean_ci(group) for reason, group in sorted(values.items())}


def compare(raw, calibrated):
    validate_matched(raw, calibrated)
    pairs = list(zip(raw["rows"], calibrated["rows"]))
    action_changes = Counter()
    changed_rows = []
    for index, (left, right) in enumerate(pairs):
        for planner in ("root", "puct"):
            changed = (
                left[planner]["proposedAction"]
                != right[planner]["proposedAction"]
            )
            action_changes[planner] += int(changed)
            if changed:
                changed_rows.append(
                    {
                        "state": index,
                        "turn": left["turn"],
                        "reasons": left.get("opportunityReasons", []),
                        "planner": planner,
                        "rawAction": left[planner].get(
                            "proposedActionDetail",
                            left[planner]["proposedAction"],
                        ),
                        "calibratedAction": right[planner].get(
                            "proposedActionDetail",
                            right[planner]["proposedAction"],
                        ),
                    }
                )
    paired_effects = {
        "puctProposedVsRaw": mean_ci(
            right["puct"]["independentProposedReturnAdvantageVsRaw"]
            - left["puct"]["independentProposedReturnAdvantageVsRaw"]
            for left, right in pairs
        ),
        "puctProposedRootOnlyVsRaw": mean_ci(
            right["puct"][
                "independentProposedRootOnlyReturnAdvantageVsRaw"
            ]
            - left["puct"][
                "independentProposedRootOnlyReturnAdvantageVsRaw"
            ]
            for left, right in pairs
        ),
        "puctPlanningContinuationLift": mean_ci(
            right["puct"]["independentPlanningContinuationLift"]
            - left["puct"]["independentPlanningContinuationLift"]
            for left, right in pairs
        ),
    }
    return {
        "schemaVersion": 1,
        "checkpoint": raw.get("checkpoint"),
        "simulatorRevision": raw.get("simulatorRevision"),
        "states": len(pairs),
        "raw": report_metrics(raw),
        "calibrated": report_metrics(calibrated),
        "calibratedMinusRaw": paired_effects,
        "proposalActionChanges": dict(sorted(action_changes.items())),
        "changedRows": changed_rows,
        "rawByOpportunityReason": reason_metrics(raw["rows"]),
        "calibratedByOpportunityReason": reason_metrics(
            calibrated["rows"]
        ),
    }


def main():
    args = parse_args()
    raw = json.loads(args.raw.read_text())
    calibrated = json.loads(args.calibrated.read_text())
    result = compare(raw, calibrated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
