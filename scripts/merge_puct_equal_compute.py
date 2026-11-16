#!/usr/bin/env python3
"""Merge equal-compute root-versus-PUCT benchmark shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return {
            "mean": mean,
            "standardError": None,
            "lowerBound95": None,
            "upperBound95": None,
        }
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "mean": mean,
        "standardError": standard_error,
        "lowerBound95": mean - 1.96 * standard_error,
        "upperBound95": mean + 1.96 * standard_error,
    }


def main():
    args = parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    if not payloads:
        raise SystemExit("at least one shard is required")
    metadata_fields = (
        "puctVersion",
        "searchVersion",
        "simulatorRevision",
        "checkpoint",
        "miningVersion",
    )
    metadata = {}
    for field in metadata_fields:
        values = {
            payload.get(field, "legacy-first-opportunity")
            if field == "miningVersion" else payload[field]
            for payload in payloads
        }
        if len(values) != 1:
            raise SystemExit(f"incompatible {field}: {values}")
        metadata[field] = values.pop()
    planner_configs = {
        json.dumps(payload.get("plannerConfig"), sort_keys=True)
        for payload in payloads
    }
    if len(planner_configs) != 1 or "null" in planner_configs:
        raise SystemExit(
            f"incompatible plannerConfig: {sorted(planner_configs)}"
        )
    metadata["plannerConfig"] = payloads[0]["plannerConfig"]
    state_modes = {
        payload.get("stateMode", "uniform-turn") for payload in payloads
    }
    if len(state_modes) != 1:
        raise SystemExit(f"incompatible stateMode: {sorted(state_modes)}")
    metadata["stateMode"] = state_modes.pop()
    metadata["sourceTurnWindows"] = [
        payload.get("turnWindow") for payload in payloads
    ]
    rows = []
    for shard, payload in enumerate(payloads):
        for row in payload["rows"]:
            row = dict(row)
            # A short-lived telemetry-only benchmark revision serialized the
            # PUCT phase transition counters under ``root``. Normalize those
            # already-running shards during merge; planner decisions and
            # numeric results were unaffected.
            if (
                "phaseSimulatorTransitions" not in row["puct"]
                and "phaseSimulatorTransitions" in row["root"]
            ):
                row["root"] = dict(row["root"])
                row["puct"] = dict(row["puct"])
                row["puct"]["phaseSimulatorTransitions"] = row[
                    "root"
                ].pop("phaseSimulatorTransitions")
            row["localState"] = row["state"]
            row["state"] = len(rows)
            row["shard"] = shard
            rows.append(row)
    root_advantages = [row["root"]["returnAdvantageVsRaw"] for row in rows]
    puct_advantages = [row["puct"]["returnAdvantageVsRaw"] for row in rows]
    root_proposal_advantages = [
        row["root"]["independentProposedReturnAdvantageVsRaw"]
        for row in rows
    ]
    puct_proposal_advantages = [
        row["puct"]["independentProposedReturnAdvantageVsRaw"]
        for row in rows
    ]
    puct_root_only_proposal_advantages = [
        row["puct"].get(
            "independentProposedRootOnlyReturnAdvantageVsRaw"
        )
        for row in rows
    ]
    puct_planning_continuation_lifts = [
        row["puct"].get("independentPlanningContinuationLift")
        for row in rows
    ]
    differences = [row["puctMinusRootReturn"] for row in rows]
    proposal_differences = [
        row["puctMinusRootProposedReturn"] for row in rows
    ]
    summary = {
        "states": len(rows),
        "rootAccepted": sum(row["root"]["accepted"] for row in rows),
        "puctAccepted": sum(row["puct"]["accepted"] for row in rows),
        "puctIndependentlyConfirmed": sum(
            row["puct"].get("independentlyConfirmed", False)
            for row in rows
        ),
        "puctUniqueProposals": sum(
            row["puct"]["proposedAction"]
            != row["root"]["proposedAction"]
            for row in rows
        ),
        "puctUniqueIndependentlyConfirmed": sum(
            row["puct"].get("independentlyConfirmed", False)
            and row["puct"]["proposedAction"]
            != row["root"]["proposedAction"]
            for row in rows
        ),
        "rootVsRaw": mean_ci(root_advantages),
        "puctVsRaw": mean_ci(puct_advantages),
        "rootProposedVsRaw": mean_ci(root_proposal_advantages),
        "puctProposedVsRaw": mean_ci(puct_proposal_advantages),
        **(
            {
                "puctProposedRootOnlyVsRaw": mean_ci(
                    puct_root_only_proposal_advantages
                ),
                "puctPlanningContinuationLift": mean_ci(
                    puct_planning_continuation_lifts
                ),
            }
            if all(
                value is not None
                for value in puct_root_only_proposal_advantages
                + puct_planning_continuation_lifts
            )
            else {}
        ),
        "puctMinusRoot": mean_ci(differences),
        "puctMinusRootProposed": mean_ci(proposal_differences),
        "meanRootWallSeconds": float(
            np.mean([row["root"]["wallSeconds"] for row in rows])
        ),
        "meanPuctWallSeconds": float(
            np.mean([row["puct"]["wallSeconds"] for row in rows])
        ),
        "meanBudgetUtilization": float(
            np.mean(
                [
                    row["puct"]["simulatorTransitions"]
                    / row["selectionTransitionBudget"]
                    for row in rows
                ]
            )
        ),
        "meanSelectionTransitionBudget": float(
            np.mean([row["selectionTransitionBudget"] for row in rows])
        ),
        "meanPuctTreeNodes": float(
            np.mean([row["puct"]["treeNodes"] for row in rows])
        ),
        "meanPuctTrainingTargets": float(
            np.mean([row["puct"]["trainingTargets"] for row in rows])
        ),
        **(
            {
                "meanRootPhaseWallSeconds": {
                    phase: float(
                        np.mean(
                            [
                                row["root"]["phaseWallSeconds"][phase]
                                for row in rows
                            ]
                        )
                    )
                    for phase in ("selection", "validation", "audit")
                }
            }
            if all("phaseWallSeconds" in row["root"] for row in rows)
            else {}
        ),
        **(
            {
                "meanPuctPhaseWallSeconds": {
                    phase: float(
                        np.mean(
                            [
                                row["puct"]["phaseWallSeconds"][phase]
                                for row in rows
                            ]
                        )
                    )
                    for phase in ("tree", "screening", "validation", "audit")
                }
            }
            if all("phaseWallSeconds" in row["puct"] for row in rows)
            else {}
        ),
        **(
            {
                "meanPuctPhaseSimulatorTransitions": {
                    phase: float(
                        np.mean(
                            [
                                row["puct"]["phaseSimulatorTransitions"][phase]
                                for row in rows
                            ]
                        )
                    )
                    for phase in ("tree", "screening", "validation", "audit")
                }
            }
            if all(
                "phaseSimulatorTransitions" in row["puct"] for row in rows
            )
            else {}
        ),
    }
    if all("phaseSimulatorTransitions" in row["puct"] for row in rows):
        puct_selection_transitions = [
            row["puct"]["phaseSimulatorTransitions"]["tree"]
            + row["puct"]["phaseSimulatorTransitions"]["screening"]
            for row in rows
        ]
        root_selection_transitions = [
            row["root"]["selectionSimulatorTransitions"] for row in rows
        ]
        ratios = [
            puct / root
            for puct, root in zip(
                puct_selection_transitions, root_selection_transitions
            )
        ]
        summary["meanPuctSelectionSimulatorTransitions"] = float(
            np.mean(puct_selection_transitions)
        )
        summary["meanRootSelectionSimulatorTransitions"] = float(
            np.mean(root_selection_transitions)
        )
        summary["meanPuctToRootSelectionTransitionRatio"] = float(
            np.mean(ratios)
        )
    lower = summary["puctMinusRootProposed"]["lowerBound95"]
    upper = summary["puctMinusRootProposed"]["upperBound95"]
    selection_only = bool(metadata["plannerConfig"].get("selectionOnly"))
    externally_confirmed = bool(
        metadata["plannerConfig"].get("externalConfirmationOnly")
    )
    budget_ratio = summary.get("meanPuctToRootSelectionTransitionRatio")
    budget_matched = bool(
        (selection_only or externally_confirmed)
        and budget_ratio is not None
        and 0.90 <= budget_ratio <= 1.10
    )
    if lower is not None and lower > 0.0:
        recommendation = (
            "puct-more-transition-efficient"
            if budget_matched
            else "puct-higher-return-at-unequal-total-work"
        )
    elif upper is not None and upper <= 0.0:
        recommendation = (
            "root-more-transition-efficient"
            if budget_matched
            else "root-higher-return-at-unequal-total-work"
        )
    else:
        recommendation = (
            "expand-equal-transition-ablation"
            if budget_matched
            else "run-selection-only-equal-transition-ablation"
        )
    merged = {
        **metadata,
        "shards": len(payloads),
        "states": len(rows),
        "samplingAttempts": sum(
            int(payload["samplingAttempts"]) for payload in payloads
        ),
        "summary": summary,
        "recommendation": recommendation,
        "sourceReports": [str(path) for path in args.inputs],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "recommendation": recommendation}, sort_keys=True))


if __name__ == "__main__":
    main()
