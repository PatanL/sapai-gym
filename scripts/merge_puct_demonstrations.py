#!/usr/bin/env python3
"""Merge independently generated PUCT teacher shards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", type=Path, nargs="+", required=True)
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def merge_jsonl(inputs, output, *, prefix_ids=False, add_shard=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as destination:
        for shard, path in enumerate(inputs):
            with path.open() as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if prefix_ids and "replay_id" in row:
                        row["replay_id"] = (
                            f"shard-{shard:02d}-" + row["replay_id"]
                        )
                    if add_shard:
                        row["shard"] = shard
                    destination.write(
                        json.dumps(row, separators=(",", ":")) + "\n"
                    )
                    count += 1
    return count


def mean_ci(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standardError": None,
            "lowerBound95": None,
        }
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return {
            "count": len(values),
            "mean": mean,
            "standardError": None,
            "lowerBound95": None,
        }
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "count": len(values),
        "mean": mean,
        "standardError": standard_error,
        "lowerBound95": mean - 1.96 * standard_error,
    }


def main():
    args = parse_args()
    groups = (args.train, args.holdout, args.audit, args.summaries)
    shards = len(args.summaries)
    if shards == 0 or any(len(group) != shards for group in groups):
        raise SystemExit("all input groups must have the same nonzero length")
    summaries = [json.loads(path.read_text()) for path in args.summaries]
    versions = {summary["puctVersion"] for summary in summaries}
    revisions = {summary["simulatorRevision"] for summary in summaries}
    if len(versions) != 1 or len(revisions) != 1:
        raise SystemExit(
            f"incompatible shards: versions={versions}, revisions={revisions}"
        )
    train_count = merge_jsonl(args.train, args.output, prefix_ids=True)
    holdout_count = merge_jsonl(
        args.holdout, args.holdout_output, prefix_ids=True
    )
    audit_count = merge_jsonl(
        args.audit, args.audit_output, add_shard=True
    )
    candidate_rows = []
    inner_accepted = 0
    deployed_accepted = 0
    with args.audit_output.open() as source:
        for line in source:
            row = json.loads(line)
            accepted = bool(row.get("accepted"))
            inner_accepted += int(accepted)
            independently_confirmed = row.get("independentlyConfirmed")
            if independently_confirmed is None:
                independently_confirmed = bool(
                    accepted
                    and float(row.get("outerAuditReturnLowerBound") or 0.0)
                    > 0.0
                )
            deployed_accepted += int(independently_confirmed)
            if row.get("candidateEligible", row.get("accepted", False)):
                candidate_rows.append(row)
    outer = mean_ci(
        [
            row["outerAuditReturnImprovement"]
            for row in candidate_rows
            if row.get("outerAuditReturnImprovement") is not None
        ]
    )
    validation = mean_ci(
        [row["validationReturnImprovement"] for row in candidate_rows]
    )
    reasons = Counter()
    changes = Counter()
    for summary in summaries:
        reasons.update(summary.get("reasonCounts", {}))
        changes.update(summary.get("actionChanges", {}))
    attempted = sum(int(summary["attempted"]) for summary in summaries)
    elapsed_weighted = sum(
        float(summary["meanWallSeconds"] or 0.0)
        * int(summary["attempted"])
        for summary in summaries
    )
    multi_fidelity = [
        summary.get("multiFidelity")
        for summary in summaries
        if summary.get("multiFidelity", {}).get("enabled")
    ]
    exhaustive_rows = [
        row
        for row in (
            json.loads(line)
            for line in args.audit_output.read_text().splitlines()
            if line.strip()
        )
        if row.get("exhaustiveRoot")
        and (row.get("multiFidelity") or {}).get("fullSearch", True)
    ]
    confirmed_pruned_rows = [
        row
        for row in exhaustive_rows
        if row.get("proposedExcludedByPruning")
        and row.get("accepted")
        and float(row.get("outerAuditReturnLowerBound") or 0.0) > 0.0
    ]
    pruned_winner_actions = Counter(
        (row.get("proposedActionDetail") or {}).get("type", "unknown")
        for row in confirmed_pruned_rows
    )
    removed_actions = Counter()
    validation_planned_sources = Counter()
    validation_policy_sources = Counter()
    audit_planned_sources = Counter()
    audit_policy_sources = Counter()
    for row in exhaustive_rows:
        removed_actions.update(row.get("prunedActionCounts") or {})
    with args.audit_output.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            validation_planned_sources.update(
                row.get("validationPlannedOpponentSourceCounts") or {}
            )
            validation_policy_sources.update(
                row.get("validationPolicyOpponentSourceCounts") or {}
            )
            audit_planned_sources.update(
                row.get("outerAuditPlannedOpponentSourceCounts") or {}
            )
            audit_policy_sources.update(
                row.get("outerAuditPolicyOpponentSourceCounts") or {}
            )
    merged = {
        "puctVersion": versions.pop(),
        "simulatorRevision": revisions.pop(),
        "shards": shards,
        "attempted": attempted,
        "accepted": inner_accepted,
        "candidateEligible": len(candidate_rows),
        "candidateRate": len(candidate_rows) / max(1, attempted),
        "deployedAccepted": deployed_accepted,
        "acceptanceRate": inner_accepted / max(1, attempted),
        "independentConfirmationRate": (
            deployed_accepted / max(1, attempted)
        ),
        "trainExamples": train_count,
        "holdoutExamples": holdout_count,
        "auditRecords": audit_count,
        "rootTargetNodes": sum(
            int(summary["rootTargetNodes"]) for summary in summaries
        ),
        "internalTargetNodes": sum(
            int(summary["internalTargetNodes"]) for summary in summaries
        ),
        "validation": validation,
        "outerAudit": outer,
        "meanWallSeconds": elapsed_weighted / max(1, attempted),
        "reasonCounts": dict(sorted(reasons.items())),
        "actionChanges": dict(sorted(changes.items())),
        "multiFidelity": (
            {
                "enabled": True,
                "screeningAttempts": sum(
                    row["screeningAttempts"] for row in multi_fidelity
                ),
                "screeningPasses": sum(
                    row["screeningPasses"] for row in multi_fidelity
                ),
                "screeningBypasses": sum(
                    row["screeningBypasses"] for row in multi_fidelity
                ),
                "fullSearches": sum(
                    row["fullSearches"] for row in multi_fidelity
                ),
                "screeningPassCandidates": sum(
                    row.get("screeningPassCandidates", 0)
                    for row in multi_fidelity
                ),
                "bypassCandidates": sum(
                    row.get("bypassCandidates", 0)
                    for row in multi_fidelity
                ),
            }
            if multi_fidelity
            else {"enabled": False}
        ),
        "rootPruningAudit": {
            "exhaustiveFullSearches": len(exhaustive_rows),
            "proposedExcludedByPruning": sum(
                bool(row.get("proposedExcludedByPruning"))
                for row in exhaustive_rows
            ),
            "treeProposedExcludedByPruning": sum(
                bool(row.get("treeProposedExcludedByPruning"))
                for row in exhaustive_rows
            ),
            "independentlyConfirmedPrunedWinners": len(
                confirmed_pruned_rows
            ),
            "confirmedFalseNegativeRate": (
                len(confirmed_pruned_rows) / max(1, len(exhaustive_rows))
            ),
            "confirmedWinnersByAction": dict(
                sorted(pruned_winner_actions.items())
            ),
            "removedActionsByType": dict(sorted(removed_actions.items())),
        },
        "opponentSourceTelemetry": {
            "validationPlanned": dict(
                sorted(validation_planned_sources.items())
            ),
            "validationPolicy": dict(
                sorted(validation_policy_sources.items())
            ),
            "outerAuditPlanned": dict(
                sorted(audit_planned_sources.items())
            ),
            "outerAuditPolicy": dict(
                sorted(audit_policy_sources.items())
            ),
        },
        "trainSha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "holdoutSha256": hashlib.sha256(
            args.holdout_output.read_bytes()
        ).hexdigest(),
        "sourceSummaries": [str(path) for path in args.summaries],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(merged, sort_keys=True))


if __name__ == "__main__":
    main()
