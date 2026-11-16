#!/usr/bin/env python3
"""Merge independently generated rollout-search datasets and summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MEAN_FIELDS = (
    "meanConfirmedReturnImprovement",
    "meanFirstBattleScoreImprovement",
    "meanHorizonBattleScoreImprovement",
    "meanValidationReturnImprovement",
    "meanValidationFirstBattleScoreImprovement",
    "meanValidationHorizonBattleScoreImprovement",
    "meanOuterAuditReturnImprovement",
    "meanOuterAuditFirstBattleScoreImprovement",
    "meanOuterAuditHorizonBattleScoreImprovement",
)


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


def merge_jsonl(inputs, output, *, add_replay_prefix=False, add_shard=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as destination:
        for shard, path in enumerate(inputs):
            with path.open() as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if add_replay_prefix and "replay_id" in record:
                        record["replay_id"] = (
                            f"shard-{shard:02d}-" + record["replay_id"]
                        )
                    if add_shard:
                        record["shard"] = shard
                    destination.write(
                        json.dumps(record, separators=(",", ":")) + "\n"
                    )
                    count += 1
    return count


def weighted_mean(summaries, split, field):
    weighted = 0.0
    count = 0
    for summary in summaries:
        values = summary["splits"][split]
        value = values.get(field)
        accepted = int(values["accepted"])
        if value is None or accepted == 0:
            continue
        weighted += float(value) * accepted
        count += accepted
    return None if count == 0 else weighted / count


def outer_audit_summary(path):
    rows = []
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("accepted"):
                continue
            value = record.get("outerAuditReturnImprovement")
            if value is None:
                raise SystemExit(
                    "accepted search record is missing untouched outer audit"
                )
            rows.append(record)
    returns = [float(record["outerAuditReturnImprovement"]) for record in rows]
    mean = sum(returns) / len(returns) if returns else None
    if len(returns) < 2:
        standard_error = None
        lower_bound = None
    else:
        variance = sum((value - mean) ** 2 for value in returns) / (
            len(returns) - 1
        )
        standard_error = math.sqrt(variance / len(returns))
        lower_bound = mean - 1.96 * standard_error
    return {
        "acceptedLabels": len(rows),
        "meanReturnImprovement": mean,
        "standardError": standard_error,
        "lowerBound95": lower_bound,
        "meanFirstBattleScoreImprovement": (
            None
            if not rows
            else sum(
                float(record["outerAuditFirstBattleScoreImprovement"])
                for record in rows
            )
            / len(rows)
        ),
        "meanHorizonBattleScoreImprovement": (
            None
            if not rows
            else sum(
                float(record["outerAuditHorizonBattleScoreImprovement"])
                for record in rows
            )
            / len(rows)
        ),
    }


def merge_pruning_audits(summaries):
    fields = (
        "prunedActionsAudited",
        "confirmedPrunedWinners",
        "validatedPrunedWinners",
    )
    merged = {field: {} for field in fields}
    for summary in summaries:
        audit = summary.get("pruningAudit", {})
        for field in fields:
            for action_type, count in audit.get(field, {}).items():
                merged[field][action_type] = (
                    merged[field].get(action_type, 0) + int(count)
                )
    action_types = set(merged["prunedActionsAudited"])
    merged["validatedFalseNegativeRate"] = {
        action_type: (
            merged["validatedPrunedWinners"].get(action_type, 0)
            / max(1, merged["prunedActionsAudited"][action_type])
        )
        for action_type in sorted(action_types)
    }
    return merged


def main():
    args = parse_args()
    groups = (args.train, args.holdout, args.audit, args.summaries)
    shard_count = len(args.summaries)
    if shard_count == 0 or any(len(group) != shard_count for group in groups):
        raise SystemExit("all input groups must contain the same nonzero shard count")

    summaries = [json.loads(path.read_text()) for path in args.summaries]
    versions = {summary["searchVersion"] for summary in summaries}
    revisions = {summary["simulatorRevision"] for summary in summaries}
    if len(versions) != 1 or len(revisions) != 1:
        raise SystemExit(
            f"incompatible shard metadata: versions={versions}, revisions={revisions}"
        )

    train_count = merge_jsonl(
        args.train, args.output, add_replay_prefix=True
    )
    holdout_count = merge_jsonl(
        args.holdout, args.holdout_output, add_replay_prefix=True
    )
    audit_count = merge_jsonl(
        args.audit, args.audit_output, add_shard=True
    )
    audit_summary = outer_audit_summary(args.audit_output)
    pruning_audit = merge_pruning_audits(summaries)

    split_counts = {}
    for split, expected_targets in (
        ("train", train_count),
        ("holdout", holdout_count),
    ):
        split_counts[split] = {
            "attempted": sum(
                int(summary["splits"][split]["attempted"])
                for summary in summaries
            ),
            "accepted": sum(
                int(summary["splits"][split]["accepted"])
                for summary in summaries
            ),
            "weightedTargets": expected_targets,
            **{
                field: weighted_mean(summaries, split, field)
                for field in MEAN_FIELDS
            },
        }

    attempted = sum(int(summary["attempted"]) for summary in summaries)
    accepted = sum(int(summary["accepted"]) for summary in summaries)
    merged = {
        "searchVersion": versions.pop(),
        "simulatorRevision": revisions.pop(),
        "shards": shard_count,
        "attempted": attempted,
        "accepted": accepted,
        "acceptanceRate": accepted / max(1, attempted),
        "rootRecallAudits": sum(
            int(summary["rootRecallAudits"]) for summary in summaries
        ),
        "screeningBestExcludedByPruning": sum(
            int(summary["screeningBestExcludedByPruning"])
            for summary in summaries
        ),
        "splits": split_counts,
        "auditRecords": audit_count,
        "outerAudit": audit_summary,
        "pruningAudit": pruning_audit,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
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
    if (
        audit_summary["lowerBound95"] is None
        or audit_summary["lowerBound95"] <= 0.0
    ):
        raise SystemExit(
            "untouched outer-audit 95% lower bound did not exceed zero; "
            "refusing to distill search labels"
        )


if __name__ == "__main__":
    main()
