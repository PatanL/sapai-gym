#!/usr/bin/env python3
"""Merge independently validated candidate-action shards."""

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


def merge_jsonl(inputs, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as destination:
        for path in inputs:
            with path.open() as source:
                for line in source:
                    if line.strip():
                        destination.write(line)
                        count += 1
    return count


def statistics(values):
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
    if not shards or any(len(group) != shards for group in groups):
        raise SystemExit("all input groups must have the same nonzero length")
    summaries = [json.loads(path.read_text()) for path in args.summaries]
    versions = {summary["validatorVersion"] for summary in summaries}
    revisions = {summary["simulatorRevision"] for summary in summaries}
    scenario_versions = {summary["scenarioVersion"] for summary in summaries}
    if len(versions) != 1 or len(revisions) != 1 or len(scenario_versions) != 1:
        raise SystemExit(
            "incompatible candidate shards: "
            f"versions={versions}, revisions={revisions}, "
            f"scenario_versions={scenario_versions}"
        )
    train_examples = merge_jsonl(args.train, args.output)
    holdout_examples = merge_jsonl(args.holdout, args.holdout_output)
    audit_records = merge_jsonl(args.audit, args.audit_output)
    accepted_rows = []
    scenarios = Counter()
    action_changes = Counter()
    with args.audit_output.open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("accepted"):
                continue
            accepted_rows.append(row)
            scenarios[row["scenario"]] += 1
            action_changes[
                f"{row['policyAction'][0]}->{row['candidateAction'][0]}"
            ] += 1
    attempted = sum(int(summary["attempted"]) for summary in summaries)
    merged = {
        "validatorVersion": versions.pop(),
        "scenarioVersion": scenario_versions.pop(),
        "simulatorRevision": revisions.pop(),
        "shards": shards,
        "attempted": attempted,
        "accepted": len(accepted_rows),
        "acceptanceRate": len(accepted_rows) / max(1, attempted),
        "trainExamples": train_examples,
        "holdoutExamples": holdout_examples,
        "auditRecords": audit_records,
        "validation": statistics(
            [row["validation"]["mean"] for row in accepted_rows]
        ),
        "outerAudit": statistics(
            [row["outerAudit"]["mean"] for row in accepted_rows]
        ),
        "scenarioCounts": dict(sorted(scenarios.items())),
        "actionChanges": dict(sorted(action_changes.items())),
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
