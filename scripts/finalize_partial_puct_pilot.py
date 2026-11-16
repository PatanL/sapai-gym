#!/usr/bin/env python3
"""Finalize streamed PUCT pilot parts after a deliberate early stop."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.puct_search import PUCT_VERSION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--puct-version")
    return parser.parse_args()


def merge_parts(paths, output, *, add_shard=False, prefix_ids=False):
    count = 0
    with output.open("w") as destination:
        for shard, path in enumerate(paths):
            if not path.exists():
                continue
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
    standard_error = (
        None
        if len(values) < 2
        else float(values.std(ddof=1) / math.sqrt(len(values)))
    )
    return {
        "count": len(values),
        "mean": mean,
        "standardError": standard_error,
        "lowerBound95": (
            None if standard_error is None else mean - 1.96 * standard_error
        ),
    }


def main():
    args = parse_args()
    prefix = args.prefix
    train_parts = [
        prefix.with_name(f"{prefix.name}-part-{shard:02d}.jsonl")
        for shard in range(args.shards)
    ]
    holdout_parts = [
        prefix.with_name(
            f"{prefix.name}-holdout-part-{shard:02d}.jsonl"
        )
        for shard in range(args.shards)
    ]
    audit_parts = [
        prefix.with_name(f"{prefix.name}-audit-part-{shard:02d}.jsonl")
        for shard in range(args.shards)
    ]
    train_output = prefix.with_suffix(".jsonl")
    holdout_output = prefix.with_name(prefix.name + "-holdout.jsonl")
    audit_output = prefix.with_suffix(".audit.jsonl")
    summary_output = prefix.with_suffix(".summary.json")
    train_count = merge_parts(train_parts, train_output, prefix_ids=True)
    holdout_count = merge_parts(
        holdout_parts, holdout_output, prefix_ids=True
    )
    audit_count = merge_parts(audit_parts, audit_output, add_shard=True)
    rows = [json.loads(line) for line in audit_output.read_text().splitlines()]
    accepted = [row for row in rows if row.get("accepted")]
    candidates = [row for row in rows if row.get("candidateEligible")]
    reasons = Counter(
        reason
        for row in rows
        for reason in (row.get("reasons") or ["general"])
    )
    changes = Counter(
        f"{row['policyActionDetail']['type']}->{row['proposedActionDetail']['type']}"
        for row in accepted
    )
    summary = {
        "puctVersion": args.puct_version or PUCT_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "earlyStopped": True,
        "attempted": audit_count,
        "accepted": len(accepted),
        "candidateEligible": len(candidates),
        "candidateRate": len(candidates) / max(1, audit_count),
        "acceptanceRate": len(accepted) / max(1, audit_count),
        "trainExamples": train_count,
        "holdoutExamples": holdout_count,
        "rootTargetNodes": 0,
        "internalTargetNodes": 0,
        "validation": mean_ci(
            [row["validationReturnImprovement"] for row in accepted]
        ),
        "outerAudit": mean_ci(
            [row["outerAuditReturnImprovement"] for row in accepted]
        ),
        "meanWallSeconds": (
            None
            if not rows
            else float(np.mean([row["wallSeconds"] for row in rows]))
        ),
        "reasonCounts": dict(sorted(reasons.items())),
        "actionChanges": dict(sorted(changes.items())),
        "multiFidelity": {
            "enabled": True,
            "screeningAttempts": audit_count,
            "screeningPasses": sum(
                row.get("simulatorTransitions", 0) > 1000 for row in rows
            ),
            "screeningBypasses": 0,
            "fullSearches": sum(
                row.get("simulatorTransitions", 0) > 1000 for row in rows
            ),
            "screeningPassCandidates": len(candidates),
            "bypassCandidates": 0,
            "partialInference": True,
        },
        "trainSha256": hashlib.sha256(train_output.read_bytes()).hexdigest(),
        "holdoutSha256": hashlib.sha256(
            holdout_output.read_bytes()
        ).hexdigest(),
    }
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
