#!/usr/bin/env python3
"""Keep only PUCT targets backed by strict confirmation and fresh audit gain."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--root-only",
        action="store_true",
        help=(
            "Keep only the independently confirmed root target. Internal tree "
            "targets inherit the root audit rather than receiving their own "
            "counterfactual confirmation."
        ),
    )
    parser.add_argument(
        "--minimum-outer-lower-bound",
        type=float,
        help=(
            "Optionally require each root's fresh outer-audit confidence "
            "lower bound to exceed this value."
        ),
    )
    return parser.parse_args()


def keep_row(
    row,
    *,
    root_only=False,
    minimum_outer_lower_bound=None,
):
    """Return whether a generated PUCT action target clears the strict gate."""

    if 'recordSchema' in row:
        raise ValueError('Primitive PUCT filtering is not joint-aware; use the validated JointSearchDataset/corpus boundary')
    teacher = row.get("teacher", {})
    if teacher.get("name") != "stochastic-puct":
        return False
    if not teacher.get("rootAccepted"):
        return False
    outer = teacher.get("outerAuditReturnImprovement")
    if outer is None or float(outer) <= 0:
        return False
    if root_only and not teacher.get("targetIsRoot"):
        return False
    if minimum_outer_lower_bound is not None:
        outer_lower = teacher.get("outerAuditReturnLowerBound")
        if (
            outer_lower is None
            or float(outer_lower) <= minimum_outer_lower_bound
        ):
            return False
    return True


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    versions = Counter()
    reasons = Counter()
    root_nodes = set()
    internal_nodes = set()
    with args.output.open("w") as destination:
        for source_index, path in enumerate(args.input):
            with path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    total += 1
                    row = json.loads(line)
                    teacher = row.get("teacher", {})
                    if not keep_row(
                        row,
                        root_only=args.root_only,
                        minimum_outer_lower_bound=(
                            args.minimum_outer_lower_bound
                        ),
                    ):
                        continue
                    row["replay_id"] = (
                        f"strict-source-{source_index:02d}-" + row["replay_id"]
                    )
                    destination.write(
                        json.dumps(row, separators=(",", ":")) + "\n"
                    )
                    kept += 1
                    versions[teacher.get("version", "unknown")] += 1
                    reasons.update(teacher.get("opportunityReasons", []))
                    node = (
                        source_index,
                        row["replay_id"].rsplit("-action-", 1)[0],
                    )
                    if teacher.get("targetIsRoot"):
                        root_nodes.add(node)
                    else:
                        internal_nodes.add(node)
    if not kept:
        raise SystemExit("no strict positive-audit PUCT targets found")
    summary = {
        "inputs": [str(path) for path in args.input],
        "inputExamples": total,
        "keptExamples": kept,
        "keptFraction": kept / total,
        "rootTargetNodes": len(root_nodes),
        "internalTargetNodes": len(internal_nodes),
        "versionCounts": dict(sorted(versions.items())),
        "reasonCounts": dict(sorted(reasons.items())),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "criteria": {
            "rootAccepted": True,
            "outerAuditReturnImprovement": ">0",
            "targetIsRoot": True if args.root_only else "root-or-internal",
            "outerAuditReturnLowerBound": (
                "not-required"
                if args.minimum_outer_lower_bound is None
                else f">{args.minimum_outer_lower_bound}"
            ),
        },
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
