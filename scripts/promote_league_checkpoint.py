#!/usr/bin/env python3
"""Promote a decisively stronger checkpoint to permanent league residency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sapai_gym1.rl import LeagueManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.65)
    parser.add_argument("--minimum-pairs", type=int, default=100)
    parser.add_argument("--minimum-ci-low", type=float, default=0.50)
    parser.add_argument(
        "--allow-opponent-training-revision-mismatch",
        action="store_true",
        help=(
            "Allow a frozen opponent trained under an older simulator revision. "
            "The evaluation runtime must still exactly match the candidate."
        ),
    )
    parser.add_argument("--member-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_reports(args):
    reports = [json.loads(path.read_text()) for path in args.reports]
    checkpoints = {report.get("checkpoint") for report in reports}
    revisions = {
        report.get("checkpointSimulatorRevision") for report in reports
    }
    if len(checkpoints) != 1 or None in checkpoints:
        raise SystemExit("Every report must evaluate the same candidate checkpoint")
    if len(revisions) != 1 or None in revisions:
        raise SystemExit("Every report must use the same candidate revision")
    failures = []
    for path, report in zip(args.reports, reports):
        score = float(report["pairedMeanScore"])
        ci_low = float(report["pairedScore95CI"][0])
        pairs = int(report["pairs"])
        if score < args.minimum_score:
            failures.append(
                f"{path}: score {score:.3f} < {args.minimum_score:.3f}"
            )
        if ci_low <= args.minimum_ci_low:
            failures.append(
                f"{path}: CI low {ci_low:.3f} <= {args.minimum_ci_low:.3f}"
            )
        if pairs < args.minimum_pairs:
            failures.append(
                f"{path}: pairs {pairs} < {args.minimum_pairs}"
            )
        if (
            report.get("runtimeSimulatorRevision")
            != report.get("checkpointSimulatorRevision")
        ):
            failures.append(f"{path}: candidate/runtime simulator mismatch")
        if (
            report.get("opponentCheckpointSimulatorRevision")
            != report.get("checkpointSimulatorRevision")
            and not args.allow_opponent_training_revision_mismatch
        ):
            failures.append(
                f"{path}: opponent training revision mismatch; explicitly "
                "allow only when both policies were evaluated in the "
                "candidate's runtime"
            )
    if failures:
        raise SystemExit("Promotion gate failed:\n" + "\n".join(failures))
    return reports, checkpoints.pop(), revisions.pop()


def main():
    args = parse_args()
    reports, checkpoint, revision = validate_reports(args)
    checkpoint_path = Path(checkpoint)
    update = reports[0].get("checkpointUpdate")
    transitions = max(
        int(report.get("checkpointTransitions") or 0) for report in reports
    )
    member_id = args.member_id or f"champion-{update or checkpoint_path.stem}"
    payload = {
        "memberId": member_id,
        "checkpoint": checkpoint,
        "simulatorRevision": revision,
        "reports": [str(path) for path in args.reports],
        "minimumScore": args.minimum_score,
        "minimumPairs": args.minimum_pairs,
        "minimumCiLow": args.minimum_ci_low,
        "allowOpponentTrainingRevisionMismatch": (
            args.allow_opponent_training_revision_mismatch
        ),
        "opponentTrainingRevisions": sorted(
            {
                report.get("opponentCheckpointSimulatorRevision")
                for report in reports
            }
        ),
        "dryRun": args.dry_run,
    }
    if not args.dry_run:
        league = LeagueManager.load(args.league)
        league.promote(member_id, checkpoint, created_step=transitions)
        league.save(args.league)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
