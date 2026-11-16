#!/usr/bin/env python3
"""Summarize PUCT coverage, escalation quality, and accepted decisions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


TURN_BANDS = (
    ("early", 2, 4),
    ("mid", 5, 8),
    ("late", 9, 10),
    ("endgame", 11, 99),
)


def turn_band(turn):
    for name, minimum, maximum in TURN_BANDS:
        if minimum <= int(turn) <= maximum:
            return name
    return "outside"


def escalation_mode(row):
    multi = row.get("multiFidelity") or {}
    if not multi.get("enabled"):
        return "unscreened"
    if multi.get("screenBypass"):
        return "bypass"
    if multi.get("screenPass"):
        return "screen-pass"
    return "screen-reject"


def summarize(
    rows,
    *,
    minimum_states_per_band=0,
    minimum_full_searches_per_band=0,
    minimum_strict_roots=0,
    minimum_strict_bands=0,
    minimum_strict_train_roots=0,
    minimum_strict_holdout_roots=0,
    minimum_exhaustive_root_audits=0,
    minimum_opponent_sources=0,
):
    turns = Counter()
    bands = Counter()
    full_search_bands = Counter()
    reasons = defaultdict(Counter)
    modes = defaultdict(Counter)
    wall_by_mode = defaultdict(list)
    accepted = []
    strict_accepted = []
    exhaustive_rows = []
    confirmed_pruned_rows = []
    removed_actions = Counter()
    validation_planned_sources = Counter()
    validation_policy_sources = Counter()
    audit_planned_sources = Counter()
    audit_policy_sources = Counter()
    for row in rows:
        turn = int(row["turn"])
        turns[turn] += 1
        bands[turn_band(turn)] += 1
        mode = escalation_mode(row)
        full = bool((row.get("multiFidelity") or {}).get("fullSearch"))
        if full:
            full_search_bands[turn_band(turn)] += 1
        candidate = bool(row.get("candidateEligible"))
        winner = bool(row.get("accepted"))
        if row.get("exhaustiveRoot") and full:
            exhaustive_rows.append(row)
            removed_actions.update(row.get("prunedActionCounts") or {})
            if (
                row.get("proposedExcludedByPruning")
                and winner
                and float(row.get("outerAuditReturnLowerBound") or 0.0) > 0
            ):
                confirmed_pruned_rows.append(row)
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
        modes[mode]["states"] += 1
        modes[mode]["fullSearches"] += int(full)
        modes[mode]["candidates"] += int(candidate)
        modes[mode]["accepted"] += int(winner)
        if row.get("wallSeconds") is not None:
            wall_by_mode[mode].append(float(row["wallSeconds"]))
        for reason in set(row.get("reasons") or ["general"]):
            reasons[reason]["states"] += 1
            reasons[reason]["fullSearches"] += int(full)
            reasons[reason]["candidates"] += int(candidate)
            reasons[reason]["accepted"] += int(winner)
        if winner:
            accepted_row = {
                    "game": row.get("game"),
                    "side": row.get("side"),
                    "turn": turn,
                    "band": turn_band(turn),
                    "split": row.get("split", "unknown"),
                    "reasons": row.get("reasons", []),
                    "mode": mode,
                    "policyAction": (
                        row.get("policyActionDetail") or {}
                    ).get("description"),
                    "puctAction": (
                        row.get("proposedActionDetail") or {}
                    ).get("description"),
                    "validationMean": row.get(
                        "validationReturnImprovement"
                    ),
                    "validationLower95": row.get(
                        "validationReturnLowerBound"
                    ),
                    "outerMean": row.get(
                        "outerAuditReturnImprovement"
                    ),
                    "outerLower95": row.get(
                        "outerAuditReturnLowerBound"
                    ),
                }
            accepted.append(accepted_row)
            if float(row.get("outerAuditReturnLowerBound") or 0.0) > 0:
                strict_accepted.append(accepted_row)
    mode_payload = {}
    for mode, counts in sorted(modes.items()):
        payload = dict(counts)
        payload["candidateRate"] = (
            counts["candidates"] / counts["states"]
            if counts["states"]
            else 0.0
        )
        payload["acceptanceRate"] = (
            counts["accepted"] / counts["states"]
            if counts["states"]
            else 0.0
        )
        payload["meanWallSeconds"] = (
            statistics.mean(wall_by_mode[mode])
            if wall_by_mode[mode]
            else None
        )
        mode_payload[mode] = payload
    strict_band_counts = Counter(
        row["band"] for row in strict_accepted
    )
    strict_split_counts = Counter(
        row["split"] for row in strict_accepted
    )
    coverage_checks = {
        "minimumStatesPerBand": all(
            bands.get(name, 0) >= minimum_states_per_band
            for name, _, _ in TURN_BANDS
        ),
        "minimumFullSearchesPerBand": all(
            full_search_bands.get(name, 0)
            >= minimum_full_searches_per_band
            for name, _, _ in TURN_BANDS
        ),
        "minimumStrictRoots": (
            len(strict_accepted) >= minimum_strict_roots
        ),
        "minimumStrictBands": (
            sum(bool(strict_band_counts.get(name, 0)) for name, _, _ in TURN_BANDS)
            >= minimum_strict_bands
        ),
        "minimumStrictTrainRoots": (
            strict_split_counts.get("train", 0)
            >= minimum_strict_train_roots
        ),
        "minimumStrictHoldoutRoots": (
            strict_split_counts.get("holdout", 0)
            >= minimum_strict_holdout_roots
        ),
        "minimumExhaustiveRootAudits": (
            len(exhaustive_rows) >= minimum_exhaustive_root_audits
        ),
        "minimumOpponentSources": (
            len(validation_planned_sources) >= minimum_opponent_sources
        ),
    }
    return {
        "states": len(rows),
        "turnCounts": {
            str(key): value for key, value in sorted(turns.items())
        },
        "turnBandCounts": {
            name: bands.get(name, 0)
            for name, _, _ in TURN_BANDS
        }
        | ({"outside": bands["outside"]} if bands["outside"] else {}),
        "fullSearchesByBand": {
            name: full_search_bands.get(name, 0)
            for name, _, _ in TURN_BANDS
        },
        "escalationModes": mode_payload,
        "reasonTelemetry": {
            reason: dict(counts)
            for reason, counts in sorted(reasons.items())
        },
        "accepted": accepted,
        "strictAccepted": strict_accepted,
        "strictAcceptedCount": len(strict_accepted),
        "strictAcceptedByBand": {
            name: strict_band_counts.get(name, 0)
            for name, _, _ in TURN_BANDS
        },
        "strictAcceptedBySplit": dict(sorted(strict_split_counts.items())),
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
                sorted(
                    Counter(
                        (row.get("proposedActionDetail") or {}).get(
                            "type", "unknown"
                        )
                        for row in confirmed_pruned_rows
                    ).items()
                )
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
        "readiness": {
            "requirements": {
                "minimumStatesPerBand": minimum_states_per_band,
                "minimumFullSearchesPerBand": (
                    minimum_full_searches_per_band
                ),
                "minimumStrictRoots": minimum_strict_roots,
                "minimumStrictBands": minimum_strict_bands,
                "minimumStrictTrainRoots": minimum_strict_train_roots,
                "minimumStrictHoldoutRoots": minimum_strict_holdout_roots,
                "minimumExhaustiveRootAudits": (
                    minimum_exhaustive_root_audits
                ),
                "minimumOpponentSources": minimum_opponent_sources,
            },
            "checks": coverage_checks,
            "readyForDistillation": all(coverage_checks.values()),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-states-per-band", type=int, default=0)
    parser.add_argument(
        "--minimum-full-searches-per-band", type=int, default=0
    )
    parser.add_argument("--minimum-strict-roots", type=int, default=0)
    parser.add_argument("--minimum-strict-bands", type=int, default=0)
    parser.add_argument("--minimum-strict-train-roots", type=int, default=0)
    parser.add_argument(
        "--minimum-strict-holdout-roots", type=int, default=0
    )
    parser.add_argument(
        "--minimum-exhaustive-root-audits", type=int, default=0
    )
    parser.add_argument("--minimum-opponent-sources", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.audit.read_text().splitlines()
        if line.strip()
    ]
    if any(
        value < 0
        for value in (
            args.minimum_states_per_band,
            args.minimum_full_searches_per_band,
            args.minimum_strict_roots,
            args.minimum_strict_bands,
            args.minimum_strict_train_roots,
            args.minimum_strict_holdout_roots,
            args.minimum_exhaustive_root_audits,
            args.minimum_opponent_sources,
        )
    ):
        raise SystemExit("readiness thresholds must be non-negative")
    report = summarize(
        rows,
        minimum_states_per_band=args.minimum_states_per_band,
        minimum_full_searches_per_band=(
            args.minimum_full_searches_per_band
        ),
        minimum_strict_roots=args.minimum_strict_roots,
        minimum_strict_bands=args.minimum_strict_bands,
        minimum_strict_train_roots=args.minimum_strict_train_roots,
        minimum_strict_holdout_roots=args.minimum_strict_holdout_roots,
        minimum_exhaustive_root_audits=(
            args.minimum_exhaustive_root_audits
        ),
        minimum_opponent_sources=args.minimum_opponent_sources,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
