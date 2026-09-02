#!/usr/bin/env python3
"""Aggregate unattended v16 milestone evaluations into a compact trend report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", type=Path)
    return parser.parse_args()


def opponent_update(report: dict, path: Path) -> int:
    """Read paired metadata or recover the update from the checkpoint path."""

    value = report.get("opponentCheckpointUpdate")
    if value is not None:
        return int(value)
    checkpoint = str(report.get("opponentCheckpoint", ""))
    match = re.search(r"update-(\d+)\.pt$", checkpoint)
    if match:
        return int(match.group(1))
    match = re.search(r"-vs-(\d+)\.json$", path.name)
    if match:
        return int(match.group(1))
    raise KeyError(f"Cannot determine opponent update for {path}")


def main():
    args = parse_args()
    rows = []
    for path in sorted(args.evaluation_dir.glob("paired-*.json")):
        report = json.loads(path.read_text())
        rows.append(
            {
                "candidateUpdate": int(report["checkpointUpdate"]),
                "opponentUpdate": opponent_update(report, path),
                "panel": (
                    "fresh" if path.name.startswith("paired-fresh-") else "fixed"
                ),
                "games": int(report["games"]),
                "wins": int(report["outcomes"].get("win", 0)),
                "draws": int(report["outcomes"].get("draw", 0)),
                "losses": int(report["outcomes"].get("loss", 0)),
                "score": float(report["pairedMeanScore"]),
                "confidence95": [
                    float(value) for value in report["pairedScore95CI"]
                ],
                "seed": int(report["seed"]),
                "runtimeSimulatorRevision": report[
                    "runtimeSimulatorRevision"
                ],
                "path": str(path),
            }
        )
    rows.sort(
        key=lambda row: (
            row["candidateUpdate"],
            0 if row["panel"] == "fixed" else 1,
        )
    )
    behavior_rows = []
    for path in sorted(args.evaluation_dir.glob("behavior-v2-*.json")):
        report = json.loads(path.read_text())
        behavior_rows.append(
            {
                "candidateUpdate": int(report["checkpointUpdate"]),
                "opponentUpdate": opponent_update(report, path),
                "games": int(report["games"]),
                "winRate": float(report["matchWinRate"]),
                "meanFinalTeamStats": float(report["meanFinalTeamStats"]),
                "meanSellsPerMatch": float(report["meanSellsPerMatch"]),
                "meanPermanentlyStattedPets": float(
                    report["meanPermanentlyStattedPets"]
                ),
                "meanTierOneTwoPetsAfterTurnEight": float(
                    report["meanTierOneTwoPetsAfterTurnEight"]
                ),
                "fractionEndTurnsWithFullBoard": float(
                    report["fractionEndTurnsWithFullBoard"]
                ),
                "meanUnspentGold": float(report["meanUnspentGold"]),
                "discoverGroupsSeen": int(report["discoverGroupsSeen"]),
                "discoverPurchases": int(report["discoverPurchases"]),
                "discoverFreezeActions": int(report["discoverFreezeActions"]),
                "synergyMatchCounts": report.get("synergyMatchCounts", {}),
                "synergyFinalTeamCounts": report.get(
                    "synergyFinalTeamCounts", {}
                ),
                "synergyMatchOutcomes": report.get(
                    "synergyMatchOutcomes", {}
                ),
                "synergyLabels": report.get("synergyLabels", {}),
                "levelTwoTargetMetrics": report.get(
                    "levelTwoTargetMetrics", {}
                ),
                "earlyCakeMetrics": report.get("earlyCakeMetrics", {}),
                "path": str(path),
            }
        )
    payload = {
        "schemaVersion": 1,
        "comparisons": rows,
        "behavior": behavior_rows,
    }
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    (args.evaluation_dir / "trend.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    lines = [
        "# V16 milestone evaluation trend",
        "",
        "All comparisons use deterministic, paired-seat CPU evaluation under "
        "the current v16 runtime simulator.",
        "",
        "| Candidate | Opponent | Panel | Record | Score | 95% CI | Games |",
        "|---:|---:|:---|---:|---:|---:|---:|",
    ]
    for row in rows:
        low, high = row["confidence95"]
        lines.append(
            f"| {row['candidateUpdate']} | {row['opponentUpdate']} | {row['panel']} | "
            f"{row['wins']}-{row['draws']}-{row['losses']} | "
            f"{100 * row['score']:.1f}% | "
            f"{100 * low:.1f}–{100 * high:.1f}% | {row['games']} |"
        )
    lines.extend(
        [
            "",
            "## Behavior by checkpoint opponent",
            "",
            "| Candidate | Opponent | Win rate | Final stats | Sells/match | Statted pets | "
            "Late T1/2 | Full board | Unspent gold | Discover buy/freeze |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in behavior_rows:
        lines.append(
            f"| {row['candidateUpdate']} | {row['opponentUpdate']} | "
            f"{100 * row['winRate']:.1f}% | "
            f"{row['meanFinalTeamStats']:.1f} | {row['meanSellsPerMatch']:.2f} | "
            f"{row['meanPermanentlyStattedPets']:.2f} | "
            f"{row['meanTierOneTwoPetsAfterTurnEight']:.2f} | "
            f"{100 * row['fractionEndTurnsWithFullBoard']:.1f}% | "
            f"{row['meanUnspentGold']:.2f} | "
            f"{row['discoverPurchases']}/{row['discoverFreezeActions']} |"
        )
    lines.extend(
        [
            "",
            "## Learned synergy prevalence",
            "",
            "Counts are matches (out of 100) in which the line appeared at "
            "least once; final is the number still assembled on the final board.",
            "",
            "| Candidate | Synergy | Seen | Final | Conditional W-L-D |",
            "|---:|:---|---:|---:|:---|",
        ]
    )
    for row in behavior_rows:
        labels = row["synergyLabels"]
        for synergy, count in row["synergyMatchCounts"].items():
            if not count:
                continue
            outcomes = row["synergyMatchOutcomes"].get(synergy, {})
            record = (
                f"{outcomes.get('win', 0)}-"
                f"{outcomes.get('loss', 0)}-"
                f"{outcomes.get('draw', 0)}"
            )
            lines.append(
                f"| {row['candidateUpdate']} | {labels.get(synergy, synergy)} | "
                f"{count} | {row['synergyFinalTeamCounts'].get(synergy, 0)} | "
                f"{record} |"
            )
    lines.extend(
        [
            "",
            "## Targeted scaling milestones",
            "",
            "| Candidate | Level-2 Dragon matches/final | Level-2 Monkey "
            "matches/final | Early Cake bought/retained/sold 9+ | Retention | "
            "Late sale gold/spent |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in behavior_rows:
        level_two = row["levelTwoTargetMetrics"]
        dragon = level_two.get("pet-dragon", {})
        monkey = level_two.get("pet-monkey", {})
        cake = row["earlyCakeMetrics"]
        retention = cake.get("retentionRate")
        retention_text = "n/a" if retention is None else f"{100 * retention:.1f}%"
        lines.append(
            f"| {row['candidateUpdate']} | {dragon.get('matchCount', 0)}/"
            f"{dragon.get('finalTeamCount', 0)} | {monkey.get('matchCount', 0)}/"
            f"{monkey.get('finalTeamCount', 0)} | {cake.get('purchases', 0)}/"
            f"{cake.get('retainedPurchases', 0)}/"
            f"{cake.get('realizedPurchases', 0)} | {retention_text} | "
            f"{cake.get('lateSaleGoldReceived', 0)}/"
            f"{cake.get('cashoutGoldSpentSameTurn', 0)} |"
        )
    (args.evaluation_dir / "TREND.md").write_text("\n".join(lines) + "\n")
    print(
        f"comparisons={len(rows)} behavior={len(behavior_rows)} "
        f"output={args.evaluation_dir / 'trend.json'}"
    )


if __name__ == "__main__":
    main()
