#!/usr/bin/env python3
"""Join blinded expert ratings back to hidden search/policy provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    parser.add_argument("ratings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    match = re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        args.review.read_text(),
        flags=re.DOTALL,
    )
    if match is None:
        raise SystemExit("review file does not contain embedded cases")
    cases = json.loads(match.group(1))
    ratings = json.loads(args.ratings.read_text()).get("ratings", {})
    rows = []
    choices = Counter()
    preferences = Counter()
    for case in cases:
        rating = ratings.get(case["id"])
        if not rating or not rating.get("choice"):
            continue
        choice = rating["choice"]
        choices[choice] += 1
        if choice == "a":
            preference = case["options"][0]["origin"]
        elif choice == "b":
            preference = case["options"][1]["origin"]
        else:
            preference = choice
        preferences[preference] += 1
        rows.append(
            {
                "id": case["id"],
                "turn": case["turn"],
                "reasons": case["reasons"],
                "sequenceOrder": case.get("sequenceOrder", False),
                "optionA": case["options"][0],
                "optionB": case["options"][1],
                "blindChoice": choice,
                "revealedPreference": preference,
                "note": rating.get("note", ""),
                "evidence": case["evidence"],
            }
        )
    report = {
        "schemaVersion": 1,
        "cases": len(cases),
        "rated": len(rows),
        "choiceCounts": dict(sorted(choices.items())),
        "revealedPreferenceCounts": dict(sorted(preferences.items())),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "cases", "rated", "choiceCounts", "revealedPreferenceCounts"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
