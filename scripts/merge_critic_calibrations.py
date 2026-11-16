#!/usr/bin/env python3
"""Merge independent critic-calibration shards into one gated report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_critic_calibration import (
    _recommended_leaf_weight,
    metrics,
    recommended_leaf_weight,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    parts = [json.loads(path.read_text()) for path in args.parts]
    for field in (
        "checkpoint",
        "simulatorRevision",
        "rolloutsPerState",
        "horizonBattles",
    ):
        values = {part[field] for part in parts}
        if len(values) != 1:
            raise SystemExit(f"Calibration shards disagree on {field}: {values}")
    rows = []
    for part_index, part in enumerate(parts):
        for row in part["rows"]:
            rows.append({**row, "shard": part_index})
    overall = metrics(rows)
    by_turn = {
        str(turn): metrics([row for row in rows if row["turn"] == turn])
        for turn in sorted({row["turn"] for row in rows})
    }
    bands = {"early": (2, 4), "mid": (5, 8), "late": (9, 25)}
    by_band = {
        name: metrics(
            [row for row in rows if lower <= row["turn"] <= upper]
        )
        for name, (lower, upper) in bands.items()
        if any(lower <= row["turn"] <= upper for row in rows)
    }
    by_position = {
        position: metrics(
            [row for row in rows if row["position"] == position]
        )
        for position in ("ahead", "even", "behind")
        if any(row["position"] == position for row in rows)
    }
    slices = [*by_band.values(), *by_position.values()]
    slice_recommendations = {
        **{
            f"band:{name}": _recommended_leaf_weight(value)
            for name, value in by_band.items()
            if value["states"] >= 20
        },
        **{
            f"position:{name}": _recommended_leaf_weight(value)
            for name, value in by_position.items()
            if value["states"] >= 20
        },
    }
    payload = {
        "checkpoint": parts[0]["checkpoint"],
        "simulatorRevision": parts[0]["simulatorRevision"],
        "states": len(rows),
        "rolloutsPerState": parts[0]["rolloutsPerState"],
        "horizonBattles": parts[0]["horizonBattles"],
        "shards": [str(path) for path in args.parts],
        "overall": overall,
        "byTurn": by_turn,
        "byBand": by_band,
        "byPosition": by_position,
        "sliceRecommendations": slice_recommendations,
        "recommendedLeafValueWeight": recommended_leaf_weight(
            overall, slices
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "states": len(rows),
        "overall": overall,
        "byBand": by_band,
        "byPosition": by_position,
        "recommendedLeafValueWeight": payload["recommendedLeafValueWeight"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
