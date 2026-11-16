#!/usr/bin/env python3
"""Add conservative slice-aware leaf weighting to a calibration report."""

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
    parser.add_argument("report", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = json.loads(args.report.read_text())
    rows = payload["rows"]
    bands = {
        "early": (2, 4),
        "mid": (5, 8),
        "late": (9, 25),
    }
    by_band = {
        name: metrics(
            [row for row in rows if lower <= row["turn"] <= upper]
        )
        for name, (lower, upper) in bands.items()
        if any(lower <= row["turn"] <= upper for row in rows)
    }
    by_position = payload.get("byPosition", {})
    slices = [*by_band.values(), *by_position.values()]
    payload["byBand"] = by_band
    payload["sliceRecommendations"] = {
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
    payload["aggregateRecommendedLeafValueWeight"] = payload.get(
        "recommendedLeafValueWeight"
    )
    payload["recommendedLeafValueWeight"] = recommended_leaf_weight(
        payload["overall"], slices
    )
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps({
        "aggregate": payload["aggregateRecommendedLeafValueWeight"],
        "sliceRecommendations": payload["sliceRecommendations"],
        "final": payload["recommendedLeafValueWeight"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
