#!/usr/bin/env python3
"""Cross-validate a conservative additive critic bias calibrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_critic_calibration import metrics


FEATURE_NAMES = ("intercept", "mid", "late", "even", "behind")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-mae-improvement", type=float, default=0.02)
    return parser.parse_args()


def band(turn):
    return "early" if turn <= 4 else "mid" if turn <= 8 else "late"


def design_row(row):
    row_band = band(int(row["turn"]))
    position = row["position"]
    return np.asarray(
        [
            1.0,
            float(row_band == "mid"),
            float(row_band == "late"),
            float(position == "even"),
            float(position == "behind"),
        ],
        dtype=np.float64,
    )


def fit_bias(rows, ridge=1.0):
    if not rows:
        raise ValueError("Cannot fit critic calibration without rows")
    matrix = np.stack([design_row(row) for row in rows])
    errors = np.asarray(
        [row["predicted"] - row["monteCarlo"] for row in rows],
        dtype=np.float64,
    )
    regularizer = np.eye(matrix.shape[1], dtype=np.float64) * float(ridge)
    # Leave the shared intercept effectively unregularized while shrinking the
    # sparse slice adjustments toward the global correction.
    regularizer[0, 0] = 1e-9
    return np.linalg.solve(
        matrix.T @ matrix + regularizer,
        matrix.T @ errors,
    )


def calibrated_prediction(row, coefficients):
    return float(row["predicted"] - design_row(row) @ coefficients)


def rows_with_predictions(rows, predictions):
    return [
        {"predicted": float(value), "monteCarlo": row["monteCarlo"]}
        for row, value in zip(rows, predictions)
    ]


def grouped_metrics(rows, predictions):
    result = {"overall": metrics(rows_with_predictions(rows, predictions))}
    result["byBand"] = {
        name: metrics(
            rows_with_predictions(
                [row for row in rows if band(int(row["turn"])) == name],
                [
                    value
                    for row, value in zip(rows, predictions)
                    if band(int(row["turn"])) == name
                ],
            )
        )
        for name in ("early", "mid", "late")
        if any(band(int(row["turn"])) == name for row in rows)
    }
    result["byPosition"] = {
        name: metrics(
            rows_with_predictions(
                [row for row in rows if row["position"] == name],
                [
                    value
                    for row, value in zip(rows, predictions)
                    if row["position"] == name
                ],
            )
        )
        for name in ("ahead", "even", "behind")
        if any(row["position"] == name for row in rows)
    }
    return result


def cross_validated_predictions(rows, ridge=1.0):
    predictions = [None] * len(rows)
    shards = sorted({int(row["shard"]) for row in rows})
    if len(shards) < 2:
        raise ValueError("At least two independent shards are required")
    for shard in shards:
        train = [row for row in rows if int(row["shard"]) != shard]
        coefficients = fit_bias(train, ridge=ridge)
        for index, row in enumerate(rows):
            if int(row["shard"]) == shard:
                predictions[index] = calibrated_prediction(row, coefficients)
    if any(value is None for value in predictions):
        raise AssertionError("Cross-validation left rows without predictions")
    return list(map(float, predictions))


def max_abs_slice_bias(report):
    slices = [*report["byBand"].values(), *report["byPosition"].values()]
    return max(abs(float(value["bias"])) for value in slices)


def analyze(payload, ridge=1.0, minimum_mae_improvement=0.02):
    rows = list(payload["rows"])
    raw_predictions = [float(row["predicted"]) for row in rows]
    calibrated = cross_validated_predictions(rows, ridge=ridge)
    raw = grouped_metrics(rows, raw_predictions)
    cross_validated = grouped_metrics(rows, calibrated)
    mae_improvement = (
        raw["overall"]["mae"] - cross_validated["overall"]["mae"]
    )
    raw_max_bias = max_abs_slice_bias(raw)
    calibrated_max_bias = max_abs_slice_bias(cross_validated)
    use_calibrator = bool(
        mae_improvement >= minimum_mae_improvement
        and calibrated_max_bias < raw_max_bias
    )
    final = fit_bias(rows, ridge=ridge)
    return {
        "schemaVersion": 1,
        "method": "leave-one-calibration-shard-out-additive-bias",
        "sourceCheckpoint": payload.get("checkpoint"),
        "simulatorRevision": payload.get("simulatorRevision"),
        "states": len(rows),
        "ridge": float(ridge),
        "featureNames": list(FEATURE_NAMES),
        "finalCoefficients": list(map(float, final)),
        "raw": raw,
        "crossValidated": cross_validated,
        "maeImprovement": float(mae_improvement),
        "rawMaxAbsSliceBias": float(raw_max_bias),
        "calibratedMaxAbsSliceBias": float(calibrated_max_bias),
        "recommended": "use-additive-bias" if use_calibrator else "raw-critic",
    }


def main():
    args = parse_args()
    if args.ridge < 0:
        raise SystemExit("--ridge must be non-negative")
    payload = json.loads(args.input.read_text())
    report = analyze(
        payload,
        ridge=args.ridge,
        minimum_mae_improvement=args.minimum_mae_improvement,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
