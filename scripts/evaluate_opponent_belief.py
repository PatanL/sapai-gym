#!/usr/bin/env python3
"""Evaluate the learned next-board belief and its reservoir reranking value."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
import torch

from sapai import Team
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.entity_observation import EntityObservationEncoder
from sapai_gym1.rl.opponent_belief import OpponentBoardBelief
from sapai_gym1.rl.opponent_reservoir import (
    OpponentTrajectoryReservoir,
    team_distance,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-reservoir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3805101)
    return parser.parse_args()


def mean_summary(values):
    values = list(map(float, values))
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def stack_histories(encoder, rows):
    encoded = [
        encoder._encode_opponent_history(row["older"], row["latest"])
        for row in rows
    ]
    return {
        key: np.stack([sample[key] for sample in encoded])
        for key in encoded[0]
    }


def target_arrays(encoder, team):
    encoded = encoder._encode_opponent_history(None, team)
    return {
        key: value[-1]
        for key, value in encoded.items()
        if key != "opponent_history_board_mask"
    }


def candidate_score(prediction, encoded, composite):
    items = encoded["opponent_history_item_ids"]
    occupied = encoded["opponent_history_mask"].astype(bool)
    pet = float(
        prediction["pet_log_prob"][
            np.arange(len(items)), items
        ].mean()
    )
    if not composite or not occupied.any():
        return pet
    statuses = encoded["opponent_history_status_ids"]
    numeric = encoded["opponent_history_numeric"]
    levels = np.rint(numeric[:, 2] * 3).astype(np.int64)
    status = float(
        prediction["status_log_prob"][
            np.arange(len(statuses)), statuses
        ][occupied].mean()
    )
    level = float(
        prediction["level_log_prob"][
            np.arange(len(levels)), levels
        ][occupied].mean()
    )
    stats_mse = float(
        np.square(
            prediction["stats"][occupied] - numeric[occupied, :2]
        ).mean()
    )
    return pet + 0.25 * status + 0.25 * level - 0.5 * stats_mse


def softmax_expected(scores, distances, temperature, uniform_mix=0.1):
    scores = np.asarray(scores, dtype=np.float64) / float(temperature)
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    weights = (1.0 - uniform_mix) * weights + uniform_mix / len(weights)
    return float(np.dot(weights, np.asarray(distances, dtype=np.float64)))


def direct_prediction_distance(prediction, target, encoder):
    """Apply the reservoir's public-board distance to the model's modal board."""

    id_to_item = {value: key for key, value in encoder.item_to_id.items()}
    predicted_items = prediction["pet_log_prob"].argmax(-1)
    predicted_statuses = prediction["status_log_prob"].argmax(-1)
    predicted_levels = prediction["level_log_prob"].argmax(-1)
    target_items = target["opponent_history_item_ids"]
    target_statuses = target["opponent_history_status_ids"]
    target_levels = np.rint(
        target["opponent_history_numeric"][:, 2] * 3
    ).astype(np.int64)
    distance = 0.0
    invalid = 0
    for position in range(len(target_items)):
        predicted_name = id_to_item.get(int(predicted_items[position]), "none")
        predicted_pet = predicted_name.startswith("pet-")
        target_pet = int(target_items[position]) != 0
        invalid += int(int(predicted_items[position]) != 0 and not predicted_pet)
        if predicted_pet != target_pet:
            distance += 4.0
            continue
        if not target_pet:
            continue
        distance += 3.0 * (
            int(predicted_items[position]) != int(target_items[position])
        )
        distance += 0.5 * abs(
            int(predicted_levels[position]) - int(target_levels[position])
        )
        distance += 0.5 * (
            int(predicted_statuses[position])
            != int(target_statuses[position])
        )
        distance += 2.0 * float(
            np.abs(
                prediction["stats"][position]
                - target["opponent_history_numeric"][position, :2]
            ).sum()
        )
    return float(distance), invalid / len(target_items)


def summarize(rows):
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and key not in {"turn", "sourceIndex"}
        }
    )
    overall = {
        name: mean_summary(row[name] for row in rows if name in row)
        for name in metric_names
    }
    by_turn = {}
    for turn in sorted({row["turn"] for row in rows}):
        selected = [row for row in rows if row["turn"] == turn]
        by_turn[str(turn)] = {
            name: mean_summary(row[name] for row in selected if name in row)
            for name in metric_names
        }
    by_source = {}
    for source in sorted(
        {str(row["sourceIndex"]) for row in rows}
    ):
        selected = [
            row for row in rows if str(row["sourceIndex"]) == source
        ]
        by_source[source] = {
            name: mean_summary(row[name] for row in selected if name in row)
            for name in metric_names
        }
    return {"overall": overall, "byTurn": by_turn, "bySource": by_source}


@torch.inference_mode()
def main():
    args = parse_args()
    if args.states <= 0 or args.candidate_limit <= 0:
        raise SystemExit("states and candidate-limit must be positive")
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("simulator_revision") != SIMULATOR_REVISION:
        raise SystemExit("Checkpoint simulator revision mismatch")
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    if reservoir.metadata.get("simulatorRevision") != SIMULATOR_REVISION:
        raise SystemExit("Opponent reservoir simulator revision mismatch")
    encoder = EntityObservationEncoder()

    anchors = []
    for trajectory in reservoir.trajectories:
        boards = trajectory["boards"]
        for turn in sorted(boards):
            if turn < 2 or turn - 1 not in boards:
                continue
            anchors.append(
                {
                    "trajectory": trajectory,
                    "sourceIndex": trajectory.get("sourceIndex"),
                    "turn": turn,
                    "older": (
                        None if turn - 2 not in boards else boards[turn - 2]["team"]
                    ),
                    "latest": boards[turn - 1]["team"],
                    "latest_enemy": boards[turn - 1].get("opponentTeam"),
                    "older_enemy": (
                        None
                        if turn - 2 not in boards
                        else boards[turn - 2].get("opponentTeam")
                    ),
                    "target": boards[turn]["team"],
                }
            )
    rng = random.Random(args.seed)
    rng.shuffle(anchors)
    anchors = anchors[: args.states]
    rows = []
    for start in range(0, len(anchors), args.batch_size):
        batch_rows = anchors[start : start + args.batch_size]
        history = stack_histories(encoder, batch_rows)
        tensors = {
            key: torch.as_tensor(value, device=device)
            for key, value in history.items()
        }
        belief = policy.encode_opponent_history(tensors)
        output = policy.predict_opponent_board(belief)
        pet_log_prob = torch.log_softmax(
            output["opponent_pet_logits"].float(), dim=-1
        ).cpu().numpy()
        status_log_prob = torch.log_softmax(
            output["opponent_status_logits"].float(), dim=-1
        ).cpu().numpy()
        level_log_prob = torch.log_softmax(
            output["opponent_level_logits"].float(), dim=-1
        ).cpu().numpy()
        stats = output["opponent_stats"].float().cpu().numpy()
        temporary_probability = torch.sigmoid(
            output["opponent_temporary_logits"].float()
        ).cpu().numpy()

        for local_index, anchor in enumerate(batch_rows):
            prediction = {
                "pet_log_prob": pet_log_prob[local_index],
                "status_log_prob": status_log_prob[local_index],
                "level_log_prob": level_log_prob[local_index],
                "stats": stats[local_index],
                "temporary_probability": temporary_probability[local_index],
            }
            target = target_arrays(encoder, anchor["target"])
            occupied = target["opponent_history_mask"].astype(bool)
            target_items = target["opponent_history_item_ids"]
            predicted_items = prediction["pet_log_prob"].argmax(-1)
            target_levels = np.rint(
                target["opponent_history_numeric"][:, 2] * 3
            ).astype(np.int64)
            predicted_levels = prediction["level_log_prob"].argmax(-1)
            target_statuses = target["opponent_history_status_ids"]
            predicted_statuses = prediction["status_log_prob"].argmax(-1)
            target_temporary = target["opponent_history_numeric"][:, 5:7]
            latest_items = history["opponent_history_item_ids"][
                local_index, -1
            ]
            latest_numeric = history["opponent_history_numeric"][
                local_index, -1
            ]
            row = {
                "turn": int(anchor["turn"]),
                "sourceIndex": anchor["sourceIndex"],
                "petAccuracyAllSlots": float(
                    np.mean(predicted_items == target_items)
                ),
                "petAccuracyOccupied": float(
                    np.mean(predicted_items[occupied] == target_items[occupied])
                )
                if occupied.any()
                else 1.0,
                "boardIdentityExact": float(
                    np.all(predicted_items == target_items)
                ),
                "persistencePetAccuracyAllSlots": float(
                    np.mean(latest_items == target_items)
                ),
                "statusAccuracyOccupied": float(
                    np.mean(
                        predicted_statuses[occupied]
                        == target_statuses[occupied]
                    )
                )
                if occupied.any()
                else 1.0,
                "levelAccuracyOccupied": float(
                    np.mean(predicted_levels[occupied] == target_levels[occupied])
                )
                if occupied.any()
                else 1.0,
                "statMae": float(
                    50
                    * np.abs(
                        prediction["stats"][occupied]
                        - target["opponent_history_numeric"][occupied, :2]
                    ).mean()
                )
                if occupied.any()
                else 0.0,
                "persistenceStatMae": float(
                    50
                    * np.abs(
                        latest_numeric[occupied, :2]
                        - target["opponent_history_numeric"][occupied, :2]
                    ).mean()
                )
                if occupied.any()
                else 0.0,
                "temporaryAccuracyOccupied": float(
                    np.mean(
                        (prediction["temporary_probability"][occupied] >= 0.5)
                        == target_temporary[occupied].astype(bool)
                    )
                )
                if occupied.any()
                else 1.0,
                "persistenceDistance": team_distance(
                    anchor["latest"], anchor["target"]
                ),
            }
            direct_distance, invalid_rate = direct_prediction_distance(
                prediction, target, encoder
            )
            row["directPredictionDistance"] = direct_distance
            row["invalidPredictedItemRate"] = invalid_rate
            legal_belief = OpponentBoardBelief(
                pet_log_prob=prediction["pet_log_prob"],
                status_log_prob=prediction["status_log_prob"],
                level_log_prob=prediction["level_log_prob"],
                stats=prediction["stats"],
                temporary_probability=prediction[
                    "temporary_probability"
                ],
                item_to_id=dict(encoder.item_to_id),
                status_to_id=dict(encoder.status_to_id),
            )
            row["legalSyntheticDistance"] = team_distance(
                legal_belief.modal_team(), anchor["target"]
            )
            matches = reservoir.match(
                anchor["latest"],
                observed_enemy=anchor["latest_enemy"],
                older_observed=anchor["older"],
                older_observed_enemy=anchor["older_enemy"],
                current_turn=anchor["turn"],
                horizon=1,
                limit=args.candidate_limit,
            )
            matches = [
                match
                for match in matches
                if match.trajectory_id != anchor["trajectory"]["id"]
            ]
            if matches:
                distances = [
                    team_distance(anchor["target"], match.teams[0])
                    for match in matches
                ]
                identity_scores = []
                composite_scores = []
                for match in matches:
                    candidate = target_arrays(encoder, match.teams[0])
                    identity_scores.append(
                        candidate_score(prediction, candidate, False)
                    )
                    composite_scores.append(
                        candidate_score(prediction, candidate, True)
                    )
                persistence_candidate = target_arrays(
                    encoder, anchor["latest"]
                )
                augmented_distances = distances + [
                    row["persistenceDistance"]
                ]
                augmented_composite_scores = composite_scores + [
                    candidate_score(
                        prediction, persistence_candidate, True
                    )
                ]
                row.update(
                    {
                        "reservoirUniformDistance": float(
                            statistics.mean(distances)
                        ),
                        "reservoirAnchorNearestDistance": float(distances[0]),
                        "reservoirModelIdentityTopDistance": float(
                            distances[int(np.argmax(identity_scores))]
                        ),
                        "reservoirModelCompositeTopDistance": float(
                            distances[int(np.argmax(composite_scores))]
                        ),
                        "reservoirOracleDistance": float(min(distances)),
                        "reservoirPlusPersistenceModelCompositeTopDistance": (
                            float(
                                augmented_distances[
                                    int(np.argmax(augmented_composite_scores))
                                ]
                            )
                        ),
                    }
                )
                for temperature in (0.25, 0.5, 1.0, 2.0):
                    row[
                        f"reservoirModelCompositeSoftmax{temperature:g}Distance"
                    ] = softmax_expected(
                        composite_scores, distances, temperature
                    )
                    row[
                        "reservoirPlusPersistenceModelCompositeSoftmax"
                        f"{temperature:g}Distance"
                    ] = softmax_expected(
                        augmented_composite_scores,
                        augmented_distances,
                        temperature,
                    )
            rows.append(row)

    payload = {
        "schemaVersion": 1,
        "checkpoint": str(args.checkpoint),
        "simulatorRevision": SIMULATOR_REVISION,
        "reservoir": str(args.opponent_reservoir),
        "states": len(rows),
        "candidateLimit": args.candidate_limit,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"states": len(rows), "summary": payload["summary"]["overall"]}, sort_keys=True))


if __name__ == "__main__":
    main()
