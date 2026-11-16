#!/usr/bin/env python3
"""Measure checkpoint behavior on targeted recurrent scenario sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.replay import ReplaySequenceDataset, collate_replay_sequences
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader):
    def empty_totals():
        return {
            "actions": 0,
            "typeCorrect": 0,
            "fullCorrect": 0,
            "expertProbabilitySum": 0.0,
            "negativeLogLikelihoodSum": 0.0,
            "weight": 0.0,
        }

    def add(totals, prediction, teacher, target, valid, weights):
        totals["actions"] += int(valid.sum())
        weights = weights * valid.to(weights.dtype)
        totals["weight"] += float(weights.sum())
        totals["typeCorrect"] += float(
            (
                (prediction.actions[:, 0] == target[:, 0]).to(weights.dtype)
                * weights
            ).sum()
        )
        totals["fullCorrect"] += float(
            (
                prediction.actions.eq(target).all(dim=-1).to(weights.dtype)
                * weights
            ).sum()
        )
        totals["expertProbabilitySum"] += float(
            (teacher.log_prob.exp() * weights).sum()
        )
        totals["negativeLogLikelihoodSum"] += float(
            (-teacher.log_prob * weights).sum()
        )

    def finish(totals):
        denominator = max(1e-12, totals["weight"])
        cross_entropy = totals["negativeLogLikelihoodSum"] / denominator
        return {
            "actions": totals["actions"],
            "weight": totals["weight"],
            "typeAccuracy": totals["typeCorrect"] / denominator,
            "fullActionAccuracy": totals["fullCorrect"] / denominator,
            "meanTargetProbability": (
                totals["expertProbabilitySum"] / denominator
            ),
            "weightedCrossEntropy": cross_entropy,
            "weightedPerplexity": float(torch.exp(torch.tensor(cross_entropy))),
        }

    totals = empty_totals()
    first_totals = empty_totals()
    for batch in loader:
        time_steps, batch_size = batch["actions"].shape[:2]
        memory = model.initial_memory(batch_size, device="cpu")
        for batch_index, recorded_memory in enumerate(
            batch.get("initial_memories", [])
        ):
            if recorded_memory is None:
                continue
            recorded_memory = torch.as_tensor(
                recorded_memory, dtype=torch.float32
            )
            if recorded_memory.shape != (model.config.memory_dim,):
                raise ValueError(
                    "Recorded demonstration memory has incompatible shape"
                )
            memory[batch_index].copy_(recorded_memory)
        for time_index in range(time_steps):
            observation = {
                key: value[time_index]
                for key, value in batch["observations"].items()
            }
            target = batch["actions"][time_index]
            valid = batch["valid_mask"][time_index]
            weights = batch.get("example_weights")
            weights = (
                valid.to(torch.float32)
                if weights is None
                else weights[time_index].to(torch.float32)
            )
            prediction = model(
                observation, memory=memory, deterministic=True
            )
            teacher = model(
                observation, memory=memory, actions=target
            )
            add(totals, prediction, teacher, target, valid, weights)
            if time_index == 0:
                add(
                    first_totals,
                    prediction,
                    teacher,
                    target,
                    valid,
                    weights,
                )
            memory = torch.where(valid[:, None], prediction.memory, memory)
    result = finish(totals)
    result["firstDecision"] = finish(first_totals)
    return result


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    dataset = ReplaySequenceDataset([args.data])
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).cpu()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scenario_indices = {}
    for sequence_index, example_indices in enumerate(dataset.sequences):
        example = dataset.base.examples[example_indices[0]]
        scenario = example["teacher"].get("scenario", "search")
        scenario_indices.setdefault(scenario, []).append(sequence_index)

    results = {}
    for scenario, indices in sorted(scenario_indices.items()):
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=min(args.batch_size, len(indices)),
            shuffle=False,
            collate_fn=collate_replay_sequences,
        )
        results[scenario] = evaluate(model, loader)
    all_loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        collate_fn=collate_replay_sequences,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpointUpdate": checkpoint.get("update"),
        # Retain the old field for readers of historical reports, but make its
        # training-time meaning explicit and separately identify the simulator
        # that actually executed this evaluation.
        "simulatorRevision": checkpoint.get("simulator_revision"),
        "checkpointSimulatorRevision": checkpoint.get("simulator_revision"),
        "runtimeSimulatorRevision": SIMULATOR_REVISION,
        "data": str(args.data),
        "sequences": len(dataset),
        "actions": dataset.action_count,
        "overall": evaluate(model, all_loader),
        "scenarios": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
