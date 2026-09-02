#!/usr/bin/env python3
"""Create a diagnostic weight-space average of compatible policy checkpoints."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    loaded = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.checkpoints
    ]
    if len(loaded) < 2:
        raise SystemExit("At least two checkpoints are required")
    weights = args.weights or [1.0] * len(loaded)
    if len(weights) != len(loaded) or any(weight < 0 for weight in weights):
        raise SystemExit("Provide one non-negative weight per checkpoint")
    total = sum(weights)
    if total <= 0:
        raise SystemExit("Checkpoint weights must sum to a positive value")
    weights = [weight / total for weight in weights]

    reference = loaded[0]
    for checkpoint in loaded[1:]:
        for key in ("config", "simulator_revision", "source_of_truth"):
            if checkpoint.get(key) != reference.get(key):
                raise SystemExit(f"Incompatible checkpoint field {key}")
        if checkpoint["model"].keys() != reference["model"].keys():
            raise SystemExit("Model state dictionaries have different keys")

    result = copy.deepcopy(loaded[-1])
    averaged = {}
    for name in reference["model"]:
        tensors = [checkpoint["model"][name] for checkpoint in loaded]
        if torch.is_floating_point(tensors[0]):
            value = torch.zeros_like(tensors[0], dtype=torch.float32)
            for weight, tensor in zip(weights, tensors):
                value.add_(tensor.float(), alpha=weight)
            averaged[name] = value.to(tensors[0].dtype)
        else:
            averaged[name] = tensors[-1].clone()
    result["model"] = averaged
    result["averaged_from"] = [str(path) for path in args.checkpoints]
    result["average_weights"] = weights
    result["update"] = max(int(checkpoint.get("update", 0)) for checkpoint in loaded)
    result["transitions"] = max(
        int(checkpoint.get("transitions", 0)) for checkpoint in loaded
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"saved={args.output} weights={weights}")


if __name__ == "__main__":
    main()
