#!/usr/bin/env python3
"""Measure direct policy shock from v18 history features on a v17 checkpoint."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from sapai_gym1.rl import (
    EntityTransformerConfig,
    EntityTransformerPolicy,
    FlatActionCodec,
    ParallelSelfPlayEnv,
)
from sapai_gym1.rl.training import stack_observations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def legacy_view(observation: dict) -> dict:
    result = {key: value.copy() for key, value in observation.items()}
    numeric = result["numeric"]
    numeric[0, [0, 5, 6, 7, 15]] = 0
    # Feature 8 was previously zero for every shop token.
    numeric[6:16, 8] = 0
    return result


def collect_states(count: int, seed: int):
    rng = np.random.default_rng(seed)
    env = ParallelSelfPlayEnv()
    observations = env.reset(seed=seed)
    states = []
    while len(states) < count:
        active = env.active_players()
        actions = []
        for side in range(2):
            if not active[side]:
                actions.append(None)
                continue
            states.append(copy.deepcopy(observations[side]))
            legal = tuple(env.core._avail_actions(side))
            actions.append(
                FlatActionCodec.decode(
                    int(legal[int(rng.integers(0, len(legal)))])
                )
            )
        step = env.step(actions[0], actions[1])
        observations = step.observations
        if step.terminated:
            observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return states[:count]


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).eval()
    policy.load_state_dict(checkpoint["model"])
    states = collect_states(args.states, args.seed)
    changed = 0
    value_deltas = []
    type_probability_l1 = []
    for start in range(0, len(states), args.batch_size):
        current_rows = states[start : start + args.batch_size]
        legacy_rows = [legacy_view(row) for row in current_rows]
        current = policy(
            stack_observations(current_rows, "cpu"), deterministic=True
        )
        legacy = policy(
            stack_observations(legacy_rows, "cpu"), deterministic=True
        )
        changed += int((current.actions != legacy.actions).any(dim=1).sum())
        value_deltas.extend(
            (current.value - legacy.value).abs().cpu().tolist()
        )
        current_type = torch.softmax(current.type_logits, dim=-1)
        legacy_type = torch.softmax(legacy.type_logits, dim=-1)
        type_probability_l1.extend(
            (current_type - legacy_type).abs().sum(dim=-1).cpu().tolist()
        )
    print(
        f"states={len(states)} changed_actions={changed} "
        f"changed_rate={changed / len(states):.4f} "
        f"mean_abs_value_delta={np.mean(value_deltas):.6f} "
        f"max_abs_value_delta={np.max(value_deltas):.6f} "
        f"mean_type_l1={np.mean(type_probability_l1):.6f}"
    )


if __name__ == "__main__":
    main()
