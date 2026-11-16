#!/usr/bin/env python3
"""Find reproducible targeted-scenario seeds where the raw policy disagrees."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import (
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    EntityObservationEncoder,
)
from sapai_gym1.rl.curriculum import (
    BUILDERS,
    PUCT_SCENARIO_NAMES,
    PUCT_SCENARIO_VERSION,
    SCENARIO_NAMES,
)
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario", action="append", choices=PUCT_SCENARIO_NAMES
    )
    parser.add_argument("--failures-per-scenario", type=int, default=2)
    parser.add_argument("--attempts-per-scenario", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2730001)
    return parser.parse_args()


def as_batch(observation, device):
    return {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


@torch.inference_mode()
def choose(policy, encoder, env, memory, device):
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        output = policy(
            as_batch(encoder.encode(env, 0), device),
            memory=memory.unsqueeze(0).to(device),
            deterministic=True,
        )
    return AutoregressiveAction(*map(int, output.actions[0].cpu()))


def main():
    args = parse_args()
    if args.failures_per_scenario <= 0 or args.attempts_per_scenario <= 0:
        raise SystemExit("failure and attempt counts must be positive")
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("simulator_revision") != SIMULATOR_REVISION:
        raise SystemExit("Checkpoint simulator revision mismatch")
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    memory = policy.initial_memory(1, device="cpu")[0]
    encoder = EntityObservationEncoder()
    scenarios = args.scenario or list(SCENARIO_NAMES)
    rows = []
    summary = {}
    for scenario_index, scenario in enumerate(scenarios):
        failures = 0
        matches = 0
        for attempt in range(args.attempts_per_scenario):
            seed = args.seed + scenario_index * 100_000 + attempt
            rng = np.random.default_rng(seed)
            env, actions = BUILDERS[scenario](rng)
            expected = actions[0]
            raw = choose(policy, encoder, env, memory, device)
            if raw == expected:
                matches += 1
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "attempt": attempt,
                    "publicState": view_snapshot(env, 0),
                    "expectedAction": list(expected.as_tuple()),
                    "expectedActionDetail": action_snapshot(
                        env.agent, expected
                    ),
                    "rawAction": list(raw.as_tuple()),
                    "rawActionDetail": action_snapshot(env.agent, raw),
                }
            )
            failures += 1
            if failures >= args.failures_per_scenario:
                break
        summary[scenario] = {
            "attempted": matches + failures,
            "matches": matches,
            "failuresCaptured": failures,
        }
    payload = {
        "checkpoint": str(args.checkpoint),
        "scenarioVersion": PUCT_SCENARIO_VERSION,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "failures": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
