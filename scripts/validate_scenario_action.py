#!/usr/bin/env python3
"""Directly validate a scenario's expert first action against the raw action."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from sapai import Team
from sapai_gym1.rl import (
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
)
from sapai_gym1.rl.curriculum import BUILDERS, PUCT_SCENARIO_NAMES
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.rollout_search import (
    CounterfactualRolloutSearch,
    NeuralSearchEvaluator,
    PublicSearchState,
    RolloutSearchConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-reservoir", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=PUCT_SCENARIO_NAMES, required=True
    )
    parser.add_argument("--scenario-seed", type=int, required=True)
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2740001)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def as_batch(observation, device):
    return {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


@torch.inference_mode()
def raw_action(policy, state, memory, device):
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        output = policy(
            as_batch(state.observation(), device),
            memory=memory.unsqueeze(0).to(device),
            deterministic=True,
        )
    return AutoregressiveAction(*map(int, output.actions[0].cpu()))


def main():
    args = parse_args()
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    scenario_rng = np.random.default_rng(args.scenario_seed)
    env, actions = BUILDERS[args.scenario](scenario_rng)
    if env.previous_agent_state is None:
        env.previous_agent_state = Team.from_state(env.agent.team.state)
    state = PublicSearchState.from_core(env, 0)
    matches = reservoir.match(
        state.core.previous_opponent_state,
        observed_enemy=state.core.previous_agent_state,
        older_observed=state.core.previous_previous_opponent_state,
        older_observed_enemy=state.core.previous_previous_agent_state,
        current_turn=state.core.turn,
        horizon=args.horizon_battles,
        limit=args.opponent_candidates,
    )
    state.set_opponent_trajectories(
        [match.teams for match in matches],
        [match.trajectory_id for match in matches],
    )
    state.set_opponent_transition_model(
        reservoir, candidate_limit=args.opponent_candidates
    )
    memory = policy.initial_memory(1, device="cpu")[0]
    expected = actions[0]
    raw = raw_action(policy, state, memory, device)
    evaluator = NeuralSearchEvaluator(
        policy, device=device, chunk_size=args.batch_size
    )
    search = CounterfactualRolloutSearch(
        evaluator,
        RolloutSearchConfig(
            horizon_battles=args.horizon_battles,
            max_continuation_steps=180,
            leaf_value_weight=0.5,
            audit_rollouts=0,
        ),
    )
    search.simulator_transitions = 0
    root = evaluator.evaluate(state, memory)
    seeds = np.random.default_rng(args.seed).integers(
        0, 2**63 - 1, size=args.rollouts
    )
    records = search._rollouts(
        state,
        [expected, raw],
        root.next_memory,
        seeds,
        allow_nested=False,
        retain_trace=False,
    )
    values = {}
    for action in (expected, raw):
        values[action] = np.asarray(
            [
                record["total"]
                for record in records
                if record["root_action"] == action
            ],
            dtype=np.float64,
        )
    differences = values[expected] - values[raw]
    mean = float(differences.mean())
    standard_error = float(
        differences.std(ddof=1) / math.sqrt(len(differences))
    )
    payload = {
        "scenario": args.scenario,
        "scenarioSeed": args.scenario_seed,
        "rollouts": args.rollouts,
        "horizonBattles": args.horizon_battles,
        "expectedAction": list(expected.as_tuple()),
        "rawAction": list(raw.as_tuple()),
        "meanImprovement": mean,
        "standardError": standard_error,
        "lowerBound95": mean - 1.96 * standard_error,
        "upperBound95": mean + 1.96 * standard_error,
        "expectedMean": float(values[expected].mean()),
        "rawMean": float(values[raw].mean()),
        "simulatorTransitions": search.simulator_transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
