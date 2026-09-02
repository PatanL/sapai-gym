#!/usr/bin/env python3
"""Rank every legal action in controlled shop states with CPU inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sapai import Pet, Player, Shop, Team
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.rl import (
    EntityObservationEncoder,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    FlatActionCodec,
)
from sapai_gym1.rl.game_trace import action_snapshot
from sapai_gym1.rl.training import stack_observations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def make_state(name):
    if name == "chocolate-levelup":
        fish = Pet("fish")
        fish.experience = 1
        return 11, 3, Team([fish, "cow", "skunk"]), Shop(
            ["chocolate"], turn=11, fixed_rules=True
        )
    if name == "turtle-pill":
        return 9, 1, Team(["turtle", "ox", "peacock", "rabbit"]), Shop(
            ["sleeping-pill"], turn=9, fixed_rules=True
        )
    if name == "flamingo-pill":
        return 3, 1, Team(["flamingo", "peacock", "fish"]), Shop(
            ["sleeping-pill"], turn=3, fixed_rules=True
        )
    raise ValueError(name)


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
    encoder = EntityObservationEncoder()

    for scenario in ("chocolate-levelup", "turtle-pill", "flamingo-pill"):
        turn, gold, team, shop = make_state(scenario)
        env = SuperAutoPetsEnv(valid_actions_only=True, manual_battles=True)
        env.reset(seed=20260831)
        env.turn = turn
        env.agent = Player(
            shop=shop,
            team=team,
            turn=turn,
            gold=gold,
            record_actions=False,
        )
        env.shop_agent = env.agent.shop
        env.team_agent = env.agent.team
        env.reorder_count_agent = 0
        env.shop_action_count_agent = 0
        env._freeze_toggled_agent.clear()
        env._seen_team_orders_agent = {env._team_order_key(env.agent)}
        env._agent_actions_cache = None
        observation = encoder.encode(env, 0)
        legal = list(env._avail_actions(0))
        observations = [observation] * len(legal)
        actions = torch.tensor(
            [FlatActionCodec.decode(action).as_tuple() for action in legal],
            dtype=torch.long,
        )
        output = policy(
            stack_observations(observations, "cpu"),
            memory=policy.initial_memory(len(legal)),
            actions=actions,
            deterministic=True,
        )
        rows = []
        for flat, log_probability in zip(legal, output.log_prob):
            decoded = FlatActionCodec.decode(flat)
            rows.append(
                (
                    float(log_probability.exp()),
                    action_snapshot(env.agent, decoded)["description"],
                )
            )
        rows.sort(reverse=True)
        print(f"\nscenario={scenario} legal_actions={len(rows)}")
        for rank, (probability, description) in enumerate(rows[: args.top], 1):
            print(f"{rank:2d} {probability:7.3%} {description}")


if __name__ == "__main__":
    main()
