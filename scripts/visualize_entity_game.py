#!/usr/bin/env python3
"""Generate a self-contained, interactive entity-policy match inspector."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch

from sapai import Player
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    ParallelSelfPlayEnv,
)
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.game_trace import (
    action_snapshot,
    audit_state,
    player_snapshot,
    render_game_html,
    state_changes,
    team_snapshot,
    view_snapshot,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a deterministic CPU match as an interactive HTML trace."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-json", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--perspective", choices=("p0", "p1"), default="p0")
    parser.add_argument("--max-decisions", type=int, default=1000)
    return parser.parse_args()


def load_policy(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).cpu()
    policy.load_state_dict(checkpoint["model"])
    return policy.eval(), checkpoint


@torch.inference_mode()
def policy_action(policy, observation, memory):
    batch = {
        key: torch.as_tensor(value, device="cpu").unsqueeze(0)
        for key, value in observation.items()
    }
    output = policy(batch, memory=memory, deterministic=True)
    action = AutoregressiveAction(*map(int, output.actions[0].cpu()))
    type_probabilities = torch.softmax(output.type_logits[0], dim=-1).cpu()
    battle_probabilities = torch.softmax(output.battle_logits[0], dim=-1).cpu()
    type_rows = [
        {
            "type": action_type.name,
            "label": action_type.name.replace("_", " ").title(),
            "probability": float(type_probabilities[int(action_type)]),
        }
        for action_type in ActionType
    ]
    type_rows.sort(key=lambda row: row["probability"], reverse=True)
    diagnostics = {
        "value": float(output.value[0].cpu()),
        "entropy": float(output.entropy[0].cpu()),
        "chosenProbability": float(math.exp(float(output.log_prob[0].cpu()))),
        "sourceProbability": float(
            torch.softmax(output.source_logits[0], dim=-1)[action.source].cpu()
        ),
        "targetProbability": float(
            torch.softmax(output.target_logits[0], dim=-1)[action.target].cpu()
        ),
        "boardStrength": float(output.board_strength[0].cpu()),
        "battleProbabilities": {
            "win": float(battle_probabilities[0]),
            "loss": float(battle_probabilities[1]),
            "draw": float(battle_probabilities[2]),
        },
        "typeProbabilities": type_rows,
    }
    return action, output.memory, diagnostics


def end_turn_snapshot(raw_player_state: dict, before_view: dict) -> dict:
    """Capture end-turn effects before ParallelSelfPlayEnv begins battle."""

    clone = Player.from_state(raw_player_state)
    clone.end_turn()
    result = {
        **before_view,
        "player": player_snapshot(clone),
        "actionCount": int(before_view["actionCount"]) + 1,
    }
    return result


def match_result_label(result: int, viewer_side: int, labels: list[str]) -> str:
    if result == 2:
        return "Match draw"
    return f"{labels[result]} wins the match" + (
        " — viewer wins" if result == viewer_side else " — viewer loses"
    )


def battle_result_label(result: int, viewer_side: int, labels: list[str]) -> str:
    if result == 2:
        return "Battle draw"
    return f"{labels[result]} wins" + (
        " — viewer wins" if result == viewer_side else " — viewer loses"
    )


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    policies_and_checkpoints = [
        load_policy(args.checkpoint),
        load_policy(args.opponent_checkpoint),
    ]
    policies = [entry[0] for entry in policies_and_checkpoints]
    checkpoints = [entry[1] for entry in policies_and_checkpoints]
    labels = [
        f"P0 update {checkpoints[0].get('update')}",
        f"P1 update {checkpoints[1].get('update')}",
    ]
    viewer_side = 0 if args.perspective == "p0" else 1

    env = ParallelSelfPlayEnv()
    observations = env.reset(seed=args.seed)
    memories = [
        policies[0].initial_memory(1, device="cpu"),
        policies[1].initial_memory(1, device="cpu"),
    ]
    frames = []
    action_counts = [Counter(), Counter()]
    decisions = 0
    turn_action_number = 0
    current_turn = int(env.core.turn)
    initial = view_snapshot(env.core, viewer_side)
    frames.append(
        {
            "turn": current_turn,
            "phase": "shop",
            "label": f"Turn {current_turn} shop opens",
            "subtitle": "State before the first shop decision",
            "before": initial,
            "after": initial,
            "action": None,
            "changes": [],
            "warnings": audit_state(None, initial, initial),
            "policy": None,
        }
    )

    while True:
        if decisions >= args.max_decisions:
            raise RuntimeError(
                f"Match exceeded --max-decisions={args.max_decisions}"
            )
        decisions += 1
        active = env.active_players()
        selected = [None, None]
        diagnostics = [None, None]
        visible_before = view_snapshot(env.core, viewer_side)
        visible_raw_state = (
            env.core.agent.state if viewer_side == 0 else env.core.opponent.state
        )
        visible_action = None
        for side in (0, 1):
            if not active[side]:
                continue
            selected[side], memories[side], diagnostics[side] = policy_action(
                policies[side], observations[side], memories[side]
            )
            action_counts[side][ActionType(selected[side].action_type).name] += 1
            if side == viewer_side:
                visible_action = action_snapshot(
                    env.core.agent if side == 0 else env.core.opponent,
                    selected[side],
                )

        step = env.step(selected[0], selected[1])
        observations = step.observations

        if visible_action is not None:
            turn_action_number += 1
            if step.battle_resolved:
                visible_after = end_turn_snapshot(
                    visible_raw_state, visible_before
                )
            else:
                visible_after = view_snapshot(env.core, viewer_side)
            frames.append(
                {
                    "turn": current_turn,
                    "phase": "shop",
                    "label": (
                        f"Action {turn_action_number}: "
                        + visible_action["type"].replace("_", " ").title()
                    ),
                    "before": visible_before,
                    "after": visible_after,
                    "action": visible_action,
                    "changes": state_changes(visible_before, visible_after),
                    "warnings": audit_state(
                        visible_action, visible_before, visible_after
                    ),
                    "policy": diagnostics[viewer_side],
                }
            )

        if not step.battle_resolved:
            continue

        viewer_team = (
            env.core.previous_agent_state
            if viewer_side == 0
            else env.core.previous_opponent_state
        )
        opponent_team = (
            env.core.previous_opponent_state
            if viewer_side == 0
            else env.core.previous_agent_state
        )
        score_after = view_snapshot(env.core, viewer_side)
        battle_changes = [
            (
                f"Hearts: {visible_before['player']['lives']} → "
                f"{score_after['player']['lives']}"
            ),
            (
                f"Wins: {visible_before['player']['wins']} → "
                f"{score_after['player']['wins']}"
            ),
            (
                f"Opponent hearts: {visible_before['opponentScore']['lives']} → "
                f"{score_after['opponentScore']['lives']}"
            ),
            (
                f"Opponent wins: {visible_before['opponentScore']['wins']} → "
                f"{score_after['opponentScore']['wins']}"
            ),
            f"Battle reward: {step.rewards[viewer_side]:+.2f}",
        ]
        viewer_lost = step.battle_result not in {2, viewer_side}
        if (
            current_turn == 2
            and viewer_lost
            and visible_before["player"]["lives"] == 6
            and score_after["player"]["lives"] == 6
        ):
            battle_changes.append(
                "Turn-2 loss flag activated with zero net heart loss"
            )
        frames.append(
            {
                "turn": current_turn,
                "phase": "battle",
                "label": f"Turn {current_turn} battle",
                "subtitle": "Both submitted boards under the seeded battle simulation",
                "before": score_after,
                "after": score_after,
                "action": None,
                "changes": battle_changes,
                "warnings": [],
                "policy": None,
                "battle": {
                    "viewerTeam": team_snapshot(viewer_team),
                    "opponentTeam": team_snapshot(opponent_team),
                    "result": int(step.battle_result),
                    "resultLabel": battle_result_label(
                        int(step.battle_result), viewer_side, labels
                    ),
                    "reward": float(step.rewards[viewer_side]),
                },
            }
        )

        if step.terminated:
            final_result = int(_completed_match_result(env.core))
            break
        current_turn = int(env.core.turn)
        turn_action_number = 0
        opened = view_snapshot(env.core, viewer_side)
        frames.append(
            {
                "turn": current_turn,
                "phase": "shop",
                "label": f"Turn {current_turn} shop opens",
                "subtitle": "Start-turn triggers and the new shop have resolved",
                "before": opened,
                "after": opened,
                "action": None,
                "changes": [],
                "warnings": audit_state(None, opened, opened),
                "policy": None,
            }
        )

    trace = {
        "schemaVersion": 1,
        "metadata": {
            "title": f"{labels[0]} vs {labels[1]} — {args.perspective.upper()} view",
            "seed": int(args.seed),
            "perspective": args.perspective,
            "candidateCheckpoint": str(args.checkpoint),
            "opponentCheckpoint": str(args.opponent_checkpoint),
            "candidateRevision": checkpoints[0].get("simulator_revision"),
            "opponentRevision": checkpoints[1].get("simulator_revision"),
            "simulatorRevision": SIMULATOR_REVISION,
            "inference": "deterministic CPU",
        },
        "frames": frames,
        "summary": {
            "result": final_result,
            "resultLabel": match_result_label(final_result, viewer_side, labels),
            "decisions": decisions,
            "actionCounts": [dict(counter) for counter in action_counts],
            "warnings": sum(len(frame["warnings"]) for frame in frames),
        },
    }
    render_game_html(trace, args.output)
    if args.trace_json:
        args.trace_json.parent.mkdir(parents=True, exist_ok=True)
        args.trace_json.write_text(json.dumps(trace, indent=2))
    print(
        f"output={args.output} frames={len(frames)} warnings={trace['summary']['warnings']} "
        f"result={trace['summary']['resultLabel']}"
    )


if __name__ == "__main__":
    main()
