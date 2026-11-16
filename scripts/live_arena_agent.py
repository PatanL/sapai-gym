#!/usr/bin/env python3
"""Persistent CPU Arena controller; dry-run unless explicitly armed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sapai import Food, Pet
from sapai_gym1.deployment.action_executor import DesktopActionMapper
from sapai_gym1.deployment.arena_controller import AutonomousArenaController
from sapai_gym1.deployment.arena_navigator import ArenaNavigator
from sapai_gym1.deployment.dashboard_client import DashboardClient
from sapai_gym1.deployment.live_perception import LiveScreenObserver
from sapai_gym1.deployment.policy_adapter import LivePolicyEngine
from sapai_gym1.rl import ActionType, AutoregressiveAction, FlatActionCodec
from sapai_gym1.deployment.state_schema import (
    BoardView,
    FoodView,
    LiveGameState,
    PetView,
    ShopSlotView,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dashboard", default="http://100.97.32.64:8770")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--match-id", default="arena-live")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--backend", choices=('cpu', 'mps'), default='cpu')
    parser.add_argument("--navigate-from-main", action="store_true")
    parser.add_argument("--new-arena", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--state-output",
        type=Path,
        default=Path("/workspace/runs/live-agent/current-arena-state.json"),
    )
    parser.add_argument(
        "--pet-model",
        type=Path,
        default=Path("/workspace/runs/live-vision/pet-mobilenet-v1.onnx"),
    )
    parser.add_argument(
        "--sprite-root",
        type=Path,
        default=Path("/workspace/sap-replay-bot"),
    )
    parser.add_argument(
        "--digit-model",
        type=Path,
        default=Path("/workspace/runs/live-vision/stat-digits-v1.npz"),
    )
    parser.add_argument("--armed", action="store_true")
    parser.add_argument(
        "--recover-action",
        help="Replay one already-applied live action into the saved mirror as TYPE,SOURCE,TARGET.",
    )
    return parser.parse_args()


def state_from_new_arena(observation):
    """Bootstrap a fresh Arena shop whose team must still be empty."""

    if any(item is not None for item in observation.team_ids):
        raise RuntimeError(
            "An existing Arena run needs --state so exact stats/XP are not guessed"
        )
    shop = []
    for visible in observation.pets:
        pet = Pet(visible.item_id)
        shop.append(
            ShopSlotView(
                len(shop),
                "pet",
                pet=PetView(
                    visible.item_id,
                    int(pet.attack),
                    int(pet.health),
                    confidence=visible.confidence,
                ),
                confidence=visible.confidence,
            )
        )
    for visible in observation.foods:
        food = Food(visible.item_id)
        shop.append(
            ShopSlotView(
                len(shop),
                "food",
                food=FoodView(
                    visible.item_id,
                    visible.item_id,
                    cost=int(food.cost),
                    attack=int(food.attack),
                    health=int(food.health),
                    status_id=food.status,
                    confidence=visible.confidence,
                ),
                confidence=visible.confidence,
            )
        )
    return LiveGameState(
        schema_version=1,
        phase="shop",
        turn=1,
        gold=10,
        lives=5,
        wins=0,
        opponent_lives=5,
        opponent_wins=0,
        team=BoardView(),
        shop=tuple(shop),
        confidence=observation.confidence,
        metadata={"source": "fresh-arena-visual-bootstrap"},
    )


def main():
    args = parse_args()
    dashboard = DashboardClient(args.dashboard)
    observer = LiveScreenObserver(
        args.pet_model, args.sprite_root, digit_model=args.digit_model
    )
    if args.navigate_from_main:
        ArenaNavigator(dashboard).from_main_menu(
            existing_run=not args.new_arena
        )
    if (
        args.state is None
        and not args.new_arena
        and args.state_output.is_file()
    ):
        state = LiveGameState.from_dict(
            json.loads(args.state_output.read_text())
        )
    elif args.state is None:
        frame, _ = dashboard.wait_for_stable_frame(maximum_wait=12.0)
        observation = observer.observe_shop(frame)
        state = state_from_new_arena(observation)
    else:
        state = LiveGameState.from_dict(json.loads(args.state.read_text()))

    engine = LivePolicyEngine(args.checkpoint, threads=args.threads, backend=args.backend)
    if args.loop:
        controller = AutonomousArenaController(
            dashboard=dashboard,
            observer=observer,
            engine=engine,
            initial_state=state,
            match_id=args.match_id,
            armed=args.armed,
            state_output=args.state_output,
        )
        if args.recover_action:
            action_name, source, target = args.recover_action.split(",")
            action = AutoregressiveAction(
                ActionType[action_name], int(source), int(target)
            )
            controller.env.resolve_action(0, FlatActionCodec.encode(action))
            if action.action_type is ActionType.ROLL:
                frame, _ = dashboard.wait_for_stable_frame(
                    maximum_wait=12.0
                )
                controller.last_frame = frame
                expected = controller._state(frame)
                observation = observer.observe_shop(
                    frame, expected_state=expected
                )
                controller.reconcile_stochastic_shop(observation)
                controller.reconcile_observed_stats(observation)
        result = controller.run()
        print(json.dumps(result, indent=2))
        return
    recommendation = engine.recommend(state, match_id=args.match_id)
    if args.armed and len(recommendation.primitive_plan) > 1:
        raise SystemExit('Joint freeze plans require the verified autonomous controller, not one-shot clicking')
    gesture = DesktopActionMapper().gesture(
        recommendation.action, state=state
    )
    payload = {
        "mode": "armed" if args.armed else "dry-run",
        "checkpointUpdate": engine.checkpoint.get("update"),
        "state": None if args.state is None else str(args.state),
        "recommendation": recommendation.to_dict(),
        "gesture": gesture.to_dict(),
    }
    dashboard.publish_agent(payload)
    print(json.dumps(payload, indent=2))
    if args.armed:
        dashboard.execute_gesture(gesture)
        frame, difference = dashboard.wait_for_stable_frame()
        print(
            json.dumps(
                {
                    "executed": True,
                    "stableFrameDifference": difference,
                    "frameShape": list(frame.shape),
                }
            )
        )


if __name__ == "__main__":
    main()
