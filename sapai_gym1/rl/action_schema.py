"""Stable autoregressive action schema shared by environments and policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from sapai.shop import MAX_SHOP_SLOTS


class ActionType(IntEnum):
    END_TURN = 0
    ROLL = 1
    BUY_PET = 2
    BUY_FOOD = 3
    SELL = 4
    COMBINE = 5
    FREEZE = 6
    UNFREEZE = 7
    MOVE = 8


NUM_ACTION_TYPES = len(ActionType)
MAX_TEAM_SLOTS = 5
NO_SOURCE = MAX_SHOP_SLOTS
NO_TARGET = MAX_TEAM_SLOTS
NUM_SOURCES = MAX_SHOP_SLOTS + 1
NUM_TARGETS = MAX_TEAM_SLOTS + 1


@dataclass(frozen=True)
class AutoregressiveAction:
    action_type: int
    source: int = NO_SOURCE
    target: int = NO_TARGET

    def as_tuple(self) -> tuple[int, int, int]:
        return int(self.action_type), int(self.source), int(self.target)


class FlatActionCodec:
    """Lossless bridge between autoregressive triples and Gym Discrete IDs."""

    END_TURN = 0
    BUY_PET = 1  # shop * NUM_TARGETS + target
    BUY_FOOD = BUY_PET + MAX_SHOP_SLOTS * NUM_TARGETS
    COMBINE = BUY_FOOD + MAX_SHOP_SLOTS * NUM_TARGETS
    SELL = COMBINE + MAX_TEAM_SLOTS * MAX_TEAM_SLOTS
    ROLL = SELL + MAX_TEAM_SLOTS
    FREEZE = ROLL + 1
    UNFREEZE = FREEZE + MAX_SHOP_SLOTS
    MOVE = UNFREEZE + MAX_SHOP_SLOTS
    NUM_ACTIONS = MOVE + MAX_TEAM_SLOTS * MAX_TEAM_SLOTS

    @classmethod
    def encode(cls, action: AutoregressiveAction) -> int:
        from .joint_action import JointAction
        if isinstance(action, JointAction):
            raise ValueError('Expand JointAction explicitly before primitive encoding; never drop its mask')
        action_type = ActionType(action.action_type)
        source, target = action.source, action.target
        if action_type is ActionType.END_TURN:
            return cls.END_TURN
        if action_type is ActionType.ROLL:
            return cls.ROLL
        if action_type is ActionType.BUY_PET:
            return cls.BUY_PET + source * NUM_TARGETS + target
        if action_type is ActionType.BUY_FOOD:
            return cls.BUY_FOOD + source * NUM_TARGETS + target
        if action_type is ActionType.COMBINE:
            return cls.COMBINE + source * MAX_TEAM_SLOTS + target
        if action_type is ActionType.SELL:
            return cls.SELL + source
        if action_type is ActionType.FREEZE:
            return cls.FREEZE + source
        if action_type is ActionType.UNFREEZE:
            return cls.UNFREEZE + source
        if action_type is ActionType.MOVE:
            return cls.MOVE + source * MAX_TEAM_SLOTS + target
        raise ValueError(f"Unsupported action type {action_type}")

    @classmethod
    def decode(cls, flat_action: int) -> AutoregressiveAction:
        if flat_action == cls.END_TURN:
            return AutoregressiveAction(ActionType.END_TURN)
        if flat_action == cls.ROLL:
            return AutoregressiveAction(ActionType.ROLL)
        if cls.BUY_PET <= flat_action < cls.BUY_FOOD:
            offset = flat_action - cls.BUY_PET
            return AutoregressiveAction(
                ActionType.BUY_PET, offset // NUM_TARGETS, offset % NUM_TARGETS
            )
        if cls.BUY_FOOD <= flat_action < cls.COMBINE:
            offset = flat_action - cls.BUY_FOOD
            return AutoregressiveAction(
                ActionType.BUY_FOOD, offset // NUM_TARGETS, offset % NUM_TARGETS
            )
        if cls.COMBINE <= flat_action < cls.SELL:
            offset = flat_action - cls.COMBINE
            return AutoregressiveAction(
                ActionType.COMBINE,
                offset // MAX_TEAM_SLOTS,
                offset % MAX_TEAM_SLOTS,
            )
        if cls.SELL <= flat_action < cls.ROLL:
            return AutoregressiveAction(ActionType.SELL, flat_action - cls.SELL)
        if cls.FREEZE <= flat_action < cls.UNFREEZE:
            return AutoregressiveAction(
                ActionType.FREEZE, flat_action - cls.FREEZE
            )
        if cls.UNFREEZE <= flat_action < cls.MOVE:
            return AutoregressiveAction(
                ActionType.UNFREEZE, flat_action - cls.UNFREEZE
            )
        if cls.MOVE <= flat_action < cls.NUM_ACTIONS:
            offset = flat_action - cls.MOVE
            return AutoregressiveAction(
                ActionType.MOVE,
                offset // MAX_TEAM_SLOTS,
                offset % MAX_TEAM_SLOTS,
            )
        raise ValueError(f"Flat action {flat_action} is out of range")
