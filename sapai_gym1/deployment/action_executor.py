"""Map autoregressive policy actions to calibrated live UI gestures."""

from __future__ import annotations

from dataclasses import dataclass

from sapai_gym1.rl import ActionType, AutoregressiveAction

from .geometry import NormalizedPoint
from .screen_profile import DESKTOP_1080P
from .state_schema import LiveGameState


@dataclass(frozen=True)
class ClickGesture:
    point: NormalizedPoint

    def to_dict(self):
        return {"kind": "click", "point": self.point.__dict__}


@dataclass(frozen=True)
class DragGesture:
    source: NormalizedPoint
    target: NormalizedPoint

    def to_dict(self):
        return {
            "kind": "drag",
            "source": self.source.__dict__,
            "target": self.target.__dict__,
        }


@dataclass(frozen=True)
class ClickSequenceGesture:
    points: tuple[NormalizedPoint, ...]

    def to_dict(self):
        return {
            "kind": "click-sequence",
            "points": [point.__dict__ for point in self.points],
        }


class DesktopActionMapper:
    sell_button_point = NormalizedPoint(0.698, 0.942)
    dismiss_detail_point = NormalizedPoint(0.700, 0.150)
    team_food_activation_point = NormalizedPoint(0.350, 0.300)

    def gesture(
        self, action: AutoregressiveAction, *, state: LiveGameState
    ) -> ClickGesture | DragGesture | ClickSequenceGesture:
        action_type = ActionType(action.action_type)
        kinds = tuple(slot.kind for slot in state.shop)
        if action_type is ActionType.ROLL:
            return ClickGesture(DESKTOP_1080P.roll_point)
        if action_type is ActionType.END_TURN:
            return ClickGesture(DESKTOP_1080P.end_turn_point)
        if action_type in {ActionType.FREEZE, ActionType.UNFREEZE}:
            item_point = (
                DESKTOP_1080P.shop_point(action.source, state.turn)
                if not kinds
                else DESKTOP_1080P.shop_point_for_kinds(
                    action.source, kinds
                )
            )
            return ClickSequenceGesture(
                (
                    item_point,
                    self.sell_button_point,
                    self.dismiss_detail_point,
                )
            )
        if action_type is ActionType.BUY_PET:
            return (
                DragGesture(
                    DESKTOP_1080P.shop_point(action.source, state.turn),
                    DESKTOP_1080P.team_points[action.target],
                )
                if not kinds
                else DragGesture(
                    DESKTOP_1080P.shop_point_for_kinds(action.source, kinds),
                    DESKTOP_1080P.team_points[action.target],
                )
            )
        if action_type is ActionType.BUY_FOOD:
            target = (
                self.team_food_activation_point
                if action.target >= len(DESKTOP_1080P.team_points)
                else DESKTOP_1080P.team_points[action.target]
            )
            return (
                DragGesture(
                    DESKTOP_1080P.shop_point(action.source, state.turn),
                    target,
                )
                if not kinds
                else DragGesture(
                    DESKTOP_1080P.shop_point_for_kinds(action.source, kinds),
                    target,
                )
            )
        if action_type is ActionType.COMBINE:
            # Team manipulation uses select-then-destination in the desktop
            # client. The destination pet's detail card opens after a combine;
            # a final inert background click dismisses it before perception.
            return ClickSequenceGesture(
                (
                    DESKTOP_1080P.team_points[action.source],
                    DESKTOP_1080P.team_points[action.target],
                    self.dismiss_detail_point,
                )
            )
        if action_type is ActionType.MOVE:
            return DragGesture(
                DESKTOP_1080P.team_points[action.source],
                DESKTOP_1080P.team_points[action.target],
            )
        if action_type is ActionType.SELL:
            return ClickSequenceGesture(
                (
                    DESKTOP_1080P.team_points[action.source],
                    self.sell_button_point,
                )
            )
        raise ValueError(f"Unsupported action {action_type}")
