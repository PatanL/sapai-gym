"""Measured 1920x1080 desktop layout for the current Unity client."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import NormalizedPoint, NormalizedRect


@dataclass(frozen=True)
class ShopLayout:
    pet_slots: int
    food_slots: int


def ordinary_shop_layout(turn: int) -> ShopLayout:
    if turn <= 0:
        raise ValueError("Turn must be positive")
    if turn <= 4:
        return ShopLayout(3, 1)
    return ShopLayout(4, 2)


@dataclass(frozen=True)
class DesktopScreenProfile:
    """Normalized anchors measured from the live 1920x1080 shop screen."""

    # Unity renders the team's back-most pet at the left and front pet at the
    # right. These are stored in model order (slot 0/front first).
    team_points: tuple[NormalizedPoint, ...] = (
        NormalizedPoint(0.5000, 0.4150),
        NormalizedPoint(0.4250, 0.4150),
        NormalizedPoint(0.3500, 0.4150),
        NormalizedPoint(0.2750, 0.4150),
        NormalizedPoint(0.2000, 0.4150),
    )
    pet_shop_points: tuple[NormalizedPoint, ...] = (
        NormalizedPoint(0.2000, 0.6550),
        NormalizedPoint(0.2750, 0.6550),
        NormalizedPoint(0.3500, 0.6550),
        NormalizedPoint(0.4250, 0.6550),
        NormalizedPoint(0.5000, 0.6550),
        NormalizedPoint(0.5750, 0.6550),
        NormalizedPoint(0.6500, 0.6550),
    )
    food_shop_points: tuple[NormalizedPoint, ...] = (
        NormalizedPoint(0.7250, 0.6550),
        NormalizedPoint(0.8000, 0.6550),
    )
    pet_freeze_points: tuple[NormalizedPoint, ...] = (
        NormalizedPoint(0.1770, 0.5600),
        NormalizedPoint(0.2520, 0.5600),
        NormalizedPoint(0.3270, 0.5600),
        NormalizedPoint(0.4020, 0.5600),
        NormalizedPoint(0.4770, 0.5600),
        NormalizedPoint(0.5520, 0.5600),
        NormalizedPoint(0.6270, 0.5600),
    )
    food_freeze_points: tuple[NormalizedPoint, ...] = (
        NormalizedPoint(0.7020, 0.5600),
        NormalizedPoint(0.7770, 0.5600),
    )
    roll_point: NormalizedPoint = NormalizedPoint(0.0750, 0.9420)
    end_turn_point: NormalizedPoint = NormalizedPoint(0.9000, 0.9420)
    opponent_viewer_point: NormalizedPoint = NormalizedPoint(0.9180, 0.0320)
    opponent_viewer_collapse_point: NormalizedPoint = NormalizedPoint(
        0.6030, 0.0320
    )
    opponent_viewer_slot_points: tuple[NormalizedPoint, ...] = (
        NormalizedPoint(0.8500, 0.1200),
        NormalizedPoint(0.8815, 0.1200),
        NormalizedPoint(0.9130, 0.1200),
        NormalizedPoint(0.9445, 0.1200),
        NormalizedPoint(0.9760, 0.1200),
    )

    gold_rect: NormalizedRect = NormalizedRect(0.004, 0.007, 0.055, 0.048)
    lives_rect: NormalizedRect = NormalizedRect(0.061, 0.007, 0.055, 0.048)
    turn_rect: NormalizedRect = NormalizedRect(0.121, 0.007, 0.055, 0.048)
    wins_rect: NormalizedRect = NormalizedRect(0.174, 0.007, 0.075, 0.048)

    def team_crop(self, model_index: int) -> NormalizedRect:
        center = self.team_points[model_index]
        return NormalizedRect(center.x - 0.055, 0.275, 0.110, 0.225)

    def pet_shop_crop(self, screen_index: int) -> NormalizedRect:
        center = self.pet_shop_points[screen_index]
        return NormalizedRect(center.x - 0.055, 0.565, 0.110, 0.195)

    def food_shop_crop(
        self, screen_index: int, count: int = 1
    ) -> NormalizedRect:
        center = self.food_points(count)[screen_index]
        return NormalizedRect(center.x - 0.055, 0.565, 0.110, 0.195)

    def food_points(self, count: int) -> tuple[NormalizedPoint, ...]:
        if count == 1:
            return (self.food_shop_points[1],)
        if count == 2:
            return self.food_shop_points
        raise ValueError("Desktop profile supports one or two food slots")

    def food_freeze_points_for_count(
        self, count: int
    ) -> tuple[NormalizedPoint, ...]:
        if count == 1:
            return (self.food_freeze_points[1],)
        if count == 2:
            return self.food_freeze_points
        raise ValueError("Desktop profile supports one or two food slots")

    def shop_point(self, source_index: int, turn: int) -> NormalizedPoint:
        layout = ordinary_shop_layout(turn)
        if source_index < layout.pet_slots:
            return self.pet_shop_points[source_index]
        food_index = source_index - layout.pet_slots
        if food_index < layout.food_slots:
            return self.food_points(layout.food_slots)[food_index]
        raise IndexError("Shop source is outside the ordinary turn layout")

    def shop_point_for_kinds(
        self, source_index: int, kinds: tuple[str, ...]
    ) -> NormalizedPoint:
        kind = kinds[source_index]
        group = "food" if kind == "food" else "pet"
        rank = sum(
            1
            for prior in kinds[:source_index]
            if ("food" if prior == "food" else "pet") == group
        )
        points = (
            self.food_points(kinds.count("food"))
            if group == "food"
            else self.pet_shop_points
        )
        return points[rank]

    def freeze_point(self, source_index: int, turn: int) -> NormalizedPoint:
        layout = ordinary_shop_layout(turn)
        if source_index < layout.pet_slots:
            return self.pet_freeze_points[source_index]
        food_index = source_index - layout.pet_slots
        if food_index < layout.food_slots:
            return self.food_freeze_points_for_count(layout.food_slots)[
                food_index
            ]
        raise IndexError("Shop source is outside the ordinary turn layout")

    def freeze_point_for_kinds(
        self, source_index: int, kinds: tuple[str, ...]
    ) -> NormalizedPoint:
        kind = kinds[source_index]
        group = "food" if kind == "food" else "pet"
        rank = sum(
            1
            for prior in kinds[:source_index]
            if ("food" if prior == "food" else "pet") == group
        )
        points = (
            self.food_freeze_points_for_count(kinds.count("food"))
            if group == "food"
            else self.pet_freeze_points
        )
        return points[rank]


DESKTOP_1080P = DesktopScreenProfile()
