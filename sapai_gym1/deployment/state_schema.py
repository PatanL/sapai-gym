"""Versioned, JSON-safe public state produced by live perception."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _confidence(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Confidence must be in [0, 1]")
    return value


@dataclass(frozen=True)
class PetView:
    item_id: str
    attack: int
    health: int
    level: int = 1
    experience: int = 0
    status_id: str = "none"
    temporary_attack: bool = False
    temporary_health: bool = False
    sell_value: int | None = None
    ability_counter: int = 0
    friends_hurt_counter: int = 0
    confidence: float = 1.0

    def __post_init__(self):
        if not self.item_id.startswith("pet-"):
            raise ValueError("Pet item_id must begin with pet-")
        if not 0 <= self.attack <= 50 or not 0 <= self.health <= 50:
            raise ValueError("Pet stats must be in [0, 50]")
        if self.level not in {1, 2, 3}:
            raise ValueError("Pet level must be 1, 2, or 3")
        if not 0 <= self.experience <= 5:
            raise ValueError("Pet experience must be in [0, 5]")
        if self.ability_counter < 0 or self.friends_hurt_counter < 0:
            raise ValueError("Pet counters cannot be negative")
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True)
class FoodView:
    visual_id: str
    canonical_id: str
    cost: int = 3
    attack: int = 0
    health: int = 0
    status_id: str = "none"
    confidence: float = 1.0

    def __post_init__(self):
        if not self.visual_id.startswith("food-"):
            raise ValueError("Food visual_id must begin with food-")
        if not self.canonical_id.startswith("food-"):
            raise ValueError("Food canonical_id must begin with food-")
        if not 0 <= self.cost <= 30:
            raise ValueError("Food cost is outside the supported range")
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True)
class BoardView:
    slots: tuple[PetView | None, ...] = (None,) * 5
    confidence: float = 1.0

    def __post_init__(self):
        if len(self.slots) != 5:
            raise ValueError("A board must contain exactly five physical slots")
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True)
class ShopSlotView:
    index: int
    kind: str
    pet: PetView | None = None
    food: FoodView | None = None
    frozen: bool = False
    reward_group: int | None = None
    confidence: float = 1.0

    def __post_init__(self):
        if not 0 <= self.index < 10:
            raise ValueError("Shop index must be in [0, 9]")
        if self.kind not in {"pet", "food", "levelup"}:
            raise ValueError("Invalid shop slot kind")
        if self.kind in {"pet", "levelup"} and self.pet is None:
            raise ValueError("Pet/levelup slots require pet data")
        if self.kind == "food" and self.food is None:
            raise ValueError("Food slots require food data")
        if self.pet is not None and self.food is not None:
            raise ValueError("A shop slot cannot contain both pet and food")
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True)
class LiveGameState:
    schema_version: int
    phase: str
    turn: int
    gold: int
    lives: int
    wins: int
    opponent_lives: int
    opponent_wins: int
    team: BoardView
    shop: tuple[ShopSlotView, ...]
    opponent_history: tuple[BoardView, ...] = ()
    shop_attack: int = 0
    shop_health: int = 0
    freeze_toggled_shop_indices: tuple[int, ...] = ()
    last_battle_outcome: str | None = None
    action_count: int = 0
    reorder_count: int = 0
    screenshot_sha256: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported live-state schema version")
        if self.phase not in {"menu", "shop", "battle", "result", "unknown"}:
            raise ValueError("Invalid game phase")
        if self.turn < 0 or self.gold < 0:
            raise ValueError("Turn and gold cannot be negative")
        if not 0 <= self.lives <= 6 or not 0 <= self.opponent_lives <= 6:
            raise ValueError("Lives must be in [0, 6]")
        if not 0 <= self.wins <= 10 or not 0 <= self.opponent_wins <= 10:
            raise ValueError("Wins must be in [0, 10]")
        indices = [slot.index for slot in self.shop]
        if indices != sorted(set(indices)):
            raise ValueError("Shop indices must be sorted and unique")
        if len(self.opponent_history) > 2:
            raise ValueError("V19 consumes at most two opponent boards")
        if self.last_battle_outcome not in {None, "win", "loss", "draw"}:
            raise ValueError("Invalid battle outcome")
        if self.shop_attack < 0 or self.shop_health < 0:
            raise ValueError("Shop stat buffs cannot be negative")
        if any(
            index < 0 or index >= len(self.shop)
            for index in self.freeze_toggled_shop_indices
        ):
            raise ValueError("Invalid freeze-toggled shop index")
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        def pet(value):
            return None if value is None else PetView(**value)

        def board(value):
            return BoardView(
                slots=tuple(pet(item) for item in value["slots"]),
                confidence=value.get("confidence", 1.0),
            )

        shop = []
        for value in payload["shop"]:
            shop.append(
                ShopSlotView(
                    index=value["index"],
                    kind=value["kind"],
                    pet=pet(value.get("pet")),
                    food=(
                        None
                        if value.get("food") is None
                        else FoodView(**value["food"])
                    ),
                    frozen=value.get("frozen", False),
                    reward_group=value.get("reward_group"),
                    confidence=value.get("confidence", 1.0),
                )
            )
        return cls(
            schema_version=payload["schema_version"],
            phase=payload["phase"],
            turn=payload["turn"],
            gold=payload["gold"],
            lives=payload["lives"],
            wins=payload["wins"],
            opponent_lives=payload["opponent_lives"],
            opponent_wins=payload["opponent_wins"],
            team=board(payload["team"]),
            shop=tuple(shop),
            opponent_history=tuple(
                board(value) for value in payload.get("opponent_history", [])
            ),
            shop_attack=payload.get("shop_attack", 0),
            shop_health=payload.get("shop_health", 0),
            freeze_toggled_shop_indices=tuple(
                payload.get("freeze_toggled_shop_indices", [])
            ),
            last_battle_outcome=payload.get("last_battle_outcome"),
            action_count=payload.get("action_count", 0),
            reorder_count=payload.get("reorder_count", 0),
            screenshot_sha256=payload.get("screenshot_sha256"),
            confidence=payload.get("confidence", 1.0),
            metadata=payload.get("metadata", {}),
        )
