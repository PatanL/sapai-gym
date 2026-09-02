"""Decode nested replay fields and build a unique-object identity catalog."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


ACTION_NAMES = {
    0: "battle",
    1: "match_transition",
    2: "battle_transition",
    4: "start_turn",
    5: "roll",
    6: "buy_pet",
    7: "combine",
    8: "buy_food",
    9: "sell",
    10: "choose_option",
    11: "end_turn",
    12: "choose_name",
}


def decode_nested_json(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def object_key(identifier: Any) -> str | None:
    if not isinstance(identifier, dict):
        return None
    board_id = identifier.get("BoId")
    unique = identifier.get("Uni")
    if board_id is None or unique is None:
        return None
    return f"{board_id}:{unique}"


@dataclass
class ParsedAction:
    index: int
    action_type: int
    action_name: str
    turn: int
    request: dict
    response: dict
    build: dict | None
    battle: dict | None
    raw: dict


class ReplayParser:
    def __init__(self, calculator_root: str | Path):
        calculator_root = Path(calculator_root)
        pets = json.loads(
            (calculator_root / "src/assets/data/pets.json").read_text()
        )
        foods = json.loads(
            (calculator_root / "src/assets/data/food.json").read_text()
        )
        self.pet_names = {str(item["Id"]): item["Name"] for item in pets}
        self.food_names = {str(item["Id"]): item["Name"] for item in foods}

    def parse_perspective(self, perspective: dict) -> list[ParsedAction]:
        raw = perspective.get("raw")
        raw = decode_nested_json(raw)
        actions = raw.get("Actions", []) if isinstance(raw, dict) else []
        parsed = []
        for index, action in enumerate(actions):
            action_type = int(action.get("Type", -1))
            parsed.append(
                ParsedAction(
                    index=index,
                    action_type=action_type,
                    action_name=ACTION_NAMES.get(action_type, "unknown"),
                    turn=int(action.get("Turn", 0) or 0),
                    request=decode_nested_json(action.get("Request")) or {},
                    response=decode_nested_json(action.get("Response")) or {},
                    build=decode_nested_json(action.get("Build")),
                    battle=decode_nested_json(action.get("Battle")),
                    raw=action,
                )
            )
        return parsed

    def build_object_catalog(
        self,
        actions: list[ParsedAction],
        extra_values=(),
    ) -> dict[str, dict]:
        catalog: dict[str, dict] = {}

        def visit(value):
            if isinstance(value, dict):
                key = object_key(value.get("Id"))
                enum = value.get("Enu")
                if key is not None and enum is not None:
                    enum_key = str(enum)
                    pet_name = self.pet_names.get(enum_key)
                    food_name = self.food_names.get(enum_key)
                    catalog[key] = {
                        "enum": enum_key,
                        # Team Wood enum spaces overlap. Preserve both
                        # candidates and resolve by action/build context later.
                        "petName": pet_name,
                        "foodName": food_name,
                    }
                for nested in value.values():
                    visit(decode_nested_json(nested))
            elif isinstance(value, list):
                for nested in value:
                    visit(decode_nested_json(nested))

        for action in actions:
            visit(action.request)
            visit(action.response)
            visit(action.build)
            visit(action.battle)
        for value in extra_values:
            visit(value)

        # A combine request is an exact equality constraint: source and target
        # are copies of the same pet. Purchased copies that are immediately
        # combined often never appear in a serialized build with their own
        # enum, while the retained target does. Propagate only a known pet
        # identity across that explicit constraint; never infer from stats or
        # position.
        changed = True
        while changed:
            changed = False
            for action in actions:
                if action.action_type != 7:
                    continue
                source = object_key(action.request.get("SourceMinionId"))
                target = object_key(action.request.get("TargetMinionId"))
                if source is None or target is None:
                    continue
                source_known = catalog.get(source)
                target_known = catalog.get(target)
                if source_known is None and target_known is not None:
                    catalog[source] = dict(target_known)
                    changed = True
                elif target_known is None and source_known is not None:
                    catalog[target] = dict(source_known)
                    changed = True
        return catalog
