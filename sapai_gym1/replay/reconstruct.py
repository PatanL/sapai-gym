"""Conservative replay reconstruction with explicit validation/quarantine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from sapai_gym1.rl.action_schema import (
    ActionType,
    MAX_SHOP_SLOTS,
    NO_SOURCE,
    NO_TARGET,
)

from .parser import ParsedAction, ReplayParser, object_key


def _stat_value(value):
    if not isinstance(value, dict):
        return 0
    return int(value.get("Perm", 0) or 0) + int(value.get("Temp", 0) or 0)


def _slug(entity_type, name):
    if not name:
        return None
    return f"{entity_type}-{name.lower().replace(' ', '-')}"


@dataclass
class QuarantineRecord:
    perspective: str
    action_index: int
    turn: int
    action_type: int
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidatedExample:
    replay_id: str
    perspective: str
    turn: int
    decision: int
    state: dict
    action: tuple[int, int, int]
    battle_outcome: int | None
    final_result: int | None
    validation: dict


@dataclass
class ReconstructionReport:
    replay_id: str
    metadata: dict
    perspectives: list[dict]
    examples: list[ValidatedExample]
    quarantine: list[QuarantineRecord]

    def to_dict(self):
        return {
            "replayId": self.replay_id,
            "metadata": self.metadata,
            "perspectives": self.perspectives,
            "examples": [asdict(example) for example in self.examples],
            "quarantine": [asdict(record) for record in self.quarantine],
        }

    def write(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{self.replay_id}.report.json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True)
        )
        with (output_dir / f"{self.replay_id}.examples.jsonl").open("w") as handle:
            for example in self.examples:
                handle.write(json.dumps(asdict(example), separators=(",", ":")) + "\n")
        with (output_dir / f"{self.replay_id}.quarantine.jsonl").open("w") as handle:
            for record in self.quarantine:
                handle.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")


class ReplayReconstructor:
    """Extract exact end-turn states and audit intermediate action resolvability."""

    def __init__(self, parser: ReplayParser):
        self.parser = parser

    def reconstruct(self, bundle: dict) -> ReconstructionReport:
        replay_id = str(bundle["replayId"])
        perspective_payload = bundle.get("perspectives", {}).get("perspectives", [])
        turns = bundle.get("turns", {}).get("turns", [])
        normalized_by_turn = {
            int(turn.get("turn", 0)): turn for turn in turns if turn.get("turn")
        }
        examples = []
        quarantine = []
        perspective_summaries = []

        for perspective in perspective_payload:
            role = str(perspective.get("role") or "unknown")
            raw = perspective.get("raw") or {}
            actions = self.parser.parse_perspective(perspective)
            catalog = self.parser.build_object_catalog(
                actions,
                extra_values=[raw.get("GenesisBuildModel")],
            )
            resolved = 0
            final_states = 0
            for action in actions:
                identity = self._resolve_action_identity(action, catalog)
                if identity.get("resolved"):
                    resolved += 1
                elif action.action_type in (6, 7, 8, 9):
                    quarantine.append(
                        QuarantineRecord(
                            perspective=role,
                            action_index=action.index,
                            turn=action.turn,
                            action_type=action.action_type,
                            reason="unresolved_object_identity",
                            details=identity,
                        )
                    )

                if action.action_type != 0 or not action.build:
                    continue
                end_action = self._find_end_action(actions, action.turn)
                state = self._parse_build(action.build, catalog)
                build_hash = state.get("boardHash")
                end_hash = self._response_hash(end_action) if end_action else None
                hash_match = (
                    build_hash is not None
                    and end_hash is not None
                    and build_hash == end_hash
                )
                if not hash_match:
                    quarantine.append(
                        QuarantineRecord(
                            perspective=role,
                            action_index=action.index,
                            turn=action.turn,
                            action_type=0,
                            reason="end_turn_hash_mismatch",
                            details={"buildHash": build_hash, "endHash": end_hash},
                        )
                    )
                    continue

                outcome = self._battle_outcome(action.battle, role)
                normalized = normalized_by_turn.get(action.turn, {})
                examples.append(
                    ValidatedExample(
                        replay_id=replay_id,
                        perspective=role,
                        turn=action.turn,
                        decision=end_action.index if end_action else action.index,
                        state=state,
                        action=(int(ActionType.END_TURN), NO_SOURCE, NO_TARGET),
                        battle_outcome=outcome,
                        final_result=self._final_result(raw),
                        validation={
                            "boardHashMatch": True,
                            "finalBuildPresent": True,
                            "normalizedTurnPresent": bool(normalized),
                        },
                    )
                )
                final_states += 1

            perspective_summaries.append(
                {
                    "role": role,
                    "participationId": perspective.get("participationId"),
                    "actions": len(actions),
                    "identityResolvedActions": resolved,
                    "validatedEndStates": final_states,
                    "catalogObjects": len(catalog),
                }
            )

        return ReconstructionReport(
            replay_id=replay_id,
            metadata=self._sanitize_metadata(bundle.get("metadata", {})),
            perspectives=perspective_summaries,
            examples=examples,
            quarantine=quarantine,
        )

    def _parse_build(self, build, catalog):
        board = build.get("Bor", build)
        team_items = ((board.get("Mins") or {}).get("Items") or [])
        team = [self._parse_pet(item, catalog, index) for index, item in enumerate(team_items)]
        while len(team) < 5:
            team.append(None)
        shop = []
        for item in board.get("MiSh") or []:
            shop.append(self._parse_shop_item(item, catalog, "pet"))
        for item in board.get("SpSh") or []:
            shop.append(self._parse_shop_item(item, catalog, "food"))
        return {
            "turn": int(board.get("Tur", 0) or 0),
            "goldSpent": int(board.get("GoSp", 0) or 0),
            "gold": max(0, 10 - int(board.get("GoSp", 0) or 0)),
            "team": team[:5],
            "shop": shop[:MAX_SHOP_SLOTS],
            "boardHash": board.get("Hash"),
            "rawLives": board.get("Sta"),
            "maxLives": board.get("LiMa"),
            "rolls": board.get("Rold"),
        }

    def _parse_pet(self, item, catalog, position):
        if not isinstance(item, dict):
            return None
        identity = self._entity_identity(item, catalog, "pet")
        return {
            **identity,
            "position": position,
            "level": int(item.get("Lvl", 1) or 1),
            "experience": int(item.get("Exp", 0) or 0),
            "attack": _stat_value(item.get("At")),
            "health": _stat_value(item.get("Hp")),
            "perkEnum": item.get("Per") or item.get("Perk"),
            "frozen": bool(item.get("Fro", False)),
        }

    def _parse_shop_item(self, item, catalog, expected_type):
        if not isinstance(item, dict):
            return None
        identity = self._entity_identity(item, catalog, expected_type)
        return {
            **identity,
            "cost": int(item.get("Pri", 3) or 0),
            "frozen": bool(item.get("Fro", False)),
            "attack": _stat_value(item.get("At")),
            "health": _stat_value(item.get("Hp")),
        }

    def _entity_identity(self, item, catalog, expected_type):
        key = object_key(item.get("Id"))
        known = catalog.get(key, {})
        enum = str(item.get("Enu")) if item.get("Enu") is not None else known.get("enum")
        entity_type = expected_type
        names = self.parser.pet_names if entity_type == "pet" else self.parser.food_names
        name = names.get(enum) if enum is not None else None
        if name is None:
            name = known.get(
                "petName" if entity_type == "pet" else "foodName"
            )
        return {
            "objectId": key,
            "enum": enum,
            "entityType": entity_type,
            "name": name,
            "sapaiId": _slug(entity_type, name),
        }

    def _resolve_action_identity(self, action: ParsedAction, catalog):
        request = action.request if isinstance(action.request, dict) else {}
        fields = {}
        if action.action_type == 6:
            fields = {"source": (request.get("MinionId"), "pet")}
        elif action.action_type == 7:
            fields = {
                "source": (request.get("SourceMinionId"), "pet"),
                "target": (request.get("TargetMinionId"), "pet"),
            }
        elif action.action_type == 8:
            fields = {"source": (request.get("SpellId"), "food")}
            # Salad/Sushi/Pizza/Canned Food requests legitimately have no Aim.
            # An absent target is therefore not an unresolved identity; a
            # present target must still resolve exactly.
            if request.get("Aim") is not None:
                fields["target"] = (request.get("Aim"), "pet")
        elif action.action_type == 9:
            fields = {"source": (request.get("Minion"), "pet")}
        else:
            return {"resolved": True, "objects": {}}
        objects = {}
        for field, (identifier, entity_type) in fields.items():
            key = object_key(identifier)
            known = catalog.get(key, {})
            enum = known.get("enum")
            names = (
                self.parser.pet_names
                if entity_type == "pet"
                else self.parser.food_names
            )
            name = names.get(str(enum)) if enum is not None else None
            if name is None:
                name = known.get(
                    "petName" if entity_type == "pet" else "foodName"
                )
            objects[field] = {
                "objectId": key,
                "enum": enum,
                "entityType": entity_type,
                "name": name,
            }
        return {
            "resolved": all(value.get("name") for value in objects.values()),
            "objects": objects,
        }

    @staticmethod
    def _find_end_action(actions, turn):
        candidates = [
            action for action in actions if action.turn == turn and action.action_type == 11
        ]
        return candidates[-1] if candidates else None

    @staticmethod
    def _response_hash(action):
        if action is None or not isinstance(action.response, dict):
            return None
        return (action.response.get("Data") or {}).get("Hash")

    @staticmethod
    def _battle_outcome(battle, role):
        if not isinstance(battle, dict):
            return None
        outcome = battle.get("Outcome")
        if outcome is None:
            return None
        # Team Wood: 1=user win, 2=opponent win, 3=draw.
        outcome = int(outcome)
        if outcome == 3:
            return 2
        if role == "player":
            return 0 if outcome == 1 else 1
        return 0 if outcome == 2 else 1

    @staticmethod
    def _final_result(raw):
        if not isinstance(raw, dict):
            return None
        outcome = raw.get("Outcome")
        return int(outcome) if outcome is not None else None

    @staticmethod
    def _sanitize_metadata(metadata):
        allowed = {
            "id",
            "participation_id",
            "pack",
            "opponent_pack",
            "game_version",
            "match_type",
            "mode",
            "max_player_count",
            "active_player_count",
            "created_at",
            "last_outcome",
        }
        return {key: value for key, value in metadata.items() if key in allowed}
