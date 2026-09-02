#!/usr/bin/env python3
"""Collect and summarize exact actions/end boards for one SAP Library player."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from sapai_gym1.replay import ReplayParser, ReplayReconstructor, SapLibraryClient
from sapai_gym1.replay.parser import object_key


PERK_FOODS = {
    "Honey",
    "Meat Bone",
    "Cake",
    "Garlic",
    "Bread",
    "Chili",
    "Melon",
    "Mushroom",
    "Steak",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("player_id")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path("replay-data/cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calculator-root", type=Path, default=Path("../SAP-Calculator"))
    return parser.parse_args()


def _expert_role(metadata: dict, player_id: str) -> str | None:
    if metadata.get("player_id") == player_id:
        return "player"
    if metadata.get("opponent_id") == player_id:
        return "opponent"
    return None


def _resolved_name(identity: dict, field: str) -> str | None:
    return (identity.get("objects", {}).get(field) or {}).get("name")


def _cake_investments(actions, catalog, reconstructor):
    records = []
    active = {}
    for action in actions:
        identity = reconstructor._resolve_action_identity(action, catalog)
        if action.action_type == 8:
            food_name = _resolved_name(identity, "source")
            target = object_key(action.request.get("Aim"))
            if food_name == "Cake" and target is not None:
                record = {
                    "turn": action.turn,
                    "targetObjectId": target,
                    "retainedAtOrAfterTurn9": False,
                    "firstRetainedTurn": None,
                    "replacedTurn": None,
                    "soldTurn": None,
                    "ambiguousCombineTurn": None,
                }
                records.append(record)
                active[target] = record
            elif food_name in PERK_FOODS and target in active:
                active[target]["replacedTurn"] = action.turn
                del active[target]
        elif action.action_type == 9:
            source = object_key(action.request.get("Minion"))
            if source in active:
                active[source]["soldTurn"] = action.turn
                del active[source]
        elif action.action_type == 7:
            source = object_key(action.request.get("SourceMinionId"))
            if source in active:
                # Equipment transfer when the Cake-equipped copy is consumed
                # depends on combine priority; do not guess it.
                active[source]["ambiguousCombineTurn"] = action.turn
                del active[source]
        elif action.action_type == 0 and action.build and action.turn >= 9:
            board = action.build.get("Bor", action.build)
            items = ((board.get("Mins") or {}).get("Items") or [])
            present = {
                object_key(item.get("Id"))
                for item in items
                if isinstance(item, dict)
            }
            for target, record in list(active.items()):
                if target in present:
                    record["retainedAtOrAfterTurn9"] = True
                    if record["firstRetainedTurn"] is None:
                        record["firstRetainedTurn"] = action.turn
    for record in records:
        terminal_turns = [
            record["replacedTurn"],
            record["soldTurn"],
            record["ambiguousCombineTurn"],
        ]
        record["heldUntilTurn9"] = bool(
            record["retainedAtOrAfterTurn9"]
            or any(turn is not None and turn >= 9 for turn in terminal_turns)
        )
    return records


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = SapLibraryClient(args.cache_dir)
    parser = ReplayParser(args.calculator_root)
    reconstructor = ReplayReconstructor(parser)
    search = client.search_player(
        args.player_id, page=args.page, page_size=args.page_size
    )

    action_counts = Counter()
    bought_pets = Counter()
    bought_foods = Counter()
    sold_pets = Counter()
    combined_pets = Counter()
    end_pet_counts = Counter()
    end_level_counts = Counter()
    turn_pet_counts = defaultdict(Counter)
    line_counts = Counter()
    pet_purchase_turns = defaultdict(Counter)
    food_purchase_turns = defaultdict(Counter)
    replay_rows = []
    cake_investments = []

    for metadata in search.get("results", []):
        if metadata.get("pack") != "Turtle" or metadata.get("opponent_pack") != "Turtle":
            continue
        role = _expert_role(metadata, args.player_id)
        if role is None:
            continue
        bundle = client.fetch_metadata_bundle(metadata)
        perspectives = bundle.get("perspectives", {}).get("perspectives", [])
        perspective = next(
            (item for item in perspectives if item.get("role") == role), None
        )
        if perspective is None:
            continue
        actions = parser.parse_perspective(perspective)
        raw = perspective.get("raw") or {}
        if isinstance(raw, str):
            raw = json.loads(raw)
        catalog = parser.build_object_catalog(
            actions, extra_values=[raw.get("GenesisBuildModel")]
        )
        replay_cakes = _cake_investments(actions, catalog, reconstructor)
        for record in replay_cakes:
            record["replayId"] = metadata["id"]
        cake_investments.extend(replay_cakes)
        resolved = unresolved = 0
        for action in actions:
            action_counts[action.action_name] += 1
            identity = reconstructor._resolve_action_identity(action, catalog)
            if action.action_type in {6, 7, 8, 9}:
                if identity.get("resolved"):
                    resolved += 1
                else:
                    unresolved += 1
            if action.action_type == 6:
                name = _resolved_name(identity, "source")
                if name:
                    bought_pets[name] += 1
                    pet_purchase_turns[name][action.turn] += 1
            elif action.action_type == 8:
                name = _resolved_name(identity, "source")
                if name:
                    bought_foods[name] += 1
                    food_purchase_turns[name][action.turn] += 1
            elif action.action_type == 9:
                name = _resolved_name(identity, "source")
                if name:
                    sold_pets[name] += 1
            elif action.action_type == 7:
                name = _resolved_name(identity, "target")
                if name:
                    combined_pets[name] += 1

        turns = bundle.get("turns", {}).get("turns", [])
        side_key = "user" if role == "player" else "opponent"
        replay_end_names = []
        for turn in turns:
            turn_number = int(turn.get("turn", 0) or 0)
            side = turn.get(side_key) or {}
            pets = side.get("pets") or []
            names = []
            for item in pets:
                name = parser.pet_names.get(str(item.get("id")))
                if not name:
                    continue
                names.append(name)
                turn_pet_counts[turn_number][name] += 1
                end_pet_counts[name] += 1
                level = int(item.get("level", 1) or 1)
                end_level_counts[f"{name}:L{level}"] += 1
            name_set = set(names)
            for label, required in {
                "Elephant+Camel": {"Elephant", "Camel"},
                "Elephant+Blowfish": {"Elephant", "Blowfish"},
                "Shark+Fly": {"Shark", "Fly"},
                "Parrot+Deer": {"Parrot", "Deer"},
                "Mammoth+Tiger": {"Mammoth", "Tiger"},
            }.items():
                if required <= name_set:
                    line_counts[label] += 1
            replay_end_names = names

        replay_rows.append(
            {
                "replayId": metadata["id"],
                "createdAt": metadata.get("created_at"),
                "role": role,
                "actions": len(actions),
                "resolvedObjectActions": resolved,
                "unresolvedObjectActions": unresolved,
                "lastOutcome": metadata.get("last_outcome"),
                "finalBoard": replay_end_names,
            }
        )

    def ordered(counter):
        return dict(counter.most_common())

    report = {
        "playerId": args.player_id,
        "page": args.page,
        "pageSize": args.page_size,
        "searchTotal": search.get("total"),
        "acceptedTurtleMirrors": len(replay_rows),
        "shopDecisionCount": sum(
            action_counts[name]
            for name in (
                "roll", "buy_pet", "buy_food", "sell", "combine", "end_turn"
            )
        ),
        "resolvedObjectRate": (
            sum(row["resolvedObjectActions"] for row in replay_rows)
            / max(
                1,
                sum(
                    row["resolvedObjectActions"]
                    + row["unresolvedObjectActions"]
                    for row in replay_rows
                ),
            )
        ),
        "replays": replay_rows,
        "actionCounts": ordered(action_counts),
        "boughtPets": ordered(bought_pets),
        "boughtFoods": ordered(bought_foods),
        "petPurchaseTurns": {
            name: {str(turn): count for turn, count in sorted(turns.items())}
            for name, turns in sorted(pet_purchase_turns.items())
        },
        "foodPurchaseTurns": {
            name: {str(turn): count for turn, count in sorted(turns.items())}
            for name, turns in sorted(food_purchase_turns.items())
        },
        "soldPets": ordered(sold_pets),
        "combinedPets": ordered(combined_pets),
        "endBoardPetAppearances": ordered(end_pet_counts),
        "endBoardLevelAppearances": ordered(end_level_counts),
        "lineTurnAppearances": ordered(line_counts),
        "cakeInvestments": cake_investments,
        "cakeTurn5Or6Purchases": sum(
            record["turn"] in {5, 6} for record in cake_investments
        ),
        "cakeTurn5Or6RetainedAt9Plus": sum(
            record["turn"] in {5, 6}
            and record["retainedAtOrAfterTurn9"]
            for record in cake_investments
        ),
        "cakeTurn5Or6HeldUntilTurn9": sum(
            record["turn"] in {5, 6} and record["heldUntilTurn9"]
            for record in cake_investments
        ),
        "turnPetAppearances": {
            str(turn): ordered(counter)
            for turn, counter in sorted(turn_pet_counts.items())
        },
    }
    output = args.output_dir / f"player-{args.player_id}-page-{args.page}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"output={output} accepted={len(replay_rows)} "
        f"actions={sum(action_counts.values())} "
        f"resolved_objects={sum(row['resolvedObjectActions'] for row in replay_rows)} "
        f"unresolved_objects={sum(row['unresolvedObjectActions'] for row in replay_rows)}"
    )


if __name__ == "__main__":
    main()
