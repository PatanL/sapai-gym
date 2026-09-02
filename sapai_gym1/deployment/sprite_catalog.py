"""Pack-aware mapping from simulator IDs to replay-bot sprite assets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


def slugify(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "-", value)
    return value.strip("-").lower()


@dataclass(frozen=True)
class SpriteEntry:
    visual_id: str
    canonical_id: str
    display_name: str
    kind: str
    sprite_path: Path
    tier: int | str
    rollable: bool

    def __post_init__(self):
        if self.kind not in {"pet", "food", "perk"}:
            raise ValueError(f"Unsupported sprite kind {self.kind}")


GENERATED_FOODS = {
    "Better Apple": "food-apple",
    "Best Apple": "food-apple",
    "Bread Crumbs": "food-bread-crumbs",
    "Milk": "food-milk",
    "Better Milk": "food-milk",
    "Best Milk": "food-milk",
}

PACK_ONE_PERKS = {
    "Honey": "status-honey-bee",
    "Meat Bone": "status-bone-attack",
    "Cake": "status-cake",
    "Garlic": "status-garlic-armor",
    "Bread": "status-bread",
    "Chili": "status-splash-attack",
    "Peanut": "status-poison-attack",
    "Melon": "status-melon-armor",
    "Mushroom": "status-extra-life",
    "Steak": "status-steak-attack",
    "Coconut": "status-coconut-shield",
}


class SpriteCatalog:
    def __init__(self, entries):
        self.entries = tuple(entries)
        self.by_visual_id = {entry.visual_id: entry for entry in self.entries}
        if len(self.by_visual_id) != len(self.entries):
            raise ValueError("Duplicate visual IDs in sprite catalog")

    @classmethod
    def from_replay_bot(cls, root: Path, pack: str = "Pack1"):
        root = Path(root)
        pet_records = json.loads((root / "pets.json").read_text())
        food_records = json.loads((root / "food.json").read_text())
        entries = []
        for record in pet_records:
            if pack not in record.get("Packs", []) or not record.get(
                "Rollable", False
            ):
                continue
            item_id = f"pet-{slugify(record['Name'])}"
            entries.append(
                SpriteEntry(
                    visual_id=item_id,
                    canonical_id=item_id,
                    display_name=record["Name"],
                    kind="pet",
                    sprite_path=root
                    / "Sprite"
                    / "Pets"
                    / f"{record['NameId']}.png",
                    tier=record["Tier"],
                    rollable=True,
                )
            )
        for record in food_records:
            if pack not in record.get("Packs", []) or not record.get(
                "Rollable", False
            ):
                continue
            item_id = f"food-{slugify(record['Name'])}"
            entries.append(
                SpriteEntry(
                    visual_id=item_id,
                    canonical_id=item_id,
                    display_name=record["Name"],
                    kind="food",
                    sprite_path=root
                    / "Sprite"
                    / "Food"
                    / f"{record['NameId']}.png",
                    tier=record["Tier"],
                    rollable=True,
                )
            )
        by_name = {record["Name"]: record for record in food_records}
        for display_name, canonical_id in GENERATED_FOODS.items():
            record = by_name[display_name]
            entries.append(
                SpriteEntry(
                    visual_id=f"food-{slugify(display_name)}",
                    canonical_id=canonical_id,
                    display_name=display_name,
                    kind="food",
                    sprite_path=root
                    / "Sprite"
                    / "Food"
                    / f"{record['NameId']}.png",
                    tier=record["Tier"],
                    rollable=False,
                )
            )
        perk_records = json.loads((root / "perks.json").read_text())
        for record in perk_records:
            if record["Name"] not in PACK_ONE_PERKS:
                continue
            entries.append(
                SpriteEntry(
                    visual_id=f"perk-{slugify(record['Name'])}",
                    canonical_id=PACK_ONE_PERKS[record["Name"]],
                    display_name=record["Name"],
                    kind="perk",
                    sprite_path=root
                    / "Sprite"
                    / "Food"
                    / f"{record['NameId']}.png",
                    tier="perk",
                    rollable=False,
                )
            )
        return cls(entries)

    def validate_files(self):
        return [entry for entry in self.entries if not entry.sprite_path.is_file()]

    def entries_of_kind(self, kind):
        return tuple(entry for entry in self.entries if entry.kind == kind)
