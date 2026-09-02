"""Detectors for strategically meaningful ordered Turtle P1 compositions."""

from __future__ import annotations


SYNERGY_LABELS = {
    "elephant_camel_components": "Elephant + Camel components",
    "elephant_camel_ordered": "Elephant → Camel",
    "elephant_blowfish_components": "Elephant + Blowfish components",
    "elephant_blowfish_ordered": "Elephant → Blowfish",
    "elephant_hurt_wolverine": "Ordered Elephant hurt engine + Wolverine",
    "shark_fly": "Shark + Fly",
    "shark_faint_core": "Shark + two faint-chain pieces",
    "deer_parrot_components": "Deer + Parrot components",
    "deer_parrot_ordered": "Deer → Parrot",
    "mammoth_mushroom_tiger_components": "Mushroom Mammoth + Tiger components",
    "mammoth_mushroom_tiger_ordered": "Mushroom Mammoth → Tiger",
}

FAINT_CHAIN_PETS = {
    "pet-cricket",
    "pet-spider",
    "pet-sheep",
    "pet-deer",
    "pet-turtle",
    "pet-rooster",
    "pet-mammoth",
    "pet-fly",
}


def detect_team_synergies(team) -> set[str]:
    """Return component and completed-line IDs for one ordered team.

    Empty physical slots are ignored because SAP resolves "nearest friend"
    relationships over living/occupied friends. The remaining list retains
    front-to-back order.
    """

    pets = [slot.pet for slot in team if not slot.empty]
    names = [pet.name for pet in pets]
    found: set[str] = set()

    def has(name: str) -> bool:
        return name in names

    def adjacent(a: str, b: str) -> bool:
        return any(
            names[index] == a and names[index + 1] == b
            for index in range(len(names) - 1)
        )

    if has("pet-elephant") and has("pet-camel"):
        found.add("elephant_camel_components")
    if adjacent("pet-elephant", "pet-camel"):
        found.add("elephant_camel_ordered")

    if has("pet-elephant") and has("pet-blowfish"):
        found.add("elephant_blowfish_components")
    if adjacent("pet-elephant", "pet-blowfish"):
        found.add("elephant_blowfish_ordered")

    if (
        has("pet-wolverine")
        and {
            "elephant_camel_ordered",
            "elephant_blowfish_ordered",
        }
        & found
    ):
        found.add("elephant_hurt_wolverine")

    if has("pet-shark") and has("pet-fly"):
        found.add("shark_fly")
    if has("pet-shark") and sum(name in FAINT_CHAIN_PETS for name in names) >= 2:
        found.add("shark_faint_core")

    if has("pet-deer") and has("pet-parrot"):
        found.add("deer_parrot_components")
    if adjacent("pet-deer", "pet-parrot"):
        found.add("deer_parrot_ordered")

    mushroom_mammoth = any(
        pet.name == "pet-mammoth" and pet.status == "status-extra-life"
        for pet in pets
    )
    if mushroom_mammoth and has("pet-tiger"):
        found.add("mammoth_mushroom_tiger_components")
    if any(
        pets[index].name == "pet-mammoth"
        and pets[index].status == "status-extra-life"
        and pets[index + 1].name == "pet-tiger"
        for index in range(len(pets) - 1)
    ):
        found.add("mammoth_mushroom_tiger_ordered")

    return found
