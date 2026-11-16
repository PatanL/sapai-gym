"""Hydrate validated semantic replay states into entity-model training batches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sapai import Food, Pet, Player, Shop, Team
from sapai.data import data
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.rl.action_schema import MAX_SHOP_SLOTS
from sapai_gym1.rl.entity_observation import EntityObservationEncoder


def semantic_state_to_observation(state: dict, encoder=None) -> dict:
    encoder = encoder or EntityObservationEncoder()
    team = Team([])
    for index, item in enumerate(state.get("team", [])[:5]):
        if not item or not item.get("sapaiId"):
            continue
        pet = Pet(item["sapaiId"])
        pet._attack = int(item.get("attack", pet._attack))
        pet._health = int(item.get("health", pet._health))
        # Team Wood replay Exp is absolute (2 means a fresh level-two pet),
        # whereas sapai stores experience relative to the current level.
        absolute_experience = item.get("experience")
        if absolute_experience is None:
            absolute_experience = {1: 0, 2: 2, 3: 5}.get(
                int(item.get("level", 1)), 0
            )
        pet.set_total_experience(int(absolute_experience))
        team[index] = pet

    shop_items = []
    frozen = []
    for item in state.get("shop", [])[:MAX_SHOP_SLOTS]:
        if not item or not item.get("sapaiId"):
            continue
        if item.get("entityType") == "pet":
            obj = Pet(item["sapaiId"])
            obj._attack = int(item.get("attack", obj._attack))
            obj._health = int(item.get("health", obj._health))
        else:
            obj = Food(item["sapaiId"])
            obj.attack = int(item.get("attack", obj.attack))
            obj.health = int(item.get("health", obj.health))
        shop_items.append(obj)
        frozen.append(bool(item.get("frozen", False)))
    shop = Shop(
        shop_items,
        turn=max(1, int(state.get("turn", 1))),
        fixed_rules=True,
    )
    for index, is_frozen in enumerate(frozen):
        shop[index].frozen = is_frozen

    player = Player(
        shop=shop,
        team=team,
        gold=int(state.get("gold", 0)),
        turn=max(1, int(state.get("turn", 1))),
        lives=int(state.get("maxLives") or 6),
    )
    env = SuperAutoPetsEnv(
        opponent_generator=None,
        valid_actions_only=True,
        manual_battles=True,
    )
    env.agent = player
    env.shop_agent = shop
    env.team_agent = team
    env.turn = player.turn
    env.previous_opponent_state = None
    return encoder.encode(env, 0)


class ReplayExampleDataset(Dataset):
    def __init__(self, paths, encoder=None):
        self.encoder = encoder or EntityObservationEncoder()
        self.examples = []
        self.rejected = []
        for path in paths:
            path = Path(path)
            candidates = sorted(path.glob("*.examples.jsonl")) if path.is_dir() else [path]
            for candidate in candidates:
                for line in candidate.read_text().splitlines():
                    if not line.strip():
                        continue
                    example = json.loads(line)
                    if 'recordSchema' in example or 'jointAction' in example:
                        raise ValueError('Typed JointAction records require the joint-aware loader; primitive replay loading is forbidden')
                    recorded_revision = (
                        example.get("validation", {}).get(
                            "simulatorRevision"
                        )
                    )
                    if (
                        "observation" in example
                        and recorded_revision != SIMULATOR_REVISION
                    ):
                        self.rejected.append(
                            {
                                "replay_id": example.get("replay_id"),
                                "turn": example.get("turn"),
                                "unsupported": [
                                    "simulator-revision:"
                                    f"{recorded_revision!r}"
                                ],
                            }
                        )
                        continue
                    unsupported = self._unsupported_ids(example.get("state", {}))
                    if unsupported:
                        self.rejected.append(
                            {
                                "replay_id": example.get("replay_id"),
                                "turn": example.get("turn"),
                                "unsupported": unsupported,
                            }
                        )
                    else:
                        self.examples.append(example)
        if not self.examples:
            raise ValueError("No validated replay examples found")

    @staticmethod
    def _unsupported_ids(state):
        unsupported = []
        for item in list(state.get("team", [])) + list(state.get("shop", [])):
            if not item or not item.get("sapaiId"):
                continue
            item_id = item["sapaiId"]
            collection = data["pets"] if item_id.startswith("pet-") else data["foods"]
            if item_id not in collection:
                unsupported.append(item_id)
        return sorted(set(unsupported))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        if "observation" in example:
            space = self.encoder.observation_space()
            observation = {
                key: np.asarray(value, dtype=space[key].dtype)
                for key, value in example["observation"].items()
            }
        else:
            observation = semantic_state_to_observation(
                example["state"], self.encoder
            )
        battle_outcome = example.get("battle_outcome")
        if battle_outcome == 0:
            return_target = 1.0
        elif battle_outcome == 1:
            return_target = -1.0
        elif battle_outcome == 2:
            return_target = 0.0
        else:
            battle_outcome = -1
            return_target = float("nan")
        weight = float(example.get("teacherWeight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("teacherWeight must be finite and positive")
        return {
            "observation": observation,
            "action": np.asarray(example["action"], dtype=np.int64),
            "battle_outcome": int(battle_outcome),
            "return": np.float32(return_target),
            "weight": np.float32(weight),
            "initial_memory": (
                None
                if example.get("initialMemory") is None
                else np.asarray(
                    example["initialMemory"], dtype=np.float32
                )
            ),
            "replay_id": example["replay_id"],
        }


def collate_replay_examples(batch):
    observations = {
        key: torch.as_tensor(
            np.stack([item["observation"][key] for item in batch])
        ).unsqueeze(0)
        for key in batch[0]["observation"]
    }
    return {
        "observations": observations,
        "actions": torch.as_tensor(
            np.stack([item["action"] for item in batch])
        ).unsqueeze(0),
        "battle_outcomes": torch.tensor(
            [[item["battle_outcome"] for item in batch]], dtype=torch.long
        ),
        "returns": torch.tensor(
            [[item["return"] for item in batch]], dtype=torch.float32
        ),
        "valid_mask": torch.ones(1, len(batch), dtype=torch.bool),
        "example_weights": torch.tensor(
            [[item.get("weight", 1.0) for item in batch]], dtype=torch.float32
        ),
        "initial_memories": [item.get("initial_memory") for item in batch],
        "replay_ids": [item["replay_id"] for item in batch],
    }


class ReplaySequenceDataset(Dataset):
    """Group exact decisions into recurrent shop-turn sequences."""

    def __init__(self, paths, encoder=None):
        self.base = ReplayExampleDataset(paths, encoder=encoder)
        self.encoder = self.base.encoder
        self.rejected = self.base.rejected
        grouped = {}
        for index, example in enumerate(self.base.examples):
            key = (
                example["replay_id"],
                example.get("perspective", "unknown"),
                int(example.get("turn", 0)),
            )
            grouped.setdefault(key, []).append(index)
        self.sequences = [
            sorted(
                indices,
                key=lambda index: int(
                    self.base.examples[index].get("decision", index)
                ),
            )
            for _, indices in sorted(grouped.items())
        ]
        self.replay_ids = [
            self.base.examples[indices[0]]["replay_id"]
            for indices in self.sequences
        ]

    def __len__(self):
        return len(self.sequences)

    @property
    def action_count(self):
        return len(self.base)

    def __getitem__(self, index):
        return [self.base[item_index] for item_index in self.sequences[index]]


def collate_replay_sequences(batch):
    """Pad a batch of variable-length decision sequences to ``[T,B]``."""

    time_steps = max(len(sequence) for sequence in batch)
    batch_size = len(batch)
    observation_keys = batch[0][0]["observation"].keys()
    observations = {}
    for key in observation_keys:
        padded = []
        for time_index in range(time_steps):
            padded.append(
                np.stack(
                    [
                        sequence[
                            min(time_index, len(sequence) - 1)
                        ]["observation"][key]
                        for sequence in batch
                    ]
                )
            )
        observations[key] = torch.as_tensor(np.stack(padded))

    actions = torch.zeros(time_steps, batch_size, 3, dtype=torch.long)
    battle_outcomes = torch.full(
        (time_steps, batch_size), -1, dtype=torch.long
    )
    returns = torch.full(
        (time_steps, batch_size), float("nan"), dtype=torch.float32
    )
    valid_mask = torch.zeros(time_steps, batch_size, dtype=torch.bool)
    example_weights = torch.zeros(
        time_steps, batch_size, dtype=torch.float32
    )
    for batch_index, sequence in enumerate(batch):
        for time_index in range(time_steps):
            item = sequence[min(time_index, len(sequence) - 1)]
            actions[time_index, batch_index] = torch.as_tensor(item["action"])
            if time_index < len(sequence):
                valid_mask[time_index, batch_index] = True
                battle_outcomes[time_index, batch_index] = item[
                    "battle_outcome"
                ]
                returns[time_index, batch_index] = float(item["return"])
                example_weights[time_index, batch_index] = float(
                    item.get("weight", 1.0)
                )
    return {
        "observations": observations,
        "actions": actions,
        "battle_outcomes": battle_outcomes,
        "returns": returns,
        "valid_mask": valid_mask,
        "example_weights": example_weights,
        "initial_memories": [
            sequence[0].get("initial_memory") for sequence in batch
        ],
        "replay_ids": [sequence[0]["replay_id"] for sequence in batch],
        "sequence_lengths": torch.tensor(
            [len(sequence) for sequence in batch], dtype=torch.long
        ),
    }
