"""Entity-token observations and autoregressive legality masks."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sapai import Team
from sapai.data import data

from .action_schema import (
    AutoregressiveAction,
    FlatActionCodec,
    MAX_SHOP_SLOTS,
    MAX_TEAM_SLOTS,
    NUM_ACTION_TYPES,
    NUM_SOURCES,
    NUM_TARGETS,
)

if TYPE_CHECKING:
    from sapai_gym1 import SuperAutoPetsEnv


class EntityKind(IntEnum):
    EMPTY = 0
    GLOBAL = 1
    TEAM_PET = 2
    SHOP_PET = 3
    SHOP_FOOD = 4
    OPPONENT_PET = 5


GLOBAL_TOKENS = 1
TEAM_TOKENS = MAX_TEAM_SLOTS
SHOP_TOKENS = MAX_SHOP_SLOTS
OPPONENT_TOKENS = MAX_TEAM_SLOTS
NUM_ENTITIES = GLOBAL_TOKENS + TEAM_TOKENS + SHOP_TOKENS + OPPONENT_TOKENS
NUM_NUMERIC_FEATURES = 16
OPPONENT_HISTORY_BOARDS = 2
OPPONENT_HISTORY_SLOTS = MAX_TEAM_SLOTS
OPPONENT_HISTORY_NUMERIC_FEATURES = 8


class EntityObservationEncoder:
    """Encode one player's information set as fixed-size entity arrays."""

    def __init__(self):
        item_names = sorted(
            set(data["pets"].keys()) | set(data["foods"].keys())
        )
        self.item_to_id = {"none": 0}
        for name in item_names:
            if name not in ("pet-none", "food-none"):
                self.item_to_id[name] = len(self.item_to_id)

        status_names = ["none", "status-bread", "status-cake"]
        status_names.extend(sorted(data.get("statuses", {}).keys()))
        self.status_to_id = {
            name: index for index, name in enumerate(dict.fromkeys(status_names))
        }

    @property
    def num_items(self) -> int:
        return len(self.item_to_id)

    @property
    def num_statuses(self) -> int:
        return len(self.status_to_id)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "item_ids": spaces.Box(
                    0, self.num_items - 1, (NUM_ENTITIES,), dtype=np.int64
                ),
                "kind_ids": spaces.Box(
                    0, len(EntityKind) - 1, (NUM_ENTITIES,), dtype=np.int64
                ),
                "status_ids": spaces.Box(
                    0, self.num_statuses - 1, (NUM_ENTITIES,), dtype=np.int64
                ),
                "positions": spaces.Box(
                    0, MAX_SHOP_SLOTS, (NUM_ENTITIES,), dtype=np.int64
                ),
                "numeric": spaces.Box(
                    -2.0,
                    2.0,
                    (NUM_ENTITIES, NUM_NUMERIC_FEATURES),
                    dtype=np.float32,
                ),
                "entity_mask": spaces.MultiBinary(NUM_ENTITIES),
                "type_mask": spaces.MultiBinary(NUM_ACTION_TYPES),
                "source_mask": spaces.MultiBinary(
                    (NUM_ACTION_TYPES, NUM_SOURCES)
                ),
                "target_mask": spaces.MultiBinary(
                    (NUM_ACTION_TYPES, NUM_SOURCES, NUM_TARGETS)
                ),
                "opponent_history_item_ids": spaces.Box(
                    0,
                    self.num_items - 1,
                    (OPPONENT_HISTORY_BOARDS, OPPONENT_HISTORY_SLOTS),
                    dtype=np.int64,
                ),
                "opponent_history_status_ids": spaces.Box(
                    0,
                    self.num_statuses - 1,
                    (OPPONENT_HISTORY_BOARDS, OPPONENT_HISTORY_SLOTS),
                    dtype=np.int64,
                ),
                "opponent_history_positions": spaces.Box(
                    0,
                    MAX_TEAM_SLOTS - 1,
                    (OPPONENT_HISTORY_BOARDS, OPPONENT_HISTORY_SLOTS),
                    dtype=np.int64,
                ),
                "opponent_history_numeric": spaces.Box(
                    -2.0,
                    2.0,
                    (
                        OPPONENT_HISTORY_BOARDS,
                        OPPONENT_HISTORY_SLOTS,
                        OPPONENT_HISTORY_NUMERIC_FEATURES,
                    ),
                    dtype=np.float32,
                ),
                "opponent_history_mask": spaces.MultiBinary(
                    (OPPONENT_HISTORY_BOARDS, OPPONENT_HISTORY_SLOTS)
                ),
                "opponent_history_board_mask": spaces.MultiBinary(
                    OPPONENT_HISTORY_BOARDS
                ),
            }
        )

    def encode(self, env: "SuperAutoPetsEnv", agent_idx: int = 0) -> dict:
        player = env.agent if agent_idx == 0 else env.opponent
        opponent_player = env.opponent if agent_idx == 0 else env.agent
        previous_opponent = (
            env.previous_opponent_state
            if agent_idx == 0
            else env.previous_agent_state
        )
        older_opponent = (
            env.previous_previous_opponent_state
            if agent_idx == 0
            else env.previous_previous_agent_state
        )

        item_ids = np.zeros(NUM_ENTITIES, dtype=np.int64)
        kind_ids = np.zeros(NUM_ENTITIES, dtype=np.int64)
        status_ids = np.zeros(NUM_ENTITIES, dtype=np.int64)
        positions = np.zeros(NUM_ENTITIES, dtype=np.int64)
        numeric = np.zeros(
            (NUM_ENTITIES, NUM_NUMERIC_FEATURES), dtype=np.float32
        )
        entity_mask = np.ones(NUM_ENTITIES, dtype=np.int8)

        kind_ids[0] = EntityKind.GLOBAL
        reorder_count = (
            env.reorder_count_agent
            if agent_idx == 0
            else env.reorder_count_opponent
        )
        action_count = (
            env.shop_action_count_agent
            if agent_idx == 0
            else env.shop_action_count_opponent
        )
        toggled = (
            env._freeze_toggled_agent
            if agent_idx == 0
            else env._freeze_toggled_opponent
        )
        seen_orders = (
            env._seen_team_orders_agent
            if agent_idx == 0
            else env._seen_team_orders_opponent
        )
        numeric[0, 0] = min(reorder_count, 2) / 2
        numeric[0, 1] = float(player.last_battle_outcome or 0)
        numeric[0, 2] = float(player.last_battle_outcome is not None)
        numeric[0, 3] = min(player.shop.shop_attack, 50) / 50
        numeric[0, 4] = min(player.shop.shop_health, 50) / 50
        numeric[0, 5] = min(action_count, 50) / 50
        numeric[0, 6] = min(len(toggled), MAX_SHOP_SLOTS) / MAX_SHOP_SLOTS
        numeric[0, 7] = min(len(seen_orders), 3) / 3
        numeric[0, 8:16] = np.array(
            [
                1.0,
                opponent_player.lives / 6,
                opponent_player.wins / 10,
                min(player.gold, 30) / 30,
                min(player.turn, 25) / 25,
                player.lives / 6,
                player.wins / 10,
                0.0,
            ],
            dtype=np.float32,
        )
        numeric[0, 15] = float(player.turn_ended)

        for index in range(MAX_TEAM_SLOTS):
            token = 1 + index
            positions[token] = index
            slot = player.team[index]
            if slot.empty:
                continue
            pet = slot.pet
            kind_ids[token] = EntityKind.TEAM_PET
            self._encode_pet(
                token, pet, item_ids, status_ids, numeric, friendly=True
            )

        reward_groups = sorted(
            {
                slot.reward_group
                for slot in player.shop.slots
                if getattr(slot, "reward_group", None) is not None
            }
        )
        reward_group_rank = {
            group: rank + 1 for rank, group in enumerate(reward_groups)
        }
        reward_group_positions = {
            group: [
                index
                for index, slot in enumerate(player.shop.slots)
                if getattr(slot, "reward_group", None) == group
            ]
            for group in reward_groups
        }

        for index in range(MAX_SHOP_SLOTS):
            token = 1 + MAX_TEAM_SLOTS + index
            positions[token] = index
            if index >= len(player.shop.slots):
                continue
            slot = player.shop[index]
            if slot.empty:
                continue
            if slot.slot_type in ("pet", "levelup"):
                kind_ids[token] = EntityKind.SHOP_PET
                self._encode_pet(
                    token,
                    slot.obj,
                    item_ids,
                    status_ids,
                    numeric,
                    shop=True,
                )
                numeric[token, 5] = slot.cost / 3
                numeric[token, 6] = float(slot.frozen)
                # Kind-conditioned reuse: feature 8 marks friendly team pets;
                # on shop tokens it marks an item already freeze-toggled in
                # the current shop state.
                numeric[token, 8] = float(id(slot.obj) in toggled)
                if slot.slot_type == "levelup":
                    group = slot.reward_group
                    positions_in_group = reward_group_positions.get(group, [index])
                    numeric[token, 11] = 1.0
                    numeric[token, 12] = (
                        reward_group_rank.get(group, 0) / MAX_SHOP_SLOTS
                    )
                    numeric[token, 13] = float(
                        positions_in_group.index(index)
                        if index in positions_in_group
                        else 0
                    )
                    sibling_positions = [
                        position
                        for position in positions_in_group
                        if position != index
                    ]
                    numeric[token, 14] = (
                        sibling_positions[0] / MAX_SHOP_SLOTS
                        if sibling_positions
                        else 0.0
                    )
            elif slot.slot_type == "food":
                kind_ids[token] = EntityKind.SHOP_FOOD
                item_ids[token] = self.item_to_id.get(slot.obj.name, 0)
                numeric[token, 0] = slot.obj.attack / 50
                numeric[token, 1] = slot.obj.health / 50
                numeric[token, 4] = (
                    slot.obj.fd.get("tier", data["foods"][slot.obj.name]["tier"])
                    if isinstance(data["foods"][slot.obj.name]["tier"], int)
                    else 0
                ) / 6
                numeric[token, 5] = slot.cost / 3
                numeric[token, 6] = float(slot.frozen)
                numeric[token, 7] = 1.0
                numeric[token, 8] = float(id(slot.obj) in toggled)
                numeric[token, 9] = 1.0

        previous_team = previous_opponent
        if previous_team is not None and not isinstance(previous_team, Team):
            previous_team = Team.from_state(previous_team)
        for index in range(MAX_TEAM_SLOTS):
            token = 1 + MAX_TEAM_SLOTS + MAX_SHOP_SLOTS + index
            positions[token] = index
            if previous_team is None or previous_team[index].empty:
                continue
            pet = previous_team[index].pet
            kind_ids[token] = EntityKind.OPPONENT_PET
            self._encode_pet(
                token, pet, item_ids, status_ids, numeric, opponent=True
            )

        type_mask, source_mask, target_mask = self._action_masks(env, agent_idx)
        history = self._encode_opponent_history(
            older_opponent, previous_opponent
        )
        return {
            "item_ids": item_ids,
            "kind_ids": kind_ids,
            "status_ids": status_ids,
            "positions": positions,
            "numeric": numeric,
            "entity_mask": entity_mask,
            "type_mask": type_mask,
            "source_mask": source_mask,
            "target_mask": target_mask,
            **history,
        }

    def _encode_opponent_history(self, older, latest):
        item_ids = np.zeros(
            (OPPONENT_HISTORY_BOARDS, OPPONENT_HISTORY_SLOTS),
            dtype=np.int64,
        )
        status_ids = np.zeros_like(item_ids)
        positions = np.broadcast_to(
            np.arange(OPPONENT_HISTORY_SLOTS, dtype=np.int64),
            item_ids.shape,
        ).copy()
        numeric = np.zeros(
            (
                OPPONENT_HISTORY_BOARDS,
                OPPONENT_HISTORY_SLOTS,
                OPPONENT_HISTORY_NUMERIC_FEATURES,
            ),
            dtype=np.float32,
        )
        mask = np.zeros_like(item_ids, dtype=np.int8)
        board_mask = np.zeros(OPPONENT_HISTORY_BOARDS, dtype=np.int8)
        for board_index, board in enumerate((older, latest)):
            if board is None:
                continue
            if not isinstance(board, Team):
                board = Team.from_state(board)
            board_mask[board_index] = 1
            recency = 0.5 if board_index == 0 else 1.0
            for position in range(OPPONENT_HISTORY_SLOTS):
                slot = board[position]
                numeric[board_index, position, 7] = recency
                if slot.empty:
                    continue
                pet = slot.pet
                mask[board_index, position] = 1
                item_ids[board_index, position] = self.item_to_id.get(
                    pet.name, 0
                )
                status_ids[board_index, position] = self.status_to_id.get(
                    pet.status, 0
                )
                numeric[board_index, position, :7] = np.array(
                    [
                        pet.attack / 50,
                        pet.health / 50,
                        pet.level / 3,
                        pet.total_experience / 5,
                        pet.tier / 6 if isinstance(pet.tier, int) else 0,
                        float(pet._until_end_of_battle_attack_buff > 0),
                        float(pet._until_end_of_battle_health_buff > 0),
                    ],
                    dtype=np.float32,
                )
        return {
            "opponent_history_item_ids": item_ids,
            "opponent_history_status_ids": status_ids,
            "opponent_history_positions": positions,
            "opponent_history_numeric": numeric,
            "opponent_history_mask": mask,
            "opponent_history_board_mask": board_mask,
        }

    def _encode_pet(
        self,
        token,
        pet,
        item_ids,
        status_ids,
        numeric,
        *,
        friendly=False,
        shop=False,
        opponent=False,
    ):
        item_ids[token] = self.item_to_id.get(pet.name, 0)
        status_ids[token] = self.status_to_id.get(pet.status, 0)
        numeric[token, 0] = pet.attack / 50
        numeric[token, 1] = pet.health / 50
        numeric[token, 2] = pet.level / 3
        numeric[token, 3] = pet.total_experience / 5
        numeric[token, 4] = pet.tier / 6 if isinstance(pet.tier, int) else 0
        numeric[token, 7] = 1.0
        numeric[token, 8] = float(friendly)
        numeric[token, 9] = float(shop)
        numeric[token, 10] = float(opponent)
        if friendly:
            numeric[token, 11] = min(
                getattr(pet, "ability_counter", 0), 5
            ) / 5
            numeric[token, 12] = min(
                getattr(pet, "friends_hurt_counter", 0), 4
            ) / 4
        numeric[token, 15] = min(getattr(pet, "sell_value", 1), 10) / 10

    def _action_masks(self, env, agent_idx):
        type_mask = np.zeros(NUM_ACTION_TYPES, dtype=np.int8)
        source_mask = np.zeros(
            (NUM_ACTION_TYPES, NUM_SOURCES), dtype=np.int8
        )
        target_mask = np.zeros(
            (NUM_ACTION_TYPES, NUM_SOURCES, NUM_TARGETS), dtype=np.int8
        )
        for flat_action in env._avail_actions(agent_idx):
            action = FlatActionCodec.decode(flat_action)
            action_type, source, target = action.as_tuple()
            type_mask[action_type] = 1
            source_mask[action_type, source] = 1
            target_mask[action_type, source, target] = 1
        return type_mask, source_mask, target_mask


class EntityObservationWrapper(gym.Wrapper):
    """Expose entity observations and autoregressive action triples."""

    def __init__(self, env):
        super().__init__(env)
        self.encoder = EntityObservationEncoder()
        self.observation_space = self.encoder.observation_space()
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTION_TYPES, NUM_SOURCES, NUM_TARGETS]
        )

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self.encoder.encode(self.env, 0), info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        autoregressive = self._validate_action(action)
        flat_action = FlatActionCodec.encode(autoregressive)
        _, reward, terminated, truncated, info = self.env.step(flat_action)
        return (
            self.encoder.encode(self.env, 0),
            reward,
            terminated,
            truncated,
            info,
        )

    def encode_perspective(self, agent_idx: int):
        return self.encoder.encode(self.env, agent_idx)

    def _validate_action(self, action) -> AutoregressiveAction:
        if action.shape != (3,):
            raise ValueError(f"Expected action shape (3,), got {action.shape}")
        action_type, source, target = map(int, action)
        observation = self.encoder.encode(self.env, 0)
        if not observation["type_mask"][action_type]:
            raise ValueError(f"Illegal action type {action_type}")
        if not observation["source_mask"][action_type, source]:
            raise ValueError(f"Illegal source {source} for type {action_type}")
        if not observation["target_mask"][action_type, source, target]:
            raise ValueError(
                f"Illegal target {target} for type {action_type}, source {source}"
            )
        return AutoregressiveAction(action_type, source, target)
