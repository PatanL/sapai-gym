"""Translate recognized live state into a v19 policy recommendation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np
import torch

from sapai import Player, Shop, Team
from sapai.foods import Food
from sapai.pets import Pet
from sapai.shop import ShopSlot
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    EntityObservationEncoder,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    FlatActionCodec,
)
from sapai_gym1.rl.decision_api import decode_decisions
from sapai_gym1.rl.joint_action import JointAction, JOINT_SCHEMA
from .inference_backend import inference_device

from .state_schema import (
    BoardView,
    FoodView,
    LiveGameState,
    PetView,
    ShopSlotView,
)


OUTCOME_VALUE = {None: None, "win": 1, "loss": -1, "draw": 0}
OUTCOME_WINNER = {None: None, "win": True, "loss": False, "draw": True}


@dataclass(frozen=True)
class PolicyRecommendation:
    action: AutoregressiveAction
    action_type: str
    source: int
    target: int
    value: float
    log_probability: float
    probability: float
    entropy: float
    state_confidence: float
    decision: JointAction | None = None
    primitive_plan: tuple[AutoregressiveAction, ...] = ()
    decision_id: str | None = None
    state_digest: str | None = None

    def to_dict(self):
        return {
            "action": {
                "type": self.action_type,
                "typeId": int(self.action.action_type),
                "source": self.source,
                "target": self.target,
            },
            "value": self.value,
            "logProbability": self.log_probability,
            "probability": self.probability,
            "entropy": self.entropy,
            "stateConfidence": self.state_confidence,
            "decision": None if self.decision is None else self.decision.to_dict(),
            "primitivePlan": [list(action.as_tuple()) for action in self.primitive_plan],
            "decisionId": self.decision_id,
            "stateDigest": self.state_digest,
        }


@dataclass(frozen=True)
class PlannedAction:
    index: int
    recommendation: PolicyRecommendation
    barrier_after: str | None

    def to_dict(self):
        return {
            "index": self.index,
            "recommendation": self.recommendation.to_dict(),
            "barrierAfter": self.barrier_after,
        }


def _pet_from_view(view: PetView) -> Pet:
    pet = Pet(view.item_id)
    # The opponent UI exposes the presence, but not magnitude, of temporary
    # stats. Preserve the displayed total and public flag with a conservative
    # one-point split. Own-state tracking can later provide exact amounts.
    temporary_attack = int(view.temporary_attack and view.attack > 0)
    temporary_health = int(view.temporary_health and view.health > 0)
    pet._attack = view.attack - temporary_attack
    pet._health = view.health - temporary_health
    pet._until_end_of_battle_attack_buff = temporary_attack
    pet._until_end_of_battle_health_buff = temporary_health
    pet.level = view.level
    pet.experience = view.experience
    pet.status = view.status_id
    pet.ability_counter = int(view.ability_counter)
    pet.friends_hurt_counter = int(view.friends_hurt_counter)
    if view.sell_value is not None:
        pet.sell_value = view.sell_value
    return pet


def _team_from_view(board: BoardView) -> Team:
    team = Team()
    for index, view in enumerate(board.slots):
        if view is not None:
            team[index] = _pet_from_view(view)
    return team


def _food_from_view(view: FoodView) -> Food:
    food = Food(view.canonical_id)
    food.cost = view.cost
    food.attack = view.attack
    food.health = view.health
    food.status = view.status_id
    return food


def environment_from_live_state(state: LiveGameState) -> SuperAutoPetsEnv:
    indices = [slot.index for slot in state.shop]
    if indices != list(range(len(indices))):
        raise ValueError(
            "Policy reconstruction currently requires contiguous shop indices"
        )
    seed_state = np.random.RandomState(0).get_state()
    shop_slots = []
    for slot in state.shop:
        if slot.kind in {"pet", "levelup"}:
            obj = _pet_from_view(slot.pet)
        else:
            obj = _food_from_view(slot.food)
        shop_slot = ShopSlot(
            obj=obj,
            slot_type=slot.kind,
            frozen=slot.frozen,
            turn=state.turn,
            reward_group=slot.reward_group,
            seed_state=seed_state,
        )
        if slot.food is not None:
            shop_slot.cost = slot.food.cost
        shop_slots.append(shop_slot)
    shop = Shop(
        slots=shop_slots,
        turn=state.turn,
        pack="StandardPack",
        seed_state=seed_state,
        fixed_rules=False,
        shop_attack=state.shop_attack,
        shop_health=state.shop_health,
    )
    team = _team_from_view(state.team)
    player = Player(
        shop=shop,
        team=team,
        lives=state.lives,
        gold=state.gold,
        turn=state.turn,
        wins=state.wins,
        lf_winner=OUTCOME_WINNER[state.last_battle_outcome],
        last_battle_outcome=OUTCOME_VALUE[state.last_battle_outcome],
        record_actions=False,
    )

    env = SuperAutoPetsEnv(
        opponent_generator=None,
        valid_actions_only=True,
        manual_battles=True,
    )
    env.agent = player
    env.shop_agent = shop
    env.team_agent = team
    env.turn = state.turn
    latest_opponent = (
        state.opponent_history[-1]
        if state.opponent_history
        else BoardView()
    )
    opponent_team = _team_from_view(latest_opponent)
    env.opponent = Player(
        team=opponent_team,
        lives=state.opponent_lives,
        wins=state.opponent_wins,
        gold=0,
        turn=state.turn,
        record_actions=False,
    )
    env.previous_opponent_state = (
        _team_from_view(state.opponent_history[-1])
        if state.opponent_history
        else None
    )
    env.previous_previous_opponent_state = (
        _team_from_view(state.opponent_history[-2])
        if len(state.opponent_history) == 2
        else None
    )
    env.previous_agent_state = None
    env.previous_previous_agent_state = None
    env.shop_action_count_agent = state.action_count
    env.reorder_count_agent = state.reorder_count
    env._freeze_toggled_agent = {
        id(env.agent.shop[index].obj)
        for index in state.freeze_toggled_shop_indices
    }
    env._seen_team_orders_agent = {env._team_order_key(env.agent)}
    env._agent_actions_cache = None
    return env


def _pet_view_from_pet(pet, confidence: float = 1.0) -> PetView:
    return PetView(
        item_id=pet.name,
        attack=int(pet.attack),
        health=int(pet.health),
        level=int(pet.level),
        experience=int(pet.experience),
        status_id=pet.status,
        temporary_attack=bool(
            getattr(pet, "_until_end_of_battle_attack_buff", 0)
        ),
        temporary_health=bool(
            getattr(pet, "_until_end_of_battle_health_buff", 0)
        ),
        sell_value=int(pet.sell_value),
        ability_counter=int(pet.ability_counter),
        friends_hurt_counter=int(pet.friends_hurt_counter),
        confidence=confidence,
    )


def _food_view_from_food(food, cost: int, confidence: float = 1.0) -> FoodView:
    return FoodView(
        visual_id=food.name,
        canonical_id=food.name,
        cost=int(cost),
        attack=int(food.attack),
        health=int(food.health),
        status_id=food.status,
        confidence=confidence,
    )


def live_state_from_environment(
    env: SuperAutoPetsEnv,
    *,
    phase: str = "shop",
    opponent_history: tuple[BoardView, ...] = (),
    screenshot_sha256: str | None = None,
    confidence: float = 1.0,
    metadata: dict | None = None,
) -> LiveGameState:
    """Project the persistent deployment mirror into the public v19 schema."""

    team = BoardView(
        tuple(
            None if slot.empty else _pet_view_from_pet(slot.pet, confidence)
            for slot in env.agent.team
        ),
        confidence=confidence,
    )
    shop = []
    for index, slot in enumerate(env.agent.shop):
        if slot.slot_type in {"pet", "levelup"}:
            shop.append(
                ShopSlotView(
                    index=index,
                    kind=slot.slot_type,
                    pet=_pet_view_from_pet(slot.obj, confidence),
                    frozen=bool(slot.frozen),
                    reward_group=slot.reward_group,
                    confidence=confidence,
                )
            )
        else:
            shop.append(
                ShopSlotView(
                    index=index,
                    kind="food",
                    food=_food_view_from_food(slot.obj, slot.cost, confidence),
                    frozen=bool(slot.frozen),
                    reward_group=slot.reward_group,
                    confidence=confidence,
                )
            )
    outcome = env.agent.last_battle_outcome
    rendered_outcome = (
        None
        if outcome is None
        else "win"
        if outcome > 0
        else "loss"
        if outcome < 0
        else "draw"
    )
    return LiveGameState(
        schema_version=1,
        phase=phase,
        turn=int(env.agent.turn),
        gold=int(env.agent.gold),
        lives=int(env.agent.lives),
        wins=int(env.agent.wins),
        opponent_lives=int(env.opponent.lives),
        opponent_wins=int(env.opponent.wins),
        team=team,
        shop=tuple(shop),
        opponent_history=tuple(opponent_history),
        shop_attack=int(env.agent.shop.shop_attack),
        shop_health=int(env.agent.shop.shop_health),
        freeze_toggled_shop_indices=tuple(
            index
            for index, slot in enumerate(env.agent.shop)
            if id(slot.obj) in env._freeze_toggled_agent
        ),
        last_battle_outcome=rendered_outcome,
        action_count=int(env.shop_action_count_agent),
        reorder_count=int(env.reorder_count_agent),
        screenshot_sha256=screenshot_sha256,
        confidence=confidence,
        metadata={} if metadata is None else metadata,
    )


class LivePolicyEngine:
    def __init__(self, checkpoint: Path, threads: int = 4, backend: str = 'cpu'):
        torch.set_num_threads(max(1, int(threads)))
        self.device = inference_device(backend)
        checkpoint_data = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        # Keep provenance/display metadata, not Adam state or a redundant
        # CPU copy of every weight for the lifetime of the deployed process.
        self.checkpoint = {key: value for key, value in checkpoint_data.items()
                           if key not in {'model', 'optimizer'}}
        if (checkpoint_data['config'].get('end_freeze', False)
                and checkpoint_data.get('training_semantics', {}).get('action_schema') != JOINT_SCHEMA):
            raise ValueError('Joint checkpoint requires matching action-schema metadata')
        self.policy = EntityTransformerPolicy(
            EntityTransformerConfig(**checkpoint_data["config"])
        ).to(self.device)
        self.policy.load_state_dict(checkpoint_data["model"])
        self.policy.eval()
        self.encoder = EntityObservationEncoder()
        self.memory = None
        self.match_id = None
        self._pending = None
        self._memory_policy = self.policy

    def reset_memory(self):
        self.memory = self.policy.initial_memory(1, device=self.device)
        self._pending = None
        self._memory_policy = self.policy

    def snapshot_memory(self):
        return None if self.memory is None else self.memory.clone()

    def restore_memory(self, memory):
        self.memory = None if memory is None else memory.to(self.device).clone()
        self._pending = None

    @staticmethod
    def _observation_digest(observation):
        digest = hashlib.sha256()
        for key in sorted(observation):
            value = np.ascontiguousarray(observation[key])
            digest.update(repr((key, value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def validate_recommendation(self, recommendation, env):
        """Call after re-observing, immediately before sending the first click."""
        if (self._pending is None or self._pending[0] != recommendation.decision_id
                or self._pending[3] is not self.policy):
            raise ValueError('Expired recommendation or replaced model')
        if self._state_digest(env, self.encoder.encode(env, 0)) != recommendation.state_digest:
            self._pending = None
            raise ValueError('Visible state changed; discard plan and re-observe')

    def _state_digest(self, env, observation):
        # Include unnormalized public scalars/stats as well as policy features:
        # clipping/normalization must not make two UI states share a digest.
        public = live_state_from_environment(env).to_dict()
        payload = json.dumps(public, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256((self._observation_digest(observation) + payload).encode()).hexdigest()

    def commit_recommendation(self, recommendation, *, confirmed=False):
        """Commit proposed memory exactly once, only after executor confirmation."""
        if confirmed is not True:
            raise ValueError('Executor confirmation is required before memory commit')
        if (self._pending is None or self._pending[0] != recommendation.decision_id
                or self._pending[3] is not self.policy):
            raise ValueError('Expired/already committed recommendation or replaced model')
        self.memory = self._pending[1].clone()
        self._pending = None

    @torch.inference_mode()
    def recommend_environment(
        self,
        env: SuperAutoPetsEnv,
        *,
        match_id: str | None = None,
        deterministic: bool = True,
        state_confidence: float = 1.0,
    ) -> PolicyRecommendation:
        """Propose without advancing live memory; repeated requests are idempotent."""

        if self.memory is None or self._memory_policy is not self.policy or (
            match_id is not None and match_id != self.match_id
        ):
            self.reset_memory()
        self.match_id = match_id
        observation = self.encoder.encode(env, 0)
        digest = self._state_digest(env, observation)
        if self._pending is not None and self._pending[3] is self.policy:
            cached = self._pending[2]
            if cached.state_digest == digest:
                return cached
        batch = {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for key, value in observation.items()
        }
        features = self.policy.encode_decision(batch, self.memory)
        decisions, output = decode_decisions(self.policy, features, deterministic=deterministic)
        decision = decisions[0]
        action = decision.primitive
        log_probability = float(output.log_prob[0])
        recommendation = PolicyRecommendation(
            action=action,
            action_type=ActionType(action.action_type).name,
            source=action.source,
            target=action.target,
            value=float(output.value[0]),
            log_probability=log_probability,
            probability=math.exp(log_probability),
            entropy=float(output.entropy[0]),
            state_confidence=state_confidence,
            decision=decision,
            primitive_plan=decision.expand(env, 0),
            decision_id=uuid.uuid4().hex,
            state_digest=digest,
        )
        self._pending = (recommendation.decision_id, output.memory.clone(), recommendation, self.policy)
        return recommendation

    @torch.inference_mode()
    def recommend(
        self,
        state: LiveGameState,
        *,
        match_id: str | None = None,
        deterministic: bool = True,
    ) -> PolicyRecommendation:
        if self.memory is None or (
            match_id is not None and match_id != self.match_id
        ):
            self.reset_memory()
        self.match_id = match_id
        env = environment_from_live_state(state)
        return self.recommend_environment(
            env,
            match_id=match_id,
            deterministic=deterministic,
            state_confidence=state.confidence,
        )

    @torch.inference_mode()
    def plan_until_barrier(
        self,
        state: LiveGameState,
        *,
        max_actions: int = 20,
        deterministic: bool = True,
    ) -> tuple[PlannedAction, ...]:
        """Roll policy/simulator forward without committing recurrent memory.

        The returned sequence is only a proposal. A live executor must verify
        every action before continuing and discard the suffix after a mismatch.
        """

        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        env = environment_from_live_state(state)
        memory = (
            self.memory.clone()
            if self.memory is not None and self._memory_policy is self.policy
            else self.policy.initial_memory(1, device=self.device)
        )
        planned = []
        for index in range(max_actions):
            observation = self.encoder.encode(env, 0)
            batch = {
                key: torch.as_tensor(value, device=self.device).unsqueeze(0)
                for key, value in observation.items()
            }
            features = self.policy.encode_decision(batch, memory)
            decisions, output = decode_decisions(self.policy, features, deterministic=deterministic)
            memory = output.memory
            decision = decisions[0]
            action = decision.primitive
            log_probability = float(output.log_prob[0])
            recommendation = PolicyRecommendation(
                action=action,
                action_type=ActionType(action.action_type).name,
                source=action.source,
                target=action.target,
                value=float(output.value[0]),
                log_probability=log_probability,
                probability=math.exp(log_probability),
                entropy=float(output.entropy[0]),
                state_confidence=state.confidence,
                decision=decision,
                primitive_plan=decision.expand(env, 0),
                state_digest=self._state_digest(env, observation),
            )
            barrier = self._barrier_reason(env, action)
            planned.append(PlannedAction(index, recommendation, barrier))
            for primitive in recommendation.primitive_plan:
                env.resolve_action(0, FlatActionCodec.encode(primitive))
            if barrier is not None or ActionType(action.action_type) is ActionType.END_TURN:
                break
        return tuple(planned)

    @staticmethod
    def _barrier_reason(env, action):
        action_type = ActionType(action.action_type)
        player = env.agent
        if action_type is ActionType.ROLL:
            return "new-shop"
        if action_type is ActionType.END_TURN:
            return "battle"
        if action_type in {ActionType.COMBINE, ActionType.BUY_PET}:
            if action_type is ActionType.COMBINE:
                source_pet = player.team[action.source].pet
                target_pet = player.team[action.target].pet
            else:
                source_pet = player.shop[action.source].obj
                target_pet = (
                    None
                    if player.team[action.target].empty
                    else player.team[action.target].pet
                )
            if target_pet is not None:
                total_experience = (
                    source_pet.total_experience
                    + target_pet.total_experience
                    + 1
                )
                # total_experience excludes the original base copy: the
                # third and sixth physical copies correspond to totals 2/5.
                if total_experience in {2, 5}:
                    return "level-up-discover"
        return None
