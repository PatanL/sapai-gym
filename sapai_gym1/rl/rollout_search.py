"""Public-information counterfactual search over complete shop continuations.

The live policy normally observes only one sampled consequence of each shop
decision.  This module forces plausible root actions, rolls the current policy
forward through later decisions, and compares returns under common battle
seeds.  It deliberately builds a sanitized single-player search state from
the acting player's public view; the opponent's current hidden team and shop
are never copied into the search tree.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from dataclasses import dataclass, field
import time
from typing import Protocol

import numpy as np
import torch

from sapai import Battle, Player, Shop, Team

from .action_schema import ActionType, AutoregressiveAction, FlatActionCodec
from .joint_action import JointAction
from .decision_api import propose_decisions
from .entity_observation import EntityObservationEncoder
from .opponent_belief import OpponentBeliefMix, OpponentBoardBelief
from .opponent_templates import FrozenOpponentTrajectories
from .identity_remap import (
    remap_shop_identities as _remap_identity_set,
    remap_team_orders as _remap_seen_orders,
)


SEARCH_VERSION = "public-rollout-search-v11-joint-20260906"


@dataclass(frozen=True)
class SearchTransition:
    reward: float
    discount: float
    battle_resolved: bool
    battle_result: int | None = None


class SearchState(Protocol):
    battles: int
    terminated: bool

    def clone(self): ...

    def observation(self) -> dict[str, np.ndarray]: ...

    def legal_actions(self) -> list[AutoregressiveAction]: ...

    def step(
        self,
        action: AutoregressiveAction,
        rng: np.random.Generator,
    ) -> SearchTransition: ...


def _copy_team(team):
    return None if team is None else Team.from_state(team.state)


@dataclass
class PublicSearchState:
    """Mutable exact simulator state containing only acting-player information."""

    core: object
    opponent_trajectories: FrozenOpponentTrajectories | tuple[tuple[Team, ...], ...]
    encoder: EntityObservationEncoder = field(
        default_factory=EntityObservationEncoder
    )
    battles: int = 0
    terminated: bool = False
    gamma: float = 0.99
    selected_opponent_trajectory: int | None = None
    opponent_trajectory_ids: tuple[str, ...] = ()
    selected_opponent_trajectory_ids: list[str] = field(default_factory=list)
    opponent_transition_model: object | None = None
    opponent_candidate_limit: int = 32
    opponent_board_belief: OpponentBoardBelief | None = None
    opponent_belief_mix: OpponentBeliefMix | None = None

    def __post_init__(self):
        self.opponent_trajectories = FrozenOpponentTrajectories.freeze(self.opponent_trajectories)

    @classmethod
    def from_core(cls, core, perspective: int, gamma: float = 0.99, *, opening_prior=None):
        """Sanitize a two-player environment into one public player view."""

        if perspective not in (0, 1):
            raise ValueError("perspective must be zero or one")
        from sapai_gym1 import SuperAutoPetsEnv

        own = core.agent if perspective == 0 else core.opponent
        hidden_opponent = core.opponent if perspective == 0 else core.agent
        latest = (
            core.previous_opponent_state
            if perspective == 0
            else core.previous_agent_state
        )
        older = (
            core.previous_previous_opponent_state
            if perspective == 0
            else core.previous_previous_agent_state
        )
        own_previous = (
            core.previous_agent_state
            if perspective == 0
            else core.previous_opponent_state
        )
        own_older = (
            core.previous_previous_agent_state
            if perspective == 0
            else core.previous_previous_opponent_state
        )

        if latest is None:
            if opening_prior is None:
                raise ValueError(
                    "Counterfactual battle search requires an observed opponent board or an explicit opening prior"
                )
            if own.turn != 1:
                raise ValueError("Missing-history priors are supported only on turn one")
            if any(board is not None for board in (older, own_previous, own_older)):
                raise ValueError("An opening state cannot contain older battle history")
            public_trajectories = FrozenOpponentTrajectories.freeze(opening_prior)
            if not any(any(not slot.empty for slot in trajectory[0]) for trajectory in public_trajectories):
                raise ValueError("Opening prior must contain populated opponent boards")
        else:
            if opening_prior is not None:
                raise ValueError("An opening prior cannot replace observed opponent history")
            public_trajectories = ((_copy_team(latest),),)
        # The older board remains encoded as temporal evidence for the neural
        # opponent model. It must not be substituted for the latest known
        # battle board during an immediate counterfactual simulation.

        search_core = SuperAutoPetsEnv(
            opponent_generator=None,
            valid_actions_only=True,
            manual_battles=True,
        )
        search_core.agent = Player.from_state(own.state)
        # The empty placeholder is never the battle opponent: step() samples
        # the explicit prior. Unknown history remains None in the observation.
        public_opponent_team = Team([]) if latest is None else _copy_team(public_trajectories[0][0])
        public_opponent_shop = Shop(
            [],
            turn=own.turn,
            pack=hidden_opponent.pack,
            fixed_rules=True,
        )
        search_core.opponent = Player(
            shop=public_opponent_shop,
            team=public_opponent_team,
            lives=hidden_opponent.lives,
            gold=0,
            turn=own.turn,
            wins=hidden_opponent.wins,
            lf_winner=hidden_opponent.lf_winner,
            pack=hidden_opponent.pack,
            record_actions=False,
        )
        search_core.opponent.turn_ended = True
        search_core.shop_agent = search_core.agent.shop
        search_core.team_agent = search_core.agent.team
        search_core.shop_opponent = search_core.opponent.shop
        search_core.team_opponent = search_core.opponent.team
        search_core.turn = own.turn
        search_core.done = bool(core.done)
        search_core.previous_opponent_state = _copy_team(latest)
        search_core.previous_previous_opponent_state = _copy_team(older)
        search_core.previous_agent_state = _copy_team(own_previous)
        search_core.previous_previous_agent_state = _copy_team(own_older)

        if perspective == 0:
            search_core.shop_action_count_agent = core.shop_action_count_agent
            search_core.reorder_count_agent = core.reorder_count_agent
            toggled = core._freeze_toggled_agent
            seen_orders = core._seen_team_orders_agent
            search_core.allow_additional_reorder_agent = (
                core.allow_additional_reorder_agent
            )
        else:
            search_core.shop_action_count_agent = core.shop_action_count_opponent
            search_core.reorder_count_agent = core.reorder_count_opponent
            toggled = core._freeze_toggled_opponent
            seen_orders = core._seen_team_orders_opponent
            search_core.allow_additional_reorder_agent = (
                core.allow_additional_reorder_opponent
            )
        search_core._freeze_toggled_agent = _remap_identity_set(
            own, search_core.agent, toggled
        )
        search_core._seen_team_orders_agent = _remap_seen_orders(
            own, search_core.agent, seen_orders
        )
        search_core._seen_team_orders_agent.add(
            search_core._team_order_key(search_core.agent)
        )
        search_core._freeze_toggled_opponent.clear()
        search_core._seen_team_orders_opponent = {
            search_core._team_order_key(search_core.opponent)
        }
        search_core._agent_actions_cache = None
        search_core._opponent_actions_cache = None
        search_core.agent_max_team_stats = sum(
            slot.pet.attack + slot.pet.health
            for slot in search_core.agent.team
            if not slot.empty
        )
        search_core.step_reward = 0.0
        return cls(
            search_core, public_trajectories, gamma=float(gamma)
        )

    def clone(self):
        cloned = type(self).from_core(
            self.core, 0, gamma=self.gamma,
            opening_prior=(self.opponent_trajectories if self.core.previous_opponent_state is None else None),
        )
        # Immutable templates are safe to share. Materializing all candidate
        # futures here dominated tree cost despite most never being sampled.
        cloned.opponent_trajectories = self.opponent_trajectories
        cloned.battles = self.battles
        cloned.terminated = self.terminated
        cloned.selected_opponent_trajectory = (
            self.selected_opponent_trajectory
        )
        cloned.opponent_trajectory_ids = tuple(self.opponent_trajectory_ids)
        cloned.selected_opponent_trajectory_ids = list(
            self.selected_opponent_trajectory_ids
        )
        cloned.opponent_transition_model = self.opponent_transition_model
        cloned.opponent_candidate_limit = self.opponent_candidate_limit
        cloned.opponent_board_belief = self.opponent_board_belief
        cloned.opponent_belief_mix = self.opponent_belief_mix
        return cloned

    def set_opponent_trajectories(self, trajectories, trajectory_ids=None):
        normalized = FrozenOpponentTrajectories.freeze(trajectories)
        normalized_ids = tuple(
            map(str, trajectory_ids or range(len(normalized)))
        )
        if len(normalized_ids) != len(normalized):
            raise ValueError("trajectory_ids must match trajectories")
        self.opponent_trajectories = normalized
        self.opponent_trajectory_ids = normalized_ids
        self.selected_opponent_trajectory = None
        self.selected_opponent_trajectory_ids.clear()

    def set_opponent_transition_model(self, reservoir, candidate_limit=32):
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        self.opponent_transition_model = reservoir
        self.opponent_candidate_limit = int(candidate_limit)

    def enable_learned_opponent_belief(self, mixture=None):
        self.opponent_belief_mix = mixture or OpponentBeliefMix()

    def set_opponent_board_belief(self, belief):
        if belief is not None and not isinstance(belief, OpponentBoardBelief):
            raise TypeError("belief must be an OpponentBoardBelief")
        self.opponent_board_belief = belief

    def redeterminize(self, seed):
        """Replace hidden future randomness without changing visible state."""

        random_state = np.random.RandomState(int(seed) % (2**32))
        self.core.agent.shop.rs = random_state
        self.core.agent.shop.seed_state = random_state.get_state()
        for slot in self.core.agent.shop:
            slot.rs = random_state
            slot.seed_state = random_state.get_state()
        self.core.agent._connect_objects()
        self.core.rs_random = np.random.RandomState(
            (int(seed) ^ 0x9E3779B9) % (2**32)
        )

    def observation(self):
        return self.encoder.encode(self.core, 0)

    def legal_actions(self):
        return [
            FlatActionCodec.decode(flat)
            for flat in self.core._avail_actions(0)
        ]

    def _belief_opponent_team(self, rng, adaptive_matches):
        if self.opponent_belief_mix is None or self.opponent_board_belief is None:
            return None
        if adaptive_matches:
            reservoir_teams = [match.teams[0] for match in adaptive_matches]
            reservoir_ids = [match.trajectory_id for match in adaptive_matches]
        else:
            reservoir_teams = [
                trajectory[min(self.battles, len(trajectory) - 1)]
                for trajectory in self.opponent_trajectories
            ]
            reservoir_ids = list(self.opponent_trajectory_ids)
        mixture = self.opponent_belief_mix
        categories = []
        weights = []
        if mixture.synthetic_weight > 0:
            categories.append("synthetic")
            weights.append(mixture.synthetic_weight)
        if (
            mixture.persistence_weight > 0
            and self.core.previous_opponent_state is not None
        ):
            categories.append("persistence")
            weights.append(mixture.persistence_weight)
        if mixture.reservoir_weight > 0 and reservoir_teams:
            categories.append("reservoir")
            weights.append(mixture.reservoir_weight)
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        category = categories[int(rng.choice(len(categories), p=weights))]
        if category == "synthetic":
            self.selected_opponent_trajectory_ids.append("belief-synthetic")
            return self.opponent_board_belief.modal_team()
        if category == "persistence":
            self.selected_opponent_trajectory_ids.append("belief-persistence")
            return _copy_team(self.core.previous_opponent_state)
        probabilities = self.opponent_board_belief.reservoir_probabilities(
            reservoir_teams,
            temperature=mixture.reservoir_temperature,
            uniform_mix=mixture.reservoir_uniform_mix,
        )
        index = int(rng.choice(len(reservoir_teams), p=probabilities))
        if reservoir_ids:
            self.selected_opponent_trajectory_ids.append(
                reservoir_ids[index]
            )
        if not adaptive_matches:
            self.selected_opponent_trajectory = index
        return _copy_team(reservoir_teams[index])

    def step(self, action, rng):
        from .joint_action import JointAction
        if isinstance(action, JointAction):
            # This is the PUBLIC clone, not the real two-player environment.
            # Share only action expansion; never query hidden opponent state.
            plan = action.expand(self.core, 0)
            for primitive in plan[:-1]:
                self.core.resolve_action(0, FlatActionCodec.encode(primitive))
            action = plan[-1]
        flat = FlatActionCodec.encode(action)
        if flat not in self.core._avail_actions(0):
            raise ValueError(f"Search attempted illegal action {action}")
        self.core.resolve_action(0, flat)
        if ActionType(action.action_type) is not ActionType.END_TURN:
            return SearchTransition(0.0, 1.0, False)

        adaptive_matches = []
        if self.battles > 0 and self.opponent_transition_model is not None:
            adaptive_matches = self.opponent_transition_model.match(
                self.core.previous_opponent_state,
                observed_enemy=self.core.previous_agent_state,
                older_observed=(
                    self.core.previous_previous_opponent_state
                ),
                older_observed_enemy=(
                    self.core.previous_previous_agent_state
                ),
                current_turn=self.core.agent.turn,
                horizon=1,
                limit=self.opponent_candidate_limit,
            )
        opponent_team = self._belief_opponent_team(rng, adaptive_matches)
        if opponent_team is None:
            if adaptive_matches:
                match = adaptive_matches[
                    int(rng.integers(0, len(adaptive_matches)))
                ]
                opponent_team = _copy_team(match.teams[0])
                self.selected_opponent_trajectory_ids.append(
                    match.trajectory_id
                )
            else:
                if self.selected_opponent_trajectory is None:
                    self.selected_opponent_trajectory = int(
                        rng.integers(0, len(self.opponent_trajectories))
                    )
                trajectory = self.opponent_trajectories[
                    self.selected_opponent_trajectory
                ]
                opponent_team = _copy_team(
                    trajectory[min(self.battles, len(trajectory) - 1)]
                )
                if self.opponent_trajectory_ids:
                    self.selected_opponent_trajectory_ids.append(
                        self.opponent_trajectory_ids[
                            self.selected_opponent_trajectory
                        ]
                    )
        self.core.opponent.team = opponent_team
        self.core.team_opponent = opponent_team
        self.core.previous_previous_opponent_state = (
            self.core.previous_opponent_state
        )
        self.core.previous_opponent_state = _copy_team(opponent_team)
        self.core.previous_previous_agent_state = self.core.previous_agent_state
        self.core.previous_agent_state = _copy_team(self.core.agent.team)

        battle_result = Battle(
            self.core.agent.team,
            opponent_team,
            seed=int(rng.integers(0, 2**32 - 1)),
            fast_training=True,
        ).battle()
        lives_before = (self.core.agent.lives, self.core.opponent.lives)
        self.core._player_fight_outcome(battle_result)
        life_deltas = (
            self.core.agent.lives - lives_before[0],
            self.core.opponent.lives - lives_before[1],
        )
        self.core.turn += 1
        self.battles += 1
        self.terminated = bool(self.core.is_done())

        reward = 0.1 * (life_deltas[0] - life_deltas[1])
        if self.core.opponent.lives <= 0:
            reward += 2.0
        if self.core.agent.lives <= 0:
            reward -= 2.0
        if (
            self.terminated
            and self.core.agent.lives > 0
            and self.core.opponent.lives > 0
        ):
            own_score = (self.core.agent.lives, self.core.agent.wins)
            opponent_score = (
                self.core.opponent.lives,
                self.core.opponent.wins,
            )
            if own_score > opponent_score:
                reward += 2.0
            elif opponent_score > own_score:
                reward -= 2.0

        if not self.terminated:
            self.core.agent.start_turn(winner=self.core.agent.lf_winner)
            self.core.opponent.turn = self.core.agent.turn
            self.core.opponent.turn_ended = True
            self.core.reorder_count_agent = 0
            self.core.allow_additional_reorder_agent = False
            self.core.shop_action_count_agent = 0
            self.core._freeze_toggled_agent.clear()
            self.core._seen_team_orders_agent = {
                self.core._team_order_key(self.core.agent)
            }
            self.core._agent_actions_cache = None
            self.opponent_board_belief = None

        return SearchTransition(
            float(reward),
            0.0 if self.terminated else self.gamma,
            True,
            battle_result,
        )


@dataclass(frozen=True)
class PolicyEvaluation:
    priors: dict[AutoregressiveAction, float]
    value: float
    next_memory: torch.Tensor
    opponent_belief: OpponentBoardBelief | None = None
    omitted_probability_mass: float = 0.0
    exact_log_probabilities: dict | None = None
    policy_action: object | None = None


class NeuralSearchEvaluator:
    """Evaluate exact legal-action probabilities under one recurrent policy."""

    def __init__(
        self, policy, device="cpu", chunk_size=64, value_calibrator=None,
        joint_mask_budget=8, joint_exploration_masks=2
    ):
        self.policy = policy
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.value_calibrator = value_calibrator
        self.joint_mask_budget = joint_mask_budget
        self.joint_exploration_masks = joint_exploration_masks
        self.reset_metrics()

    def _value(self, value, state):
        value = float(value)
        if self.value_calibrator is not None:
            value = self.value_calibrator.correct(value, state)
        return value

    def reset_metrics(self):
        self.evaluated_states = 0
        self.forced_actions = 0
        self.continuation_states = 0
        self.prior_forward_states = 0
        self.omitted_mask_mass = []

    def metrics(self):
        return {
            "evaluatedStates": self.evaluated_states,
            "forcedActions": self.forced_actions,
            "continuationStates": self.continuation_states,
            "priorForwardStates": self.prior_forward_states,
            "maxOmittedEndMaskMass": max(self.omitted_mask_mass, default=0.),
        }

    def _memory(self, memory):
        memory = torch.as_tensor(memory, dtype=torch.float32, device=self.device)
        return memory.unsqueeze(0) if memory.ndim == 1 else memory

    @staticmethod
    def _attach_opponent_belief(state, output, index):
        if (
            getattr(state, "opponent_belief_mix", None) is None
            or not hasattr(state, "set_opponent_board_belief")
        ):
            return None
        belief = OpponentBoardBelief.from_logits(
            output.opponent_pet_logits[index].float().cpu().numpy(),
            output.opponent_status_logits[index].float().cpu().numpy(),
            output.opponent_level_logits[index].float().cpu().numpy(),
            output.opponent_stats[index].float().cpu().numpy(),
            output.opponent_temporary_logits[index].float().cpu().numpy(),
            state.encoder,
        )
        state.set_opponent_board_belief(belief)
        return belief

    @torch.inference_mode()
    def evaluate(self, state: SearchState, memory) -> PolicyEvaluation:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("Search state has no legal actions")
        self.evaluated_states += 1
        self.forced_actions += len(actions)
        observation = state.observation()
        base_memory = self._memory(memory)
        if getattr(getattr(self.policy, 'config', None), 'end_freeze', False):
            batch = {key: torch.as_tensor(value, device=self.device)[None] for key,value in observation.items()}
            features = self.policy.encode_decision(batch, base_memory, forecast=False)
            digest = hashlib.sha256()
            for key in sorted(observation):
                digest.update(np.ascontiguousarray(observation[key]).tobytes())
            digest.update(base_memory.detach().cpu().numpy().tobytes())
            generator = torch.Generator().manual_seed(int.from_bytes(digest.digest()[:8], 'little') % (2**63-1))
            proposals = propose_decisions(self.policy, features, mask_budget=self.joint_mask_budget,
                exploration_masks=self.joint_exploration_masks, generator=generator)
            probabilities = np.exp(np.asarray(proposals.log_probabilities, dtype=np.float64))
            probabilities /= probabilities.sum()
            priors = dict(zip(proposals.decisions, map(float, probabilities)))
            belief = None
            if getattr(state, 'opponent_belief_mix', None) is not None:
                from types import SimpleNamespace
                belief = self._attach_opponent_belief(state, SimpleNamespace(**self.policy.forecast_opponent(features)), 0)
            self.prior_forward_states += 1
            self.omitted_mask_mass.append(proposals.omitted_probability_mass)
            actor = self.policy.sample_joint_action(features, deterministic=True)
            raw_action = JointAction(AutoregressiveAction(*actor.actions[0].cpu().tolist()), int(actor.end_freeze_mask[0].cpu()))
            return PolicyEvaluation(priors, self._value(self.policy.value_from_encoded(features)[0].float().cpu(), state),
                features.memory[0].float().cpu(), belief, proposals.omitted_probability_mass,
                dict(zip(proposals.decisions, proposals.log_probabilities)), raw_action)
        if hasattr(self.policy, "evaluate_all_actions"):
            batch = {
                key: torch.as_tensor(value_array, device=self.device)
                .unsqueeze(0)
                for key, value_array in observation.items()
            }
            output = self.policy.evaluate_all_actions(
                batch, memory=base_memory
            )
            self.prior_forward_states += 1
            joint = output.joint_log_prob[0]
            probabilities = [
                float(
                    joint[
                        int(action.action_type),
                        int(action.source),
                        int(action.target),
                    ]
                    .float()
                    .exp()
                    .cpu()
                )
                for action in actions
            ]
            total = max(
                float(sum(probabilities)), np.finfo(np.float64).tiny
            )
            priors = {
                action: float(probability / total)
                for action, probability in zip(actions, probabilities)
            }
            belief = self._attach_opponent_belief(state, output, 0)
            return PolicyEvaluation(
                priors,
                self._value(output.value[0].float().cpu(), state),
                output.memory[0].float().cpu(),
                belief,
            )
        probabilities = []
        value = None
        next_memory = None
        belief = None
        for start in range(0, len(actions), self.chunk_size):
            chunk = actions[start : start + self.chunk_size]
            repeats = len(chunk)
            batch = {
                key: torch.as_tensor(value_array, device=self.device)
                .unsqueeze(0)
                .expand(repeats, *value_array.shape)
                for key, value_array in observation.items()
            }
            action_tensor = torch.tensor(
                [action.as_tuple() for action in chunk],
                dtype=torch.long,
                device=self.device,
            )
            output = self.policy(
                batch,
                memory=base_memory.expand(repeats, -1),
                actions=action_tensor,
            )
            probabilities.extend(output.log_prob.float().exp().cpu().tolist())
            if value is None:
                value = float(output.value[0].float().cpu())
                next_memory = output.memory[0].float().cpu()
                belief = self._attach_opponent_belief(state, output, 0)
        total = max(float(sum(probabilities)), np.finfo(np.float64).tiny)
        priors = {
            action: float(probability / total)
            for action, probability in zip(actions, probabilities)
        }
        return PolicyEvaluation(
            priors, self._value(value, state), next_memory, belief
        )

    @torch.inference_mode()
    def choose(self, state: SearchState, memory, deterministic=True):
        return self.choose_many([state], [memory], deterministic)[0]

    @torch.inference_mode()
    def choose_many(self, states, memories, deterministic=True):
        self.continuation_states += len(states)
        results = []
        for start in range(0, len(states), self.chunk_size):
            state_chunk = states[start : start + self.chunk_size]
            memory_chunk = memories[start : start + self.chunk_size]
            observations = [state.observation() for state in state_chunk]
            batch = {
                key: torch.as_tensor(
                    np.stack([observation[key] for observation in observations]),
                    device=self.device,
                )
                for key in observations[0]
            }
            memory_batch = torch.stack(
                [self._memory(memory).squeeze(0) for memory in memory_chunk]
            )
            joint = getattr(getattr(self.policy, 'config', None), 'end_freeze', False)
            if joint:
                from types import SimpleNamespace
                features = self.policy.encode_decision(batch, memory_batch, forecast=False)
                actor = self.policy.sample_joint_action(features, deterministic=deterministic)
                prediction = (self.policy.forecast_opponent(features)
                              if any(getattr(state,'opponent_belief_mix',None) is not None for state in state_chunk) else {})
                output = SimpleNamespace(actions=actor.actions, memory=actor.memory, end_freeze_mask=actor.end_freeze_mask,
                                         value=self.policy.value_from_encoded(features), **prediction)
            else:
                output = self.policy(batch, memory=memory_batch, deterministic=deterministic)
            for index, state in enumerate(state_chunk):
                self._attach_opponent_belief(state, output, index)
            results.extend(
                (
                    (JointAction(AutoregressiveAction(*map(int, output.actions[index].cpu())), int(output.end_freeze_mask[index].cpu()))
                     if joint else AutoregressiveAction(*map(int, output.actions[index].cpu()))),
                    output.memory[index].float().cpu(),
                    self._value(
                        output.value[index].float().cpu(), state_chunk[index]
                    ),
                )
                for index in range(len(state_chunk))
            )
        return results

    @torch.inference_mode()
    def value(self, state: SearchState, memory):
        return self.choose(state, memory, deterministic=True)[2]

    def values_many(self, states, memories):
        return [
            result[2]
            for result in self.choose_many(states, memories, deterministic=True)
        ]


DEFAULT_ACTION_QUOTAS = {
    ActionType.END_TURN: 1,
    ActionType.ROLL: 1,
    ActionType.BUY_PET: 35,
    ActionType.BUY_FOOD: 35,
    ActionType.SELL: 5,
    ActionType.COMBINE: 20,
    ActionType.FREEZE: 10,
    ActionType.UNFREEZE: 10,
    ActionType.MOVE: 20,
}


@dataclass(frozen=True)
class RolloutSearchConfig:
    rollouts_per_action: int = 4
    confirmation_rollouts: int = 16
    confirmation_top_k: int = 4
    validation_rollouts: int = 32
    audit_rollouts: int = 32
    horizon_battles: int = 5
    max_continuation_steps: int = 60
    action_quotas: dict[ActionType, int] = field(
        default_factory=lambda: dict(DEFAULT_ACTION_QUOTAS)
    )
    minimum_improvement: float = 0.02
    confidence_z: float = 1.96
    exhaustive_root_probability: float = 0.10
    leaf_value_weight: float = 1.0
    max_nested_searches: int = 0
    nested_rollouts_per_action: int = 2
    nested_confirmation_rollouts: int = 4
    nested_confirmation_top_k: int = 3
    nested_lookahead_battles: int = 2
    # Bound the number of full simulator clones retained at once. Screening
    # can contain hundreds of action/seed pairs; keeping all of their Python
    # object graphs resident simultaneously can consume many GiB without
    # improving the statistical estimator.
    max_parallel_rollouts: int = 64
    # Optional production-teacher optimization. Scientific planner-depth
    # comparisons leave this disabled so expert strategy does not become a
    # hidden component of the planner being measured.
    prune_affordable_freezes: bool = False


@dataclass(frozen=True)
class RolloutSearchResult:
    best_action: AutoregressiveAction
    policy_action: AutoregressiveAction
    scores: dict[AutoregressiveAction, float]
    priors: dict[AutoregressiveAction, float]
    standard_errors: dict[AutoregressiveAction, float]
    improvement: float
    improvement_standard_error: float
    improvement_lower_bound: float
    accepted: bool
    rollouts: dict[AutoregressiveAction, tuple["RolloutTrace", ...]]
    screening_scores: dict[AutoregressiveAction, float]
    screening_standard_errors: dict[AutoregressiveAction, float]
    screening_best_action: AutoregressiveAction
    exhaustive_root: bool
    full_candidate_count: int
    pruned_candidate_count: int
    screening_best_excluded_by_pruning: bool | None
    pruned_action_counts: dict[str, int]
    confirmed_best_excluded_by_pruning: bool | None
    validated_pruned_winner: bool | None
    policy_first_battle_score: float
    best_first_battle_score: float
    first_battle_score_improvement: float
    policy_horizon_battle_score: float
    best_horizon_battle_score: float
    horizon_battle_score_improvement: float
    validation_return_improvement: float
    validation_return_standard_error: float
    validation_return_lower_bound: float
    validation_first_battle_score_improvement: float
    validation_horizon_battle_score_improvement: float
    validation_differences: tuple[float, ...]
    audit_return_improvement: float | None
    audit_return_standard_error: float | None
    audit_return_lower_bound: float | None
    audit_first_battle_score_improvement: float | None
    audit_horizon_battle_score_improvement: float | None
    selection_simulator_transitions: int
    total_simulator_transitions: int
    selection_wall_seconds: float
    validation_wall_seconds: float
    audit_wall_seconds: float
    omitted_end_mask_probability_mass: float = 0.0


@dataclass(frozen=True)
class RolloutTrace:
    actions: tuple[AutoregressiveAction, ...]
    rewards: tuple[float, ...]
    battle_results: tuple[int, ...]
    leaf_value: float | None
    total_return: float
    terminated: bool
    battles: int
    opponent_trajectory_index: int | None
    nested_searches: int
    opponent_trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class StateValueRollouts:
    returns: tuple[float, ...]
    mean: float
    standard_error: float


class CounterfactualRolloutSearch:
    """Common-random-number rollout policy improvement at one shop state."""

    def __init__(self, evaluator, config=None):
        self.evaluator = evaluator
        self.config = config or RolloutSearchConfig()
        if self.config.rollouts_per_action <= 0:
            raise ValueError("rollouts_per_action must be positive")
        if self.config.confirmation_rollouts < 2:
            raise ValueError("confirmation_rollouts must be at least two")
        if self.config.confirmation_top_k <= 0:
            raise ValueError("confirmation_top_k must be positive")
        if self.config.validation_rollouts < 2:
            raise ValueError("validation_rollouts must be at least two")
        if (
            self.config.audit_rollouts != 0
            and self.config.audit_rollouts < 2
        ):
            raise ValueError("audit_rollouts must be zero or at least two")
        if self.config.horizon_battles <= 0:
            raise ValueError("horizon_battles must be positive")
        if self.config.max_continuation_steps <= 0:
            raise ValueError("max_continuation_steps must be positive")
        if self.config.confidence_z < 0:
            raise ValueError("confidence_z must be non-negative")
        if not 0.0 <= self.config.exhaustive_root_probability <= 1.0:
            raise ValueError("exhaustive_root_probability must be in [0, 1]")
        if not 0.0 <= self.config.leaf_value_weight <= 1.0:
            raise ValueError("leaf_value_weight must be in [0, 1]")
        if self.config.max_nested_searches < 0:
            raise ValueError("max_nested_searches must be non-negative")
        if self.config.nested_rollouts_per_action <= 0:
            raise ValueError("nested_rollouts_per_action must be positive")
        if self.config.nested_confirmation_rollouts < 2:
            raise ValueError(
                "nested_confirmation_rollouts must be at least two"
            )
        if self.config.nested_confirmation_top_k <= 0:
            raise ValueError("nested_confirmation_top_k must be positive")
        if self.config.nested_lookahead_battles <= 0:
            raise ValueError("nested_lookahead_battles must be positive")
        if self.config.max_parallel_rollouts <= 0:
            raise ValueError("max_parallel_rollouts must be positive")

    @staticmethod
    def _root_moves_are_strategic(state):
        core = getattr(state, "core", None)
        if core is None:
            return True
        player = core.agent
        if player.gold == 0:
            return True
        return any(
            not slot.empty
            and slot.obj.name == "food-sleeping-pill"
            and slot.cost <= player.gold
            for slot in player.shop
        )

    @staticmethod
    def _root_freeze_is_strategic(state, action):
        core = getattr(state, "core", None)
        if core is None:
            return True
        player = core.agent
        source = int(action.source)
        if source < 0 or source >= len(player.shop):
            return False
        slot = player.shop[source]
        if slot.empty or slot.cost > player.gold:
            return True
        if slot.slot_type not in {"pet", "levelup"}:
            return False
        matching_pets = [
            team_slot.pet
            for team_slot in player.team
            if not team_slot.empty
            and team_slot.pet.name == slot.obj.name
        ]
        levelup_deferral = False
        for pet in matching_pets:
            combined_total = (
                pet.total_experience + slot.obj.total_experience + 1
            )
            earns_reward = combined_total == 2 or (
                combined_total == 5
                and not (pet.level == 2 and slot.obj.level == 2)
            )
            levelup_deferral |= earns_reward and player.turn % 2 == 0
        team_size = len(player.team)
        salad_isolation = player.turn in {5, 6} and team_size <= 3
        late_full_board_hold = (
            player.turn >= 7
            and team_size == 5
            and not matching_pets
        )
        # Early affordable freezes are usually dominated by buying now. Keep
        # only explicit pacing/geometry exceptions: odd-turn tier rewards,
        # turn-5/6 Salad isolation, and later full-board freeze-roll plans such
        # as holding Cow until a Turtle can be Pilled.
        return levelup_deferral or salad_isolation or late_full_board_hold

    def _candidate_actions(self, priors, state=None):
        grouped = defaultdict(list)
        for action, prior in priors.items():
            if (
                ActionType(action.action_type) is ActionType.MOVE
                and state is not None
                and not self._root_moves_are_strategic(state)
            ):
                continue
            if (
                ActionType(action.action_type) is ActionType.FREEZE
                and state is not None
                and self.config.prune_affordable_freezes
                and not self._root_freeze_is_strategic(state, action)
            ):
                continue
            grouped[ActionType(action.action_type)].append((prior, action))
        candidates = []
        for action_type, entries in grouped.items():
            entries.sort(key=lambda item: item[0], reverse=True)
            quota = self.config.action_quotas.get(action_type, len(entries))
            if action_type == ActionType.END_TURN and any(isinstance(action, JointAction) for _, action in entries) and quota:
                # Proposal budgeting already bounded this conditional branch.
                quota = len(entries)
            candidates.extend(action for _, action in entries[:quota])
        return candidates

    def _choose_many(self, records, deterministic=True):
        if hasattr(self.evaluator, "choose_many"):
            return self.evaluator.choose_many(
                [record["state"] for record in records],
                [record["memory"] for record in records],
                deterministic=deterministic,
            )
        return [
            self.evaluator.choose(
                record["state"],
                record["memory"],
                deterministic=deterministic,
            )
            for record in records
        ]

    def _values_many(self, records):
        if hasattr(self.evaluator, "values_many"):
            return self.evaluator.values_many(
                [record["state"] for record in records],
                [record["memory"] for record in records],
            )
        return [
            self.evaluator.value(record["state"], record["memory"])
            for record in records
        ]

    def _rollouts(
        self,
        initial,
        candidates,
        next_memory,
        rollout_seeds,
        *,
        allow_nested=False,
        retain_trace=True,
    ):
        target_battles = initial.battles + self.config.horizon_battles
        completed_records = []
        rollout_specs = [
            (action, rollout_seed)
            for action in candidates
            for rollout_seed in rollout_seeds
        ]
        for batch_start in range(
            0, len(rollout_specs), self.config.max_parallel_rollouts
        ):
            records = []
            batch_specs = rollout_specs[
                batch_start : batch_start
                + self.config.max_parallel_rollouts
            ]
            for action, rollout_seed in batch_specs:
                state = initial.clone()
                rng = np.random.default_rng(int(rollout_seed))
                if hasattr(state, "redeterminize"):
                    state.redeterminize(
                        int(rng.integers(0, 2**32 - 1))
                    )
                transition = state.step(action, rng)
                self.simulator_transitions += 1
                record = {
                        "root_action": action,
                        "state": state,
                        "rng": rng,
                        "memory": next_memory.clone(),
                        "total": float(transition.reward),
                        "discount": float(transition.discount),
                        "steps": 0,
                        "target_battles": target_battles,
                        "battle_results": (
                            []
                            if transition.battle_result is None
                            else [int(transition.battle_result)]
                        ),
                        "leaf_value": None,
                        "nested_enabled": bool(allow_nested),
                        "nested_searches": 0,
                        "last_action": action,
                    }
                if retain_trace:
                    record["actions"] = [action]
                    record["rewards"] = [float(transition.reward)]
                records.append(record)

            while True:
                active = [
                    record
                    for record in records
                    if not record["state"].terminated
                    and record["state"].battles
                    < target_battles
                    and record["steps"]
                    < self.config.max_continuation_steps
                ]
                if not active:
                    break
                choices = self._choose_many(active)
                for record, (action, next_memory, _) in zip(active, choices):
                    transition = record["state"].step(
                        action, record["rng"]
                    )
                    self.simulator_transitions += 1
                    record["total"] += (
                        record["discount"] * transition.reward
                    )
                    record["discount"] *= transition.discount
                    record["memory"] = next_memory
                    record["steps"] += 1
                    record["last_action"] = action
                    if retain_trace:
                        record["actions"].append(action)
                        record["rewards"].append(
                            float(transition.reward)
                        )
                    if transition.battle_result is not None:
                        record["battle_results"].append(
                            int(transition.battle_result)
                        )

            bootstrap = [
                record
                for record in records
                if not record["state"].terminated
            ]
            if bootstrap:
                for record, value in zip(
                    bootstrap, self._values_many(bootstrap)
                ):
                    record["total"] += (
                        record["discount"]
                        * self.config.leaf_value_weight
                        * value
                    )
                    record["leaf_value"] = float(value)

            # Preserve only the small audit fields once a batch is complete;
            # the simulator clone, RNG, and recurrent tensors are no longer
            # needed by the caller.
            for record in records:
                state = record.pop("state")
                record["terminated"] = bool(state.terminated)
                record["battles"] = int(state.battles)
                record["opponent_trajectory_index"] = getattr(
                    state, "selected_opponent_trajectory", None
                )
                record["opponent_trajectory_ids"] = tuple(
                    getattr(state, "selected_opponent_trajectory_ids", ())
                )
                for key in (
                    "rng",
                    "memory",
                    "discount",
                    "steps",
                    "nested_enabled",
                    "last_action",
                    "target_battles",
                ):
                    record.pop(key, None)
                completed_records.append(record)
        return completed_records

    def rollout_state_value(
        self,
        state,
        memory,
        *,
        rollouts,
        horizon_battles,
        seed,
        deterministic=False,
    ):
        """Estimate a state value with long continuations and no leaf bootstrap."""

        if rollouts < 2:
            raise ValueError("rollouts must be at least two")
        if horizon_battles <= 0:
            raise ValueError("horizon_battles must be positive")
        target_battles = state.battles + horizon_battles
        rng = np.random.default_rng(seed)
        records = []
        for rollout_seed in rng.integers(0, 2**63 - 1, size=rollouts):
            cloned_state = state.clone()
            rollout_rng = np.random.default_rng(int(rollout_seed))
            if hasattr(cloned_state, "redeterminize"):
                cloned_state.redeterminize(
                    int(rollout_rng.integers(0, 2**32 - 1))
                )
            records.append(
                {
                "state": cloned_state,
                "rng": rollout_rng,
                "memory": torch.as_tensor(memory).float().clone(),
                "total": 0.0,
                "discount": 1.0,
                "steps": 0,
                }
            )
        while True:
            active = [
                record
                for record in records
                if not record["state"].terminated
                and record["state"].battles < target_battles
                and record["steps"] < self.config.max_continuation_steps
            ]
            if not active:
                break
            choices = self._choose_many(
                active, deterministic=deterministic
            )
            for record, (action, next_memory, _) in zip(active, choices):
                transition = record["state"].step(action, record["rng"])
                record["total"] += record["discount"] * transition.reward
                record["discount"] *= transition.discount
                record["memory"] = next_memory
                record["steps"] += 1
        returns = np.asarray(
            [record["total"] for record in records], dtype=np.float64
        )
        return StateValueRollouts(
            returns=tuple(map(float, returns)),
            mean=float(returns.mean()),
            standard_error=float(
                returns.std(ddof=1) / np.sqrt(len(returns))
            ),
        )

    def search(
        self, state: SearchState, memory, seed=0, validation_seed=None
    ):
        selection_started = time.perf_counter()
        self.simulator_transitions = 0
        root = self.evaluator.evaluate(state, memory)
        pruned_candidates = self._candidate_actions(root.priors, state)
        if not pruned_candidates:
            raise ValueError("No candidate actions survived search quotas")
        # The deployed decoder is greedy at each autoregressive stage. That
        # action need not maximize the joint probability over all complete
        # triples, so query the exact deployed decision rather than using
        # ``max(priors)`` as a proxy.
        if root.policy_action is not None:
            policy_action = root.policy_action
        else:
            policy_action, _, _ = self.evaluator.choose(state, memory, deterministic=True)
        rng = np.random.default_rng(seed)
        validation_rng = (
            rng
            if validation_seed is None
            else np.random.default_rng(validation_seed)
        )
        exhaustive_root = (
            rng.random() < self.config.exhaustive_root_probability
        )
        candidates = (
            list(root.priors)
            if exhaustive_root
            else list(pruned_candidates)
        )
        if policy_action not in candidates:
            candidates.append(policy_action)
        screening_seeds = rng.integers(
            0, 2**63 - 1, size=self.config.rollouts_per_action
        )
        screening_records = self._rollouts(
            state,
            candidates,
            root.next_memory,
            screening_seeds,
            allow_nested=False,
            retain_trace=False,
        )
        screening_samples = {
            action: np.asarray(
                [
                    record["total"]
                    for record in screening_records
                    if record["root_action"] == action
                ],
                dtype=np.float64,
            )
            for action in candidates
        }
        screening_scores = {
            action: float(values.mean())
            for action, values in screening_samples.items()
        }
        screening_standard_errors = {
            action: (
                float("inf")
                if len(values) < 2
                else float(values.std(ddof=1) / np.sqrt(len(values)))
            )
            for action, values in screening_samples.items()
        }
        screening_ranked = sorted(
            screening_scores,
            key=lambda action: (
                screening_scores[action], root.priors.get(action, 0.0)
            ),
            reverse=True,
        )
        screening_best_action = screening_ranked[0]
        confirmation_candidates = screening_ranked[
            : self.config.confirmation_top_k
        ]
        if policy_action not in confirmation_candidates:
            confirmation_candidates.append(policy_action)
        confirmation_seeds = rng.integers(
            0, 2**63 - 1, size=self.config.confirmation_rollouts
        )
        records = self._rollouts(
            state,
            confirmation_candidates,
            root.next_memory,
            confirmation_seeds,
            allow_nested=self.config.max_nested_searches > 0,
        )
        samples = {
            action: np.asarray(
                [
                    record["total"]
                    for record in records
                    if record["root_action"] == action
                ],
                dtype=np.float64,
            )
            for action in confirmation_candidates
        }
        scores = {
            action: float(values.mean()) for action, values in samples.items()
        }
        standard_errors = {
            action: float(values.std(ddof=1) / np.sqrt(len(values)))
            for action, values in samples.items()
        }
        best_score = max(scores.values())
        tied = [
            action
            for action, score in scores.items()
            if np.isclose(score, best_score, atol=1e-9, rtol=0.0)
        ]
        best_action = max(
            tied, key=lambda action: root.priors.get(action, 0.0)
        )

        def battle_score(record, first_only=False):
            results = record["battle_results"]
            if not results:
                return 0.5
            if first_only:
                results = results[:1]
            return float(
                np.mean(
                    [
                        1.0 if result == 0 else 0.5 if result == 2 else 0.0
                        for result in results
                    ]
                )
            )

        def action_battle_scores(source_records, action, first_only=False):
            return np.asarray(
                [
                    battle_score(record, first_only=first_only)
                    for record in source_records
                    if record["root_action"] == action
                ],
                dtype=np.float64,
            )

        policy_first_scores = action_battle_scores(
            records, policy_action, first_only=True
        )
        best_first_scores = action_battle_scores(
            records, best_action, first_only=True
        )
        policy_horizon_scores = action_battle_scores(records, policy_action)
        best_horizon_scores = action_battle_scores(records, best_action)
        paired_improvements = samples[best_action] - samples[policy_action]
        improvement = float(paired_improvements.mean())
        improvement_standard_error = (
            float("inf")
            if len(paired_improvements) < 2
            else float(
                paired_improvements.std(ddof=1)
                / np.sqrt(len(paired_improvements))
            )
        )
        improvement_lower_bound = (
            float("-inf")
            if not np.isfinite(improvement_standard_error)
            else improvement
            - self.config.confidence_z * improvement_standard_error
        )
        selection_simulator_transitions = self.simulator_transitions
        selection_wall_seconds = time.perf_counter() - selection_started

        # The confirmation scores above select one fixed finalist. A third,
        # disjoint seed set then confirms that fixed winner against the base
        # action. This stage gates labels but never changes which action won,
        # eliminating the remaining winner's-curse bias from selecting and
        # accepting on the same samples.
        validation_actions = list(dict.fromkeys([best_action, policy_action]))
        validation_seeds = validation_rng.integers(
            0, 2**63 - 1, size=self.config.validation_rollouts
        )
        validation_started = time.perf_counter()
        validation_records = self._rollouts(
            state,
            validation_actions,
            root.next_memory,
            validation_seeds,
            allow_nested=self.config.max_nested_searches > 0,
            retain_trace=False,
        )
        validation_samples = {
            action: np.asarray(
                [
                    record["total"]
                    for record in validation_records
                    if record["root_action"] == action
                ],
                dtype=np.float64,
            )
            for action in validation_actions
        }
        validation_differences = (
            validation_samples[best_action]
            - validation_samples[policy_action]
        )
        validation_return_improvement = float(
            validation_differences.mean()
        )
        validation_return_standard_error = float(
            validation_differences.std(ddof=1)
            / np.sqrt(len(validation_differences))
        )
        validation_return_lower_bound = (
            validation_return_improvement
            - self.config.confidence_z
            * validation_return_standard_error
        )
        validation_wall_seconds = time.perf_counter() - validation_started
        validation_policy_first = action_battle_scores(
            validation_records, policy_action, first_only=True
        )
        validation_best_first = action_battle_scores(
            validation_records, best_action, first_only=True
        )
        validation_policy_horizon = action_battle_scores(
            validation_records, policy_action
        )
        validation_best_horizon = action_battle_scores(
            validation_records, best_action
        )
        accepted = (
            best_action != policy_action
            and improvement >= self.config.minimum_improvement
            and validation_return_lower_bound
            >= self.config.minimum_improvement
        )
        # When gold remains and no Pill geometry is active, an ordinary MOVE
        # and an economic click are generally two orderings of the same plan.
        # Do not distill a preference between them; after the harmless move,
        # search can reconsider the resulting equivalent state.
        order_only_move = (
            not self._root_moves_are_strategic(state)
            and (
                ActionType(best_action.action_type) is ActionType.MOVE
                or ActionType(policy_action.action_type) is ActionType.MOVE
            )
        )
        accepted &= not order_only_move
        pruned_set = set(pruned_candidates)
        pruned_action_counts = {}
        if exhaustive_root:
            for action in root.priors:
                if action in pruned_set:
                    continue
                name = ActionType(action.action_type).name
                pruned_action_counts[name] = (
                    pruned_action_counts.get(name, 0) + 1
                )
        confirmed_best_excluded = (
            best_action not in pruned_set if exhaustive_root else None
        )
        validated_pruned_winner = (
            bool(accepted and confirmed_best_excluded)
            if exhaustive_root
            else None
        )

        # Only accepted labels pay for a fourth, completely untouched audit.
        # These samples affect neither winner selection nor acceptance and are
        # therefore an unbiased held-out estimate of S(B) > B.
        audit_return_improvement = None
        audit_return_standard_error = None
        audit_return_lower_bound = None
        audit_first_battle_score_improvement = None
        audit_horizon_battle_score_improvement = None
        audit_started = time.perf_counter()
        if accepted and self.config.audit_rollouts:
            audit_seeds = validation_rng.integers(
                0, 2**63 - 1, size=self.config.audit_rollouts
            )
            audit_records = self._rollouts(
                state,
                validation_actions,
                root.next_memory,
                audit_seeds,
                allow_nested=self.config.max_nested_searches > 0,
                retain_trace=False,
            )
            audit_samples = {
                action: np.asarray(
                    [
                        record["total"]
                        for record in audit_records
                        if record["root_action"] == action
                    ],
                    dtype=np.float64,
                )
                for action in validation_actions
            }
            audit_differences = (
                audit_samples[best_action]
                - audit_samples[policy_action]
            )
            audit_return_improvement = float(audit_differences.mean())
            audit_return_standard_error = float(
                audit_differences.std(ddof=1)
                / np.sqrt(len(audit_differences))
            )
            audit_return_lower_bound = float(
                audit_return_improvement
                - self.config.confidence_z
                * audit_return_standard_error
            )
            audit_policy_first = action_battle_scores(
                audit_records, policy_action, first_only=True
            )
            audit_best_first = action_battle_scores(
                audit_records, best_action, first_only=True
            )
            audit_policy_horizon = action_battle_scores(
                audit_records, policy_action
            )
            audit_best_horizon = action_battle_scores(
                audit_records, best_action
            )
            audit_first_battle_score_improvement = float(
                (audit_best_first - audit_policy_first).mean()
            )
            audit_horizon_battle_score_improvement = float(
                (audit_best_horizon - audit_policy_horizon).mean()
            )
        audit_wall_seconds = time.perf_counter() - audit_started
        rollout_traces = {
            action: tuple(
                RolloutTrace(
                    actions=tuple(record["actions"]),
                    rewards=tuple(record["rewards"]),
                    battle_results=tuple(record["battle_results"]),
                    leaf_value=record["leaf_value"],
                    total_return=float(record["total"]),
                    terminated=record["terminated"],
                    battles=record["battles"],
                    opponent_trajectory_index=record[
                        "opponent_trajectory_index"
                    ],
                    nested_searches=int(record["nested_searches"]),
                    opponent_trajectory_ids=record[
                        "opponent_trajectory_ids"
                    ],
                )
                for record in records
                if record["root_action"] == action
            )
            for action in confirmation_candidates
        }
        return RolloutSearchResult(
            best_action=best_action,
            policy_action=policy_action,
            scores=scores,
            priors={
                action: root.priors[action]
                for action in confirmation_candidates
            },
            standard_errors=standard_errors,
            improvement=improvement,
            improvement_standard_error=improvement_standard_error,
            improvement_lower_bound=float(improvement_lower_bound),
            accepted=accepted,
            rollouts=rollout_traces,
            screening_scores=screening_scores,
            screening_standard_errors=screening_standard_errors,
            screening_best_action=screening_best_action,
            exhaustive_root=exhaustive_root,
            full_candidate_count=len(root.priors),
            pruned_candidate_count=len(pruned_candidates),
            screening_best_excluded_by_pruning=(
                screening_best_action not in pruned_candidates
                if exhaustive_root
                else None
            ),
            pruned_action_counts=pruned_action_counts,
            confirmed_best_excluded_by_pruning=(
                confirmed_best_excluded
            ),
            validated_pruned_winner=validated_pruned_winner,
            policy_first_battle_score=float(policy_first_scores.mean()),
            best_first_battle_score=float(best_first_scores.mean()),
            first_battle_score_improvement=float(
                (best_first_scores - policy_first_scores).mean()
            ),
            policy_horizon_battle_score=float(
                policy_horizon_scores.mean()
            ),
            best_horizon_battle_score=float(best_horizon_scores.mean()),
            horizon_battle_score_improvement=float(
                (best_horizon_scores - policy_horizon_scores).mean()
            ),
            validation_return_improvement=validation_return_improvement,
            validation_return_standard_error=(
                validation_return_standard_error
            ),
            validation_return_lower_bound=float(
                validation_return_lower_bound
            ),
            validation_first_battle_score_improvement=float(
                (validation_best_first - validation_policy_first).mean()
            ),
            validation_horizon_battle_score_improvement=float(
                (
                    validation_best_horizon
                    - validation_policy_horizon
                ).mean()
            ),
            validation_differences=tuple(
                map(float, validation_differences)
            ),
            audit_return_improvement=audit_return_improvement,
            audit_return_standard_error=audit_return_standard_error,
            audit_return_lower_bound=audit_return_lower_bound,
            audit_first_battle_score_improvement=(
                audit_first_battle_score_improvement
            ),
            audit_horizon_battle_score_improvement=(
                audit_horizon_battle_score_improvement
            ),
            selection_simulator_transitions=(
                selection_simulator_transitions
            ),
            total_simulator_transitions=self.simulator_transitions,
            selection_wall_seconds=selection_wall_seconds,
            validation_wall_seconds=validation_wall_seconds,
            audit_wall_seconds=audit_wall_seconds,
            omitted_end_mask_probability_mass=root.omitted_probability_mass,
        )


class SelectiveNestedRolloutSearch(CounterfactualRolloutSearch):
    """Replan at a bounded number of important continuation decisions."""

    def search(self, state, memory, seed=0, validation_seed=None):
        self._nested_cache = {}
        return super().search(
            state,
            memory,
            seed=seed,
            validation_seed=validation_seed,
        )

    @staticmethod
    def _important_continuation(record):
        if not record.get("nested_enabled"):
            return False
        state = record["state"]
        core = getattr(state, "core", None)
        if core is None:
            return False
        player = core.agent
        last_type = ActionType(record["last_action"].action_type)
        if last_type is ActionType.ROLL:
            return True
        shop_names = {
            slot.obj.name for slot in player.shop if not slot.empty
        }
        if {"food-sleeping-pill", "food-mushroom"} & shop_names:
            return True
        team_names = {
            slot.pet.name for slot in player.team if not slot.empty
        }
        if any(
            slot.slot_type in {"pet", "levelup"}
            and slot.obj.name in team_names
            for slot in player.shop
        ):
            return True
        if len(player.team) == 5 and any(
            slot.cost <= player.gold for slot in player.shop
        ):
            return True
        if player.gold == 0 and any(
            ActionType(action.action_type) is ActionType.FREEZE
            for action in state.legal_actions()
        ):
            return True
        return False

    @staticmethod
    def _state_key(record):
        digest = hashlib.blake2b(digest_size=20)
        observation = record["state"].observation()
        for name in sorted(observation):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(observation[name]).tobytes())
        digest.update(
            torch.as_tensor(record["memory"])
            .detach()
            .float()
            .cpu()
            .numpy()
            .tobytes()
        )
        digest.update(str(record["nested_searches"]).encode())
        return digest.digest()

    def _nested_action(self, record):
        key = self._state_key(record)
        cached = self._nested_cache.get(key)
        if cached is not None:
            return cached
        state = record["state"]
        horizon = max(
            1,
            min(
                record["target_battles"] - state.battles,
                self.config.nested_lookahead_battles,
            ),
        )
        nested = CounterfactualRolloutSearch(
            self.evaluator,
            RolloutSearchConfig(
                rollouts_per_action=self.config.nested_rollouts_per_action,
                confirmation_rollouts=(
                    self.config.nested_confirmation_rollouts
                ),
                confirmation_top_k=self.config.nested_confirmation_top_k,
                validation_rollouts=2,
                audit_rollouts=0,
                horizon_battles=horizon,
                max_continuation_steps=self.config.max_continuation_steps,
                action_quotas=dict(self.config.action_quotas),
                minimum_improvement=0.0,
                confidence_z=0.0,
                exhaustive_root_probability=0.0,
                leaf_value_weight=self.config.leaf_value_weight,
                max_nested_searches=0,
                max_parallel_rollouts=self.config.max_parallel_rollouts,
                prune_affordable_freezes=(
                    self.config.prune_affordable_freezes
                ),
            ),
        )
        seed = int.from_bytes(key[:8], "little", signed=False)
        result = nested.search(state, record["memory"], seed=seed)
        action = (
            result.best_action if result.accepted else result.policy_action
        )
        self._nested_cache[key] = action
        return action

    def _choose_many(self, records, deterministic=True):
        choices = super()._choose_many(records, deterministic=deterministic)
        if not deterministic or self.config.max_nested_searches == 0:
            return choices
        revised = list(choices)
        for index, record in enumerate(records):
            if (
                record["nested_searches"] >= self.config.max_nested_searches
                or not self._important_continuation(record)
            ):
                continue
            revised[index] = (
                self._nested_action(record),
                choices[index][1],
                choices[index][2],
            )
            record["nested_searches"] += 1
        return revised


def soft_search_targets(
    result: RolloutSearchResult,
    *,
    temperature: float = 0.1,
    top_k: int = 4,
    minimum_probability: float = 0.02,
):
    """Convert searched action values into an AlphaZero-style soft target."""

    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= minimum_probability < 1.0:
        raise ValueError("minimum_probability must be in [0, 1)")
    ranked = sorted(
        result.scores.items(), key=lambda item: item[1], reverse=True
    )[:top_k]
    if temperature == 0:
        return [(ranked[0][0], 1.0)]
    values = np.asarray([score for _, score in ranked], dtype=np.float64)
    logits = (values - values.max()) / temperature
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    retained = [
        (action, float(probability))
        for (action, _), probability in zip(ranked, probabilities)
        if probability >= minimum_probability
    ]
    total = sum(probability for _, probability in retained)
    return [
        (action, probability / total) for action, probability in retained
    ]
