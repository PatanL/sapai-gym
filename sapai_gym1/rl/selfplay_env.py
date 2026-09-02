"""Explicit two-player environment for batched league self-play actors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sapai import Battle, Team
from .action_schema import AutoregressiveAction, FlatActionCodec
from .entity_observation import EntityObservationEncoder


@dataclass
class SelfPlayStep:
    observations: tuple[dict, dict]
    rewards: tuple[float, float]
    battle_resolved: bool
    terminated: bool
    battle_result: int | None = None


class ParallelSelfPlayEnv:
    """Drive both shop phases explicitly instead of hiding an opponent policy."""

    def __init__(self, fast_battles: bool = True):
        from sapai_gym1.SuperAutoPetsEnv import SuperAutoPetsEnv

        self.core = SuperAutoPetsEnv(
            opponent_generator=None,
            valid_actions_only=True,
            manual_battles=True,
        )
        self.encoder = EntityObservationEncoder()
        self.fast_battles = bool(fast_battles)

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self.core.reset(seed=seed)
        return self.observations()

    def observations(self) -> tuple[dict, dict]:
        return self.encoder.encode(self.core, 0), self.encoder.encode(self.core, 1)

    def active_players(self) -> tuple[bool, bool]:
        return not self.core.agent.turn_ended, not self.core.opponent.turn_ended

    def step(
        self,
        player_zero_action: AutoregressiveAction | np.ndarray | None,
        player_one_action: AutoregressiveAction | np.ndarray | None,
    ) -> SelfPlayStep:
        actions = (player_zero_action, player_one_action)
        for player_idx, action in enumerate(actions):
            player = self.core.agent if player_idx == 0 else self.core.opponent
            if player.turn_ended:
                if action is not None:
                    raise ValueError(f"Player {player_idx} already ended the turn")
                continue
            if action is None:
                raise ValueError(f"Active player {player_idx} requires an action")
            autoregressive = self._coerce_and_validate(player_idx, action)
            self.core.resolve_action(
                agent_idx=player_idx,
                action=FlatActionCodec.encode(autoregressive),
            )

        if not (self.core.agent.turn_ended and self.core.opponent.turn_ended):
            return SelfPlayStep(self.observations(), (0.0, 0.0), False, False)

        self.core.previous_previous_opponent_state = (
            self.core.previous_opponent_state
        )
        self.core.previous_previous_agent_state = self.core.previous_agent_state
        self.core.previous_opponent_state = Team.from_state(
            self.core.opponent.team.state
        )
        self.core.previous_agent_state = Team.from_state(self.core.agent.team.state)
        battle_result = Battle(
            self.core.agent.team,
            self.core.opponent.team,
            seed=int(self.core.rs_random.randint(0, 2**32 - 1)),
            fast_training=self.fast_battles,
        ).battle()
        lives_before = (self.core.agent.lives, self.core.opponent.lives)
        self.core._player_fight_outcome(battle_result)
        life_deltas = (
            self.core.agent.lives - lives_before[0],
            self.core.opponent.lives - lives_before[1],
        )
        self.core.turn += 1
        terminated = self.core.is_done()

        life_swing = life_deltas[0] - life_deltas[1]
        rewards = [0.1 * life_swing, -0.1 * life_swing]
        if self.core.opponent.lives <= 0:
            rewards[0] += 2.0
            rewards[1] -= 2.0
        if self.core.agent.lives <= 0:
            rewards[0] -= 2.0
            rewards[1] += 2.0
        if (
            terminated
            and self.core.agent.lives > 0
            and self.core.opponent.lives > 0
        ):
            player_score = (self.core.agent.lives, self.core.agent.wins)
            opponent_score = (
                self.core.opponent.lives,
                self.core.opponent.wins,
            )
            if player_score > opponent_score:
                rewards[0] += 2.0
                rewards[1] -= 2.0
            elif opponent_score > player_score:
                rewards[0] -= 2.0
                rewards[1] += 2.0

        if not terminated:
            self.core.agent.start_turn(winner=self.core.agent.lf_winner)
            self.core.opponent.start_turn(
                winner=self.core.opponent.lf_winner
            )
            self.core.reorder_count_agent = 0
            self.core.reorder_count_opponent = 0
            self.core.shop_action_count_agent = 0
            self.core.shop_action_count_opponent = 0
            self.core._freeze_toggled_agent.clear()
            self.core._freeze_toggled_opponent.clear()
            self.core._seen_team_orders_agent = {
                self.core._team_order_key(self.core.agent)
            }
            self.core._seen_team_orders_opponent = {
                self.core._team_order_key(self.core.opponent)
            }

        return SelfPlayStep(
            observations=self.observations(),
            rewards=(float(rewards[0]), float(rewards[1])),
            battle_resolved=True,
            terminated=terminated,
            battle_result=battle_result,
        )

    def _coerce_and_validate(self, player_idx, action) -> AutoregressiveAction:
        if isinstance(action, AutoregressiveAction):
            autoregressive = action
        else:
            action = np.asarray(action, dtype=np.int64)
            if action.shape != (3,):
                raise ValueError(f"Expected action shape (3,), got {action.shape}")
            autoregressive = AutoregressiveAction(*map(int, action))
        flat_action = FlatActionCodec.encode(autoregressive)
        if flat_action not in self.core._avail_actions(player_idx):
            raise ValueError(
                f"Illegal action {autoregressive.as_tuple()} for player {player_idx}"
            )
        return autoregressive
