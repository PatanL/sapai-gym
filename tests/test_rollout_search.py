import unittest

import numpy as np
import torch

from sapai import Player, Shop, Team
from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    EntityObservationEncoder,
    FlatActionCodec,
    OpponentBeliefMix,
    OpponentBoardBelief,
    ParallelSelfPlayEnv,
)
from sapai_gym1.rl.rollout_search import (
    CounterfactualRolloutSearch,
    NeuralSearchEvaluator,
    PolicyEvaluation,
    PublicSearchState,
    RolloutSearchConfig,
    SearchTransition,
    SelectiveNestedRolloutSearch,
    soft_search_targets,
)
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir


class ToyDelayedState:
    freeze = AutoregressiveAction(ActionType.FREEZE, 0)
    end = AutoregressiveAction(ActionType.END_TURN)
    random_draws = []

    def __init__(self, phase="root", prepared=False):
        self.phase = phase
        self.prepared = prepared
        self.battles = 0
        self.terminated = False

    def clone(self):
        cloned = type(self)(self.phase, self.prepared)
        cloned.battles = self.battles
        cloned.terminated = self.terminated
        return cloned

    def observation(self):
        return {"phase": np.asarray([self.phase == "root"], dtype=np.int8)}

    def legal_actions(self):
        return [self.freeze, self.end] if self.phase == "root" else [self.end]

    def step(self, action, rng):
        if self.phase == "root":
            type(self).random_draws.append(
                int(rng.integers(0, 2**63 - 1))
            )
        if self.phase == "root" and action == self.freeze:
            self.prepared = True
            self.phase = "continuation"
            return SearchTransition(0.0, 1.0, False)
        self.battles += 1
        self.terminated = True
        return SearchTransition(
            1.0 if self.prepared else 0.0, 0.0, True, 0
        )


class ToyEvaluator:
    def evaluate(self, state, memory):
        del memory
        priors = {
            action: (0.01 if action == state.freeze else 0.99)
            for action in state.legal_actions()
        }
        total = sum(priors.values())
        return PolicyEvaluation(
            {action: value / total for action, value in priors.items()},
            0.0,
            torch.zeros(2),
        )

    def choose(self, state, memory, deterministic=True):
        del memory, deterministic
        return state.end, torch.zeros(2), 0.0

    def value(self, state, memory):
        del state, memory
        return 0.0


class AlwaysNestedSearch(SelectiveNestedRolloutSearch):
    @staticmethod
    def _important_continuation(record):
        return bool(record.get("nested_enabled"))


class ValidationRejectingSearch(CounterfactualRolloutSearch):
    """Make screening/selection like Freeze but fresh validation reject it."""

    def __init__(self, evaluator, config):
        super().__init__(evaluator, config)
        self.rollout_calls = 0

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
        del initial, next_memory, allow_nested
        call = self.rollout_calls
        self.rollout_calls += 1
        records = []
        for action in candidates:
            for _ in rollout_seeds:
                value = 1.0 if action == ToyDelayedState.freeze else 0.0
                if call == 2 and action == ToyDelayedState.freeze:
                    value = -1.0
                record = {
                    "root_action": action,
                    "total": value,
                    "battle_results": [0 if value > 0 else 1],
                    "leaf_value": None,
                    "nested_searches": 0,
                    "terminated": True,
                    "battles": 1,
                    "opponent_trajectory_index": None,
                    "opponent_trajectory_ids": (),
                }
                if retain_trace:
                    record["actions"] = [action]
                    record["rewards"] = [value]
                records.append(record)
        return records


class RolloutSearchTest(unittest.TestCase):
    @staticmethod
    def _opponent_belief(name="dragon"):
        encoder = EntityObservationEncoder()
        pet_logits = np.full((5, encoder.num_items), -20.0)
        pet_logits[:, 0] = 20.0
        pet_logits[0, :] = -20.0
        pet_logits[0, encoder.item_to_id[f"pet-{name}"]] = 20.0
        status_logits = np.full((5, encoder.num_statuses), -20.0)
        status_logits[:, encoder.status_to_id["none"]] = 20.0
        level_logits = np.full((5, 4), -20.0)
        level_logits[:, 1] = 20.0
        return OpponentBoardBelief.from_logits(
            pet_logits,
            status_logits,
            level_logits,
            np.full((5, 2), 0.2),
            np.full((5, 2), -20.0),
            encoder,
        )

    def _core_with_history(self):
        env = ParallelSelfPlayEnv()
        env.reset(seed=812)
        env.core.agent = Player(
            shop=Shop(["ant", "fish", "mosquito", "apple"], turn=3),
            team=Team(["fish", "ant"]),
            gold=3,
            turn=3,
            record_actions=False,
        )
        env.core.opponent = Player(
            shop=Shop(["pig", "horse", "beaver", "honey"], turn=3),
            team=Team(["hippo", "skunk"]),
            gold=7,
            turn=3,
            lives=4,
            wins=2,
            record_actions=False,
        )
        env.core.shop_agent = env.core.agent.shop
        env.core.team_agent = env.core.agent.team
        env.core.shop_opponent = env.core.opponent.shop
        env.core.team_opponent = env.core.opponent.team
        env.core.turn = 3
        env.core.previous_opponent_state = Team(["ant", "beaver"])
        env.core.previous_previous_opponent_state = Team(["fish"])
        env.core.previous_agent_state = Team(["fish", "ant"])
        env.core.previous_previous_agent_state = Team(["fish"])
        env.core._agent_actions_cache = None
        env.core._opponent_actions_cache = None
        env.core._freeze_toggled_agent.clear()
        env.core._seen_team_orders_agent = {
            env.core._team_order_key(env.core.agent)
        }
        env.core._freeze_toggled_opponent.clear()
        env.core._seen_team_orders_opponent = {
            env.core._team_order_key(env.core.opponent)
        }
        return env.core

    def test_public_state_does_not_copy_hidden_current_opponent(self):
        core = self._core_with_history()
        expected = EntityObservationEncoder().encode(core, 0)
        first = PublicSearchState.from_core(core, 0)
        for key, value in expected.items():
            np.testing.assert_array_equal(first.observation()[key], value)
        self.assertEqual(
            [slot.pet.name for slot in first.core.opponent.team if not slot.empty],
            ["pet-ant", "pet-beaver"],
        )

        core.opponent.team = Team(["leopard", "dragon", "tiger"])
        core.opponent.shop = Shop(["cat", "snake", "fly", "pizza"], turn=3)
        second = PublicSearchState.from_core(core, 0)
        for key in first.observation():
            np.testing.assert_array_equal(
                first.observation()[key], second.observation()[key]
            )

    def test_interactive_truth_clone_preserves_hidden_state_and_future_rng(self):
        env = ParallelSelfPlayEnv()
        env.reset(seed=4812)
        freeze = AutoregressiveAction(ActionType.FREEZE, 0)
        env.step(freeze, freeze)
        before = env.observations()
        first = env.clone()
        second = env.clone()
        different = env.clone()
        first.redeterminize(991)
        second.redeterminize(991)
        different.redeterminize(992)
        for left, right in zip(before, env.observations()):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
        roll = AutoregressiveAction(ActionType.ROLL)
        first.step(roll, roll)
        second.step(roll, roll)
        different.step(roll, roll)

        def shops(candidate):
            return tuple(
                tuple(slot.obj.name for slot in player.shop)
                for player in (candidate.core.agent, candidate.core.opponent)
            )

        self.assertEqual(shops(first), shops(second))
        self.assertNotEqual(shops(first), shops(different))
        self.assertNotEqual(shops(env), shops(first))

    def test_clone_preserves_masks_but_is_independent(self):
        core = self._core_with_history()
        core._freeze_toggled_agent.add(id(core.agent.shop[1].obj))
        state = PublicSearchState.from_core(core, 0)
        cloned = state.clone()
        for key in state.observation():
            np.testing.assert_array_equal(
                state.observation()[key], cloned.observation()[key]
            )
        action = next(
            action
            for action in cloned.legal_actions()
            if ActionType(action.action_type) is ActionType.BUY_PET
        )
        cloned.step(action, np.random.default_rng(1))
        self.assertNotEqual(
            cloned.core.agent.gold, state.core.agent.gold
        )

    def test_player_one_public_view_is_preserved_without_player_zero_shop(self):
        core = self._core_with_history()
        expected = EntityObservationEncoder().encode(core, 1)
        first = PublicSearchState.from_core(core, 1)
        for key, value in expected.items():
            np.testing.assert_array_equal(first.observation()[key], value)

        core.agent.team = Team(["dragon", "leopard", "tiger"])
        core.agent.shop = Shop(["cat", "snake", "fly", "pizza"], turn=3)
        second = PublicSearchState.from_core(core, 1)
        for key in first.observation():
            np.testing.assert_array_equal(
                first.observation()[key], second.observation()[key]
            )

    def test_zero_gold_freeze_survives_search_turn_transition(self):
        core = self._core_with_history()
        core.agent = Player(
            shop=Shop(
                ["ant", "apple", "fish", "beaver"],
                turn=2,
                seed_state=np.random.RandomState(929).get_state(),
                fixed_rules=True,
            ),
            team=Team(["fish"]),
            gold=0,
            turn=2,
            record_actions=False,
        )
        core.agent.team[0].pet.set_total_experience(1)
        core.shop_agent = core.agent.shop
        core.team_agent = core.agent.team
        core.turn = 2
        core.previous_opponent_state = Team(["ant"])
        core.previous_previous_opponent_state = Team(["fish"])
        core._agent_actions_cache = None
        core._freeze_toggled_agent.clear()
        core._seen_team_orders_agent = {core._team_order_key(core.agent)}
        state = PublicSearchState.from_core(core, 0)

        state.step(
            AutoregressiveAction(ActionType.FREEZE, 2),
            np.random.default_rng(3),
        )
        transition = state.step(
            AutoregressiveAction(ActionType.END_TURN),
            np.random.default_rng(3),
        )
        self.assertTrue(transition.battle_resolved)
        self.assertEqual(state.battles, 1)
        self.assertEqual(state.core.agent.turn, 3)
        self.assertEqual(state.core.agent.shop[0].obj.name, "pet-fish")
        self.assertTrue(state.core.agent.shop[0].frozen)
        buy = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_PET, 0, 0)
        )
        self.assertIn(buy, state.core._avail_actions(0))

    def test_common_seed_selects_same_opponent_future_for_every_action(self):
        core = self._core_with_history()
        core.agent.gold = 0
        core._agent_actions_cache = None
        state = PublicSearchState.from_core(core, 0)
        state.set_opponent_trajectories(
            [
                (Team([name]), Team([name, "fish"]))
                for name in ("ant", "beaver", "cricket", "horse")
            ]
        )
        direct = state.clone()
        frozen = state.clone()
        direct.step(
            AutoregressiveAction(ActionType.END_TURN),
            np.random.default_rng(991),
        )
        frozen.step(
            AutoregressiveAction(ActionType.FREEZE, 0),
            np.random.default_rng(991),
        )
        frozen.step(
            AutoregressiveAction(ActionType.END_TURN),
            np.random.default_rng(991),
        )
        self.assertEqual(
            direct.selected_opponent_trajectory,
            frozen.selected_opponent_trajectory,
        )

    def test_learned_public_belief_can_supply_synthetic_opponent(self):
        core = self._core_with_history()
        core.agent.gold = 0
        core._agent_actions_cache = None
        state = PublicSearchState.from_core(core, 0)
        state.enable_learned_opponent_belief(
            OpponentBeliefMix(
                synthetic_weight=1.0,
                persistence_weight=0.0,
                reservoir_weight=0.0,
            )
        )
        state.set_opponent_board_belief(self._opponent_belief("dragon"))
        cloned = state.clone()
        cloned.step(
            AutoregressiveAction(ActionType.END_TURN),
            np.random.default_rng(991),
        )
        self.assertEqual(
            cloned.selected_opponent_trajectory_ids[0],
            "belief-synthetic",
        )
        self.assertEqual(
            cloned.core.previous_opponent_state[0].pet.name,
            "pet-dragon",
        )
        self.assertIsNone(cloned.opponent_board_belief)

    def test_redeterminization_hides_future_shop_rng_but_is_reproducible(self):
        core = self._core_with_history()
        state = PublicSearchState.from_core(core, 0)
        before = state.observation()
        first = state.clone()
        second = state.clone()
        different = state.clone()
        first.redeterminize(12345)
        second.redeterminize(12345)
        different.redeterminize(54321)
        for key in before:
            np.testing.assert_array_equal(before[key], first.observation()[key])

        for candidate in (first, second, different):
            candidate.core.agent.gold = 1
            candidate.core._agent_actions_cache = None
            candidate.step(
                AutoregressiveAction(ActionType.ROLL),
                np.random.default_rng(7),
            )
        first_shop = [slot.obj.name for slot in first.core.agent.shop]
        second_shop = [slot.obj.name for slot in second.core.agent.shop]
        different_shop = [slot.obj.name for slot in different.core.agent.shop]
        self.assertEqual(first_shop, second_shop)
        self.assertNotEqual(first_shop, different_shop)

    def test_root_move_candidates_wait_for_pill_or_final_positioning(self):
        core = self._core_with_history()
        state = PublicSearchState.from_core(core, 0)
        move = AutoregressiveAction(ActionType.MOVE, 0, 1)
        roll = AutoregressiveAction(ActionType.ROLL)
        search = CounterfactualRolloutSearch(ToyEvaluator())
        candidates = search._candidate_actions(
            {move: 0.5, roll: 0.5}, state
        )
        self.assertNotIn(move, candidates)
        self.assertIn(roll, candidates)

        state.core.agent.shop = Shop(
            ["ant", "sleeping-pill"], turn=3, fixed_rules=True
        )
        self.assertTrue(search._root_moves_are_strategic(state))
        state.core.agent.shop = Shop(["ant", "apple"], turn=3, fixed_rules=True)
        state.core.agent.gold = 0
        self.assertTrue(search._root_moves_are_strategic(state))

    def test_affordable_freeze_candidates_keep_only_explicit_exceptions(self):
        state = PublicSearchState.from_core(self._core_with_history(), 0)
        search = CounterfactualRolloutSearch(ToyEvaluator())
        freeze = AutoregressiveAction(ActionType.FREEZE, 0)
        roll = AutoregressiveAction(ActionType.ROLL)
        state.core.agent.gold = 3
        self.assertFalse(search._root_freeze_is_strategic(state, freeze))
        self.assertIn(
            freeze,
            search._candidate_actions({freeze: 0.5, roll: 0.5}, state),
        )
        pruned_search = CounterfactualRolloutSearch(
            ToyEvaluator(),
            RolloutSearchConfig(prune_affordable_freezes=True),
        )
        self.assertNotIn(
            freeze,
            pruned_search._candidate_actions(
                {freeze: 0.5, roll: 0.5}, state
            ),
        )

        state.core.agent.gold = 2
        self.assertTrue(search._root_freeze_is_strategic(state, freeze))

        state.core.agent.gold = 3
        state.core.agent.turn = 2
        state.core.agent.team[1].pet.set_total_experience(1)
        self.assertTrue(search._root_freeze_is_strategic(state, freeze))

        state.core.agent.turn = 5
        state.core.agent.team = Team(["fish", "ant", "beaver"])
        self.assertTrue(search._root_freeze_is_strategic(state, freeze))

        state.core.agent.turn = 6
        state.core.agent.team = Team(
            ["fish", "ant", "beaver", "horse", "mosquito"]
        )
        state.core.agent.shop = Shop(["pig", "apple"], turn=6, fixed_rules=True)
        self.assertFalse(search._root_freeze_is_strategic(state, freeze))
        state.core.agent.turn = 7
        self.assertTrue(search._root_freeze_is_strategic(state, freeze))

    def test_future_opponent_response_depends_on_revealed_candidate_board(self):
        def board(turn, own, enemy):
            return {
                "turn": turn,
                "team": Team(own).state,
                "opponentTeam": Team(enemy).state,
            }

        reservoir = OpponentTrajectoryReservoir(
            [
                {
                    "id": "fish-response",
                    "boards": [
                        board(2, ["ant"], ["fish"]),
                        board(3, ["rhino"], ["fish"]),
                    ],
                },
                {
                    "id": "dragon-response",
                    "boards": [
                        board(2, ["ant"], ["dragon"]),
                        board(3, ["skunk"], ["dragon"]),
                    ],
                },
            ]
        )
        core = self._core_with_history()
        core.agent.gold = 0
        core.agent.turn = 3
        core.turn = 3
        core.previous_opponent_state = Team(["ant"])
        core._agent_actions_cache = None
        base = PublicSearchState.from_core(core, 0)
        base.battles = 1
        base.set_opponent_transition_model(reservoir, candidate_limit=1)

        fish = base.clone()
        fish.core.previous_agent_state = Team(["fish"])
        fish.step(
            AutoregressiveAction(ActionType.END_TURN),
            np.random.default_rng(101),
        )
        dragon = base.clone()
        dragon.core.previous_agent_state = Team(["dragon"])
        dragon.step(
            AutoregressiveAction(ActionType.END_TURN),
            np.random.default_rng(101),
        )
        self.assertEqual(
            fish.core.previous_opponent_state[0].pet.name, "pet-rhino"
        )
        self.assertEqual(
            dragon.core.previous_opponent_state[0].pet.name, "pet-skunk"
        )

    def test_rollout_search_finds_low_prior_delayed_action(self):
        ToyDelayedState.random_draws.clear()
        search = CounterfactualRolloutSearch(
            ToyEvaluator(),
            RolloutSearchConfig(
                rollouts_per_action=4,
                confirmation_rollouts=4,
                validation_rollouts=4,
                horizon_battles=1,
                minimum_improvement=0.1,
                max_parallel_rollouts=1,
            ),
        )
        result = search.search(ToyDelayedState(), torch.zeros(2), seed=17)
        self.assertEqual(result.policy_action, ToyDelayedState.end)
        self.assertEqual(result.best_action, ToyDelayedState.freeze)
        self.assertEqual(result.improvement, 1.0)
        self.assertEqual(result.validation_return_improvement, 1.0)
        self.assertEqual(result.audit_return_improvement, 1.0)
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.screening_scores), 2)
        self.assertTrue(
            all(len(traces) == 4 for traces in result.rollouts.values())
        )
        screening_draws = set(ToyDelayedState.random_draws[:8])
        confirmation_draws = set(ToyDelayedState.random_draws[8:16])
        self.assertTrue(screening_draws.isdisjoint(confirmation_draws))
        targets = soft_search_targets(
            result, temperature=0.25, top_k=2, minimum_probability=0.0
        )
        self.assertEqual([action for action, _ in targets][0], result.best_action)
        self.assertAlmostEqual(sum(weight for _, weight in targets), 1.0)
        self.assertGreater(targets[0][1], targets[1][1])

    def test_fresh_fixed_winner_validation_gates_search_label(self):
        search = ValidationRejectingSearch(
            ToyEvaluator(),
            RolloutSearchConfig(
                rollouts_per_action=2,
                confirmation_rollouts=2,
                validation_rollouts=2,
                audit_rollouts=2,
                horizon_battles=1,
                minimum_improvement=0.1,
                confidence_z=0.0,
            ),
        )
        result = search.search(ToyDelayedState(), torch.zeros(2), seed=23)
        self.assertEqual(result.best_action, ToyDelayedState.freeze)
        self.assertGreater(result.improvement, 0.0)
        self.assertLess(result.validation_return_improvement, 0.0)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.audit_return_improvement)
        self.assertEqual(search.rollout_calls, 3)

    def test_selective_nested_search_replans_but_respects_budget(self):
        search = AlwaysNestedSearch(
            ToyEvaluator(),
            RolloutSearchConfig(
                rollouts_per_action=2,
                confirmation_rollouts=2,
                confirmation_top_k=2,
                validation_rollouts=2,
                horizon_battles=1,
                minimum_improvement=0.0,
                max_nested_searches=1,
                nested_rollouts_per_action=2,
                nested_confirmation_rollouts=2,
            ),
        )
        result = search.search(ToyDelayedState(), torch.zeros(2), seed=19)
        freeze_traces = result.rollouts[ToyDelayedState.freeze]
        self.assertTrue(freeze_traces)
        self.assertTrue(
            all(trace.nested_searches == 1 for trace in freeze_traces)
        )

    def test_neural_priors_are_normalized_and_match_greedy_action(self):
        core = self._core_with_history()
        state = PublicSearchState.from_core(core, 0)
        state.enable_learned_opponent_belief()
        encoder = state.encoder
        policy = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                memory_dim=24,
            )
        ).eval()
        evaluator = NeuralSearchEvaluator(policy, device="cpu", chunk_size=8)
        evaluation = evaluator.evaluate(state, policy.initial_memory(1)[0])
        self.assertIsInstance(
            evaluation.opponent_belief, OpponentBoardBelief
        )
        self.assertIs(state.opponent_board_belief, evaluation.opponent_belief)
        self.assertAlmostEqual(sum(evaluation.priors.values()), 1.0, places=6)
        actions = state.legal_actions()
        observation = state.observation()
        repeated = {
            key: torch.as_tensor(value)
            .unsqueeze(0)
            .expand(len(actions), *value.shape)
            for key, value in observation.items()
        }
        forced = policy(
            repeated,
            memory=policy.initial_memory(len(actions)),
            actions=torch.tensor(
                [action.as_tuple() for action in actions], dtype=torch.long
            ),
        )
        expected = forced.log_prob.exp()
        expected = expected / expected.sum()
        for action, probability in zip(actions, expected.tolist()):
            self.assertAlmostEqual(
                evaluation.priors[action], probability, places=5
            )
        greedy, _, _ = evaluator.choose(
            state, policy.initial_memory(1)[0], deterministic=True
        )
        self.assertIn(greedy, evaluation.priors)
        self.assertGreater(evaluation.priors[greedy], 0.0)

    def test_neural_evaluator_applies_optional_value_calibration_everywhere(self):
        core = self._core_with_history()
        state = PublicSearchState.from_core(core, 0)
        encoder = state.encoder
        policy = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                memory_dim=24,
            )
        ).eval()

        class AddTwo:
            @staticmethod
            def correct(value, search_state):
                self.assertIs(search_state, state)
                return float(value) + 2.0

        memory = policy.initial_memory(1)[0]
        raw = NeuralSearchEvaluator(policy, device="cpu", chunk_size=8)
        calibrated = NeuralSearchEvaluator(
            policy,
            device="cpu",
            chunk_size=8,
            value_calibrator=AddTwo(),
        )
        self.assertAlmostEqual(
            calibrated.evaluate(state, memory).value,
            raw.evaluate(state, memory).value + 2.0,
            places=5,
        )
        self.assertAlmostEqual(
            calibrated.value(state, memory),
            raw.value(state, memory) + 2.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
