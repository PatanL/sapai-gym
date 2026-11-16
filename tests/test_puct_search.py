import unittest

import numpy as np
import torch

from sapai_gym1.rl import ActionType, AutoregressiveAction
from sapai_gym1.rl.puct_search import PUCTConfig, StochasticPUCTSearch
from sapai_gym1.rl.rollout_search import PolicyEvaluation, SearchTransition


class ToyComboState:
    shark = AutoregressiveAction(ActionType.BUY_PET, 0, 0)
    monkey = AutoregressiveAction(ActionType.BUY_PET, 1, 0)
    fly = AutoregressiveAction(ActionType.BUY_PET, 2, 1)
    roll = AutoregressiveAction(ActionType.ROLL)
    end = AutoregressiveAction(ActionType.END_TURN)

    def __init__(self, phase="root", shark_owned=False):
        self.phase = phase
        self.shark_owned = shark_owned
        self.battles = 0
        self.terminated = False

    def clone(self):
        cloned = type(self)(self.phase, self.shark_owned)
        cloned.battles = self.battles
        cloned.terminated = self.terminated
        return cloned

    def redeterminize(self, seed):
        del seed

    def observation(self):
        phases = {"root": 0, "shark": 1, "monkey": 2}
        return {
            "phase": np.asarray([phases[self.phase]], dtype=np.int64),
            "shark": np.asarray([self.shark_owned], dtype=np.int8),
        }

    def legal_actions(self):
        if self.phase == "root":
            return [self.shark, self.monkey]
        if self.phase == "shark":
            return [self.fly, self.roll]
        return [self.end]

    def step(self, action, rng):
        del rng
        if self.phase == "root" and action == self.shark:
            self.phase = "shark"
            self.shark_owned = True
            return SearchTransition(0.0, 1.0, False)
        if self.phase == "root" and action == self.monkey:
            self.phase = "monkey"
            return SearchTransition(0.0, 1.0, False)
        reward = 0.0
        if self.phase == "shark" and action == self.fly:
            reward = 1.0
        elif self.phase == "monkey" and action == self.end:
            reward = 0.2
        self.battles = 1
        self.terminated = True
        return SearchTransition(
            reward, 0.0, True, 0 if reward > 0.2 else 1
        )


class ToyComboEvaluator:
    def evaluate(self, state, memory):
        del memory
        if state.phase == "root":
            priors = {state.shark: 0.05, state.monkey: 0.95}
        elif state.phase == "shark":
            priors = {state.fly: 0.05, state.roll: 0.95}
        else:
            priors = {state.end: 1.0}
        return PolicyEvaluation(priors, 0.0, torch.zeros(2))

    def choose(self, state, memory, deterministic=True):
        del memory, deterministic
        if state.phase == "root":
            action = state.monkey
        elif state.phase == "shark":
            action = state.roll
        else:
            action = state.end
        return action, torch.zeros(2), 0.0

    def value(self, state, memory):
        del state, memory
        return 0.0


class ScriptedValidationPUCT(StochasticPUCTSearch):
    def __init__(self, evaluator, config, planned_values):
        super().__init__(evaluator, config)
        self.planned_values = list(planned_values)
        self.planned_index = 0

    def _validation_rollout(
        self,
        initial,
        memory,
        root_action,
        seed,
        *,
        use_tree,
    ):
        del initial, memory, root_action, seed
        if use_tree:
            value = self.planned_values[self.planned_index]
            self.planned_index += 1
        else:
            value = 0.0
        return {"total": value, "battle_results": [0 if value > 0 else 1]}

    def _confirmation_rollouts(
        self, state, memory, actions, seeds, *, use_tree
    ):
        return [
            self._validation_rollout(
                state,
                memory,
                action,
                seed,
                use_tree=use_tree,
            )
            for action in actions
            for seed in seeds
        ]


class ActionValuePUCT(StochasticPUCTSearch):
    def _screen_rollouts(
        self, state, memory, actions, seeds, *, use_tree
    ):
        del state, memory, use_tree
        values = {
            ToyComboState.monkey: 0.0,
            ToyComboState.fly: 0.25,
            ToyComboState.shark: 1.0,
        }
        return [
            {
                "root_action": action,
                "total": values[action],
                "battle_results": [0 if values[action] > 0 else 1],
            }
            for action in actions
            for _ in seeds
        ]

    def _validation_rollout(
        self,
        initial,
        memory,
        root_action,
        seed,
        *,
        use_tree,
    ):
        del initial, memory, seed, use_tree
        values = {
            ToyComboState.monkey: 0.0,
            ToyComboState.fly: 0.25,
            ToyComboState.shark: 1.0,
        }
        value = values[root_action]
        return {
            "total": value,
            "battle_results": [0 if value > 0 else 1],
            "tree_actions": 0,
        }


class SeedRecordingPUCT(ActionValuePUCT):
    def __init__(self, evaluator, config):
        super().__init__(evaluator, config)
        self.confirmation_seed_calls = []

    def _confirmation_rollouts(
        self, state, memory, actions, seeds, *, use_tree
    ):
        del state, memory
        self.confirmation_seed_calls.append(
            (use_tree, tuple(int(seed) for seed in seeds))
        )
        values = {
            ToyComboState.monkey: 0.0,
            ToyComboState.fly: 0.25,
            ToyComboState.shark: 1.0,
        }
        return [
            {
                "root_action": action,
                "total": values[action],
                "battle_results": [0 if values[action] > 0 else 1],
                "tree_actions": 0,
            }
            for action in actions
            for _ in seeds
        ]


class TestStochasticPUCT(unittest.TestCase):
    def test_opponent_source_telemetry_decodes_mixed_trajectory_ids(self):
        counts = StochasticPUCTSearch._opponent_source_counts(
            [
                {
                    "opponent_trajectory_ids": (
                        "source-00-game-a",
                        "source-02-game-b",
                    )
                },
                {
                    "opponent_trajectory_ids": (
                        "source-00-game-c",
                        "legacy-game",
                    )
                },
            ]
        )
        self.assertEqual(
            counts,
            {"source-00": 2, "source-02": 1, "unmixed": 1},
        )

    def test_exhaustive_root_audit_can_recover_quota_pruned_combo(self):
        config = PUCTConfig(
            max_simulations=256,
            max_simulator_transitions=900,
            max_tree_depth=8,
            max_rollout_steps=8,
            horizon_battles=1,
            world_pool_size=8,
            min_root_visits=2,
            cpuct=3.0,
            prior_floor=0.4,
            leaf_value_weight=0.0,
            validation_rollouts=16,
            max_validation_rollouts=16,
            audit_rollouts=16,
            confidence_z=0.0,
            minimum_improvement=0.1,
            exhaustive_root_probability=1.0,
            action_quotas={ActionType.BUY_PET: 1},
        )
        result = StochasticPUCTSearch(ToyComboEvaluator(), config).search(
            ToyComboState(), torch.zeros(2), seed=177
        )
        self.assertTrue(result.exhaustive_root)
        self.assertEqual(result.full_candidate_count, 2)
        self.assertEqual(result.pruned_candidate_count, 1)
        self.assertEqual(result.pruned_action_counts, {"BUY_PET": 1})
        self.assertEqual(result.proposed_action, ToyComboState.shark)
        self.assertTrue(result.proposed_excluded_by_pruning)
        self.assertTrue(result.accepted)

    def test_validation_and_outer_audit_use_matched_but_disjoint_seeds(self):
        search = SeedRecordingPUCT(
            ToyComboEvaluator(),
            PUCTConfig(
                validation_rollouts=8,
                max_validation_rollouts=8,
                audit_rollouts=8,
                confidence_z=0.0,
            ),
        )
        rng = np.random.default_rng(12345)
        search._validate(
            ToyComboState(),
            torch.zeros(2),
            ToyComboState.shark,
            ToyComboState.monkey,
            rng,
        )
        search._audit(
            ToyComboState(),
            torch.zeros(2),
            ToyComboState.shark,
            ToyComboState.monkey,
            rng,
        )
        validation_planned = search.confirmation_seed_calls[0][1]
        validation_policy = search.confirmation_seed_calls[1][1]
        audit_planned = search.confirmation_seed_calls[2][1]
        audit_policy = search.confirmation_seed_calls[3][1]
        self.assertEqual(validation_planned, validation_policy)
        self.assertEqual(audit_planned, audit_policy)
        self.assertTrue(set(validation_planned).isdisjoint(audit_planned))

    def test_root_racing_spends_more_fresh_samples_on_survivors(self):
        search = ActionValuePUCT(
            ToyComboEvaluator(),
            PUCTConfig(
                root_screening_top_k=3,
                root_screening_rollouts=2,
                root_screening_rounds=3,
                root_screening_min_survivors=1,
            ),
        )
        proposed, scores, counts = search._screen_root_actions(
            ToyComboState(),
            torch.zeros(2),
            (
                ToyComboState.monkey,
                ToyComboState.fly,
                ToyComboState.shark,
            ),
            ToyComboState.monkey,
            np.random.default_rng(7),
        )
        self.assertEqual(proposed, ToyComboState.shark)
        self.assertGreater(scores[ToyComboState.shark], scores[ToyComboState.fly])
        self.assertEqual(counts[ToyComboState.fly], 2)
        self.assertEqual(counts[ToyComboState.shark], 24)
        self.assertEqual(counts[ToyComboState.monkey], 24)

    def test_ambiguous_validation_escalates_but_clear_result_stops(self):
        config = PUCTConfig(
            validation_rollouts=32,
            max_validation_rollouts=128,
            confidence_z=1.96,
            minimum_improvement=0.0,
        )
        ambiguous_then_good = [1.0, -1.0] * 16 + [1.0] * 96
        search = ScriptedValidationPUCT(
            ToyComboEvaluator(), config, ambiguous_then_good
        )
        result = search._validate(
            ToyComboState(),
            torch.zeros(2),
            ToyComboState.shark,
            ToyComboState.monkey,
            np.random.default_rng(5),
        )
        self.assertTrue(result[0])
        self.assertGreater(result[3], 0.0)
        self.assertEqual(result[-1], 64)

        clearly_bad = ScriptedValidationPUCT(
            ToyComboEvaluator(), config, [-1.0] * 128
        )
        result = clearly_bad._validate(
            ToyComboState(),
            torch.zeros(2),
            ToyComboState.shark,
            ToyComboState.monkey,
            np.random.default_rng(5),
        )
        self.assertFalse(result[0])
        self.assertEqual(result[-1], 32)

    def test_puct_discovers_low_prior_two_action_combo_with_global_budget(self):
        config = PUCTConfig(
            max_simulations=256,
            max_simulator_transitions=900,
            max_tree_depth=8,
            max_rollout_steps=8,
            horizon_battles=1,
            world_pool_size=8,
            min_root_visits=2,
            cpuct=3.0,
            prior_floor=0.4,
            leaf_value_weight=0.0,
            validation_rollouts=16,
            confidence_z=0.0,
            minimum_improvement=0.1,
        )
        search = StochasticPUCTSearch(ToyComboEvaluator(), config)
        result = search.search(ToyComboState(), torch.zeros(2), seed=77)
        self.assertEqual(result.proposed_action, ToyComboState.shark)
        self.assertEqual(result.selected_action, ToyComboState.shark)
        self.assertTrue(result.accepted)
        self.assertTrue(result.candidate_eligible)
        self.assertGreater(result.validation_return_improvement, 0.7)
        self.assertGreater(result.audit_return_improvement, 0.7)
        self.assertGreater(result.validation_tree_actions_mean, 0.0)
        self.assertGreater(result.audit_tree_actions_mean, 0.0)
        self.assertGreater(result.screening_simulator_transitions, 0)
        self.assertGreater(result.validation_simulator_transitions, 0)
        self.assertGreater(result.audit_simulator_transitions, 0)
        self.assertGreaterEqual(result.max_tree_depth, 2)
        self.assertEqual(result.validation_rollouts_used, 16)
        self.assertLessEqual(
            result.simulator_transitions,
            config.max_simulator_transitions,
        )
        self.assertGreater(result.visit_counts[ToyComboState.shark], 0)
        self.assertTrue(
            any(
                ToyComboState.fly in target.actions
                and target.probabilities[target.actions.index(ToyComboState.fly)]
                > 0.5
                for target in result.targets
            )
        )

    def test_same_seed_is_reproducible(self):
        config = PUCTConfig(
            max_simulations=32,
            max_simulator_transitions=100,
            max_rollout_steps=8,
            horizon_battles=1,
            world_pool_size=4,
            min_root_visits=1,
            cpuct=2.0,
            prior_floor=0.3,
            leaf_value_weight=0.0,
            validation_rollouts=4,
            confidence_z=0.0,
        )
        first = StochasticPUCTSearch(ToyComboEvaluator(), config).search(
            ToyComboState(), torch.zeros(2), seed=99
        )
        second = StochasticPUCTSearch(ToyComboEvaluator(), config).search(
            ToyComboState(), torch.zeros(2), seed=99
        )
        self.assertEqual(first.proposed_action, second.proposed_action)
        self.assertEqual(first.visit_counts, second.visit_counts)
        self.assertEqual(first.q_values, second.q_values)
        self.assertEqual(
            first.simulator_transitions, second.simulator_transitions
        )

    def test_selection_only_uses_tree_proposal_without_confirmation_work(self):
        config = PUCTConfig(
            max_simulations=256,
            max_simulator_transitions=900,
            max_tree_depth=8,
            max_rollout_steps=8,
            horizon_battles=1,
            world_pool_size=8,
            min_root_visits=2,
            cpuct=3.0,
            prior_floor=0.4,
            leaf_value_weight=0.0,
            selection_only=True,
        )
        result = StochasticPUCTSearch(ToyComboEvaluator(), config).search(
            ToyComboState(), torch.zeros(2), seed=77
        )
        self.assertEqual(result.proposed_action, result.tree_proposed_action)
        self.assertEqual(result.proposed_action, ToyComboState.shark)
        self.assertFalse(result.accepted)
        self.assertFalse(result.candidate_eligible)
        self.assertEqual(result.validation_rollouts_used, 0)
        self.assertIsNone(result.audit_return_improvement)
        self.assertEqual(result.screening_wall_seconds, 0.0)
        self.assertEqual(result.validation_wall_seconds, 0.0)
        self.assertEqual(result.audit_wall_seconds, 0.0)
        self.assertEqual(result.screening_simulator_transitions, 0)
        self.assertEqual(result.validation_simulator_transitions, 0)
        self.assertEqual(result.audit_simulator_transitions, 0)

    def test_external_confirmation_mode_keeps_screening_but_skips_gates(self):
        config = PUCTConfig(
            max_simulations=256,
            max_simulator_transitions=900,
            max_tree_depth=8,
            max_rollout_steps=8,
            horizon_battles=1,
            world_pool_size=8,
            min_root_visits=2,
            cpuct=3.0,
            prior_floor=0.4,
            leaf_value_weight=0.0,
            external_confirmation_only=True,
            root_screening_top_k=2,
            root_screening_rollouts=2,
            root_screening_rounds=1,
            root_screening_min_survivors=2,
        )
        result = StochasticPUCTSearch(ToyComboEvaluator(), config).search(
            ToyComboState(), torch.zeros(2), seed=77
        )
        self.assertEqual(result.proposed_action, ToyComboState.shark)
        self.assertFalse(result.accepted)
        self.assertFalse(result.candidate_eligible)
        self.assertGreater(result.screening_simulator_transitions, 0)
        self.assertEqual(result.validation_simulator_transitions, 0)
        self.assertEqual(result.audit_simulator_transitions, 0)

    def test_puct_proposal_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "exclusive"):
            StochasticPUCTSearch(
                ToyComboEvaluator(),
                PUCTConfig(
                    selection_only=True,
                    external_confirmation_only=True,
                ),
            )


if __name__ == "__main__":
    unittest.main()
