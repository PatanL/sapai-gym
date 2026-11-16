import unittest

import numpy as np

from sapai_gym1.rl import ActionType, FlatActionCodec
from sapai_gym1.rl.curriculum import (
    BUILDERS,
    CURRICULUM_VERSION,
    PUCT_SCENARIO_NAMES,
    PUCT_SCENARIO_VERSION,
    SCENARIO_NAMES,
    SEARCH_ONLY_SCENARIO_NAMES,
    generate_curriculum_examples,
    scenario_counts,
)
from sapai_gym1.rl.guide_teacher import GUIDE_VERSION


class CurriculumTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = generate_curriculum_examples(
            examples_per_scenario=2, seed=123
        )

    def test_all_scenarios_generate_exact_versioned_examples(self):
        counts = scenario_counts(self.examples)
        self.assertEqual(set(counts), set(SCENARIO_NAMES))
        for example in self.examples:
            self.assertEqual(
                example["validation"]["curriculumVersion"],
                CURRICULUM_VERSION,
            )
            self.assertTrue(example["validation"]["exactSimulatorState"])
            self.assertEqual(
                example["validation"]["guideVersion"],
                GUIDE_VERSION,
            )
            self.assertEqual(len(example["validation"]["guideSha256"]), 64)
            self.assertEqual(
                len(example["validation"]["executableTeacherSha256"]), 64
            )

    def test_every_label_is_legal_in_its_recorded_observation(self):
        for example in self.examples:
            action_type, source, target = example["action"]
            observation = example["observation"]
            self.assertEqual(observation["type_mask"][action_type], 1)
            self.assertEqual(
                observation["source_mask"][action_type][source], 1
            )
            self.assertEqual(
                observation["target_mask"][action_type][source][target], 1
            )

    def test_triple_sequence_freezes_before_spending_gold(self):
        triple = [
            example
            for example in self.examples
            if example["teacher"]["scenario"] == "triple-freeze"
        ]
        by_replay = {}
        for example in triple:
            by_replay.setdefault(example["replay_id"], []).append(example)
        for sequence in by_replay.values():
            sequence.sort(key=lambda item: item["decision"])
            self.assertEqual(
                sequence[0]["action"][0], int(ActionType.FREEZE)
            )
            self.assertEqual(
                sequence[-1]["action"][0], int(ActionType.END_TURN)
            )

    def test_named_combo_targets_are_present(self):
        first_actions = {}
        for example in self.examples:
            if example["decision"] == 0:
                first_actions.setdefault(
                    example["teacher"]["scenario"], example["action"][0]
                )
        self.assertEqual(first_actions["shark-fly"], int(ActionType.BUY_PET))
        self.assertEqual(
            first_actions["turn-two-separate-pair"],
            int(ActionType.BUY_PET),
        )
        self.assertEqual(
            first_actions["pill-flamingo"], int(ActionType.BUY_FOOD)
        )
        self.assertEqual(
            first_actions["pill-turtle"], int(ActionType.BUY_FOOD)
        )
        self.assertEqual(
            first_actions["avoid-pill-no-faint"], int(ActionType.ROLL)
        )
        self.assertEqual(
            first_actions["avoid-freeze-level-three"],
            int(ActionType.END_TURN),
        )
        self.assertEqual(
            first_actions["avoid-freeze-obsolete-pair"],
            int(ActionType.END_TURN),
        )
        self.assertEqual(
            first_actions["mushroom-deer-shark"], int(ActionType.BUY_FOOD)
        )
        self.assertEqual(
            first_actions["elephant-hurt-wolverine"], int(ActionType.BUY_PET)
        )
        self.assertEqual(
            first_actions["scorpion-vs-tall"], int(ActionType.SELL)
        )

    def test_pill_examples_apply_value_and_negative_example_avoids_pill(self):
        rng = np.random.default_rng(701)

        flamingo, actions = BUILDERS["pill-flamingo"](rng)
        before = [
            (
                flamingo.agent.team[index].pet.attack,
                flamingo.agent.team[index].pet.health,
            )
            for index in (1, 2)
        ]
        flamingo.resolve_action(0, FlatActionCodec.encode(actions[0]))
        self.assertTrue(flamingo.agent.team[0].empty)
        for offset, index in enumerate((1, 2)):
            pet = flamingo.agent.team[index].pet
            self.assertEqual(pet.attack, before[offset][0] + 1)
            self.assertEqual(pet.health, before[offset][1] + 1)

        turtle, actions = BUILDERS["pill-turtle"](rng)
        turtle.resolve_action(0, FlatActionCodec.encode(actions[0]))
        self.assertTrue(turtle.agent.team[0].empty)
        self.assertEqual(
            turtle.agent.team[1].pet.status, "status-melon-armor"
        )

        avoid, actions = BUILDERS["avoid-pill-no-faint"](rng)
        self.assertEqual(actions[0].action_type, ActionType.ROLL)
        names_before = [
            None if slot.empty else slot.pet.name for slot in avoid.agent.team
        ]
        avoid.resolve_action(0, FlatActionCodec.encode(actions[0]))
        self.assertEqual(
            [None if slot.empty else slot.pet.name for slot in avoid.agent.team],
            names_before,
        )

    def test_search_only_scenarios_are_versioned_and_have_legal_roots(self):
        self.assertTrue(PUCT_SCENARIO_VERSION)
        self.assertEqual(
            set(PUCT_SCENARIO_NAMES),
            set(SCENARIO_NAMES) | set(SEARCH_ONLY_SCENARIO_NAMES),
        )
        for scenario_index, scenario in enumerate(SEARCH_ONLY_SCENARIO_NAMES):
            env, actions = BUILDERS[scenario](
                np.random.default_rng(9000 + scenario_index)
            )
            self.assertTrue(actions, scenario)
            self.assertIn(
                FlatActionCodec.encode(actions[0]),
                env._avail_actions(0),
                scenario,
            )

    def test_all_puct_manifest_seed_shapes_have_legal_reference_roots(self):
        for scenario_index, scenario in enumerate(PUCT_SCENARIO_NAMES):
            for example in range(12):
                seed = 3605800 + scenario_index * 100_000 + example
                env, actions = BUILDERS[scenario](
                    np.random.default_rng(seed)
                )
                self.assertTrue(actions, (scenario, seed))
                self.assertIn(
                    FlatActionCodec.encode(actions[0]),
                    env._avail_actions(0),
                    (scenario, seed),
                )

    def test_search_only_scenarios_include_both_sides_of_nuanced_rules(self):
        rng = np.random.default_rng(9010)
        delay, delay_actions = BUILDERS["search-turn-two-delay-triple"](rng)
        self.assertEqual(delay_actions[0].action_type, ActionType.ROLL)
        self.assertTrue(
            any(
                action.action_type == ActionType.COMBINE
                for action in map(FlatActionCodec.decode, delay._avail_actions(0))
            )
        )

        space, space_actions = BUILDERS["search-turn-two-triple-for-space"](rng)
        self.assertEqual(space_actions[0].action_type, ActionType.COMBINE)
        self.assertEqual(len(space.agent.team), 5)

        preserve, preserve_actions = BUILDERS["search-preserve-active-snails"](rng)
        self.assertEqual(preserve_actions[0].action_type, ActionType.ROLL)
        self.assertLess(len(preserve.agent.team), 5)

        combine, combine_actions = BUILDERS["search-combine-snails-for-space"](rng)
        self.assertEqual(combine_actions[0].action_type, ActionType.COMBINE)
        self.assertEqual(len(combine.agent.team), 5)

        freeze, freeze_actions = BUILDERS["search-premium-subcost-freeze"](rng)
        self.assertLess(freeze.agent.gold, 3)
        self.assertEqual(freeze_actions[0].action_type, ActionType.FREEZE)

        buy, buy_actions = BUILDERS["search-buy-premium-affordable"](rng)
        self.assertEqual(buy.agent.gold, 3)
        self.assertEqual(buy_actions[0].action_type, ActionType.BUY_FOOD)

        frozen_food, frozen_food_actions = BUILDERS[
            "search-buy-frozen-premium"
        ](rng)
        self.assertTrue(frozen_food.agent.shop[3].frozen)
        self.assertEqual(
            frozen_food_actions[0].action_type, ActionType.BUY_FOOD
        )

        frozen_pet, frozen_pet_actions = BUILDERS[
            "search-buy-frozen-match"
        ](rng)
        self.assertTrue(frozen_pet.agent.shop[1].frozen)
        self.assertEqual(
            frozen_pet_actions[0].action_type, ActionType.BUY_PET
        )

        held, held_actions = BUILDERS[
            "search-turn-three-complete-held-triple"
        ](rng)
        self.assertEqual(held.agent.turn, 3)
        self.assertEqual(held_actions[0].action_type, ActionType.COMBINE)


if __name__ == "__main__":
    unittest.main()
