import unittest

import numpy as np

from sapai import Player, Shop, Team
from sapai.shop import get_shop_rules
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    EntityObservationEncoder,
    FlatActionCodec,
    MAX_SHOP_SLOTS,
)


class TestCurrentEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = SuperAutoPetsEnv(valid_actions_only=True)
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def test_observation_contains_current_turtle_content(self):
        observation, _ = self.env.reset(seed=1)
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertIn("pet-pigeon", self.env.ALL_PETS)
        self.assertIn("pet-armadillo", self.env.ALL_PETS)
        self.assertIn("pet-wolverine", self.env.ALL_PETS)
        self.assertIn("food-cake", self.env.ALL_FOODS)
        self.assertIn("food-bread", self.env.ALL_FOODS)
        self.assertIn("food-bread-crumbs", self.env.ALL_FOODS)
        self.assertFalse(self.env.agent.record_actions)
        self.assertFalse(self.env.opponent.record_actions)

    def test_opponent_history_preserves_two_public_boards_and_temp_flags(self):
        older = Team(["ant", "fish"])
        latest = Team(["camel", "peacock"])
        latest[0].pet._until_end_of_battle_attack_buff = 3
        latest[1].pet._until_end_of_battle_health_buff = 3
        self.env.previous_previous_opponent_state = older
        self.env.previous_opponent_state = latest

        observation = EntityObservationEncoder().encode(self.env, 0)
        history = observation["opponent_history_numeric"]
        self.assertEqual(
            observation["opponent_history_board_mask"].tolist(), [1, 1]
        )
        self.assertEqual(
            observation["opponent_history_mask"].sum(axis=1).tolist(), [2, 2]
        )
        self.assertNotEqual(
            observation["opponent_history_item_ids"][0, 0],
            observation["opponent_history_item_ids"][1, 0],
        )
        self.assertEqual(history[0, 0, 7], 0.5)
        self.assertEqual(history[1, 0, 7], 1.0)
        self.assertEqual(history[1, 0, 5], 1.0)
        self.assertEqual(history[1, 0, 6], 0.0)
        self.assertEqual(history[1, 1, 5], 0.0)
        self.assertEqual(history[1, 1, 6], 1.0)

        # Only the public presence of temporary attack/health is exposed; the
        # exact temporary amount is deliberately not a separate feature.
        latest[0].pet._until_end_of_battle_attack_buff = 12
        changed = EntityObservationEncoder().encode(self.env, 0)
        self.assertEqual(
            changed["opponent_history_numeric"][1, 0, 5], 1.0
        )

    def test_action_ids_use_physical_shop_slots(self):
        self.env.agent.shop = Shop(["ant", "fish", "apple"])
        self.env.agent.team = Team(["ant"])
        self.env.agent.gold = 10
        actions = self.env._avail_actions(agent_idx=0)

        ant_to_team_zero = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_PET, 0, 0)
        )
        fish_to_team_one = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_PET, 1, 1)
        )
        apple_for_team_zero = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_FOOD, 2, 0)
        )
        self.assertEqual(actions[ant_to_team_zero][1:], (0, 0))
        self.assertEqual(actions[fish_to_team_one][1:], (1, 1))
        self.assertEqual(actions[apple_for_team_zero][1:], (2, 0))

    def test_chocolate_cannot_target_a_level_three_pet(self):
        self.env.agent = Player(
            shop=Shop(["chocolate"], turn=9),
            team=Team(["fish", "crab"]),
            turn=9,
            gold=3,
            record_actions=False,
        )
        self.env.agent.team[0].pet.set_total_experience(5)
        self.env.agent.team[1].pet.set_total_experience(4)
        self.env.shop_agent = self.env.agent.shop
        self.env.team_agent = self.env.agent.team
        self.env.turn = 9
        self.env._agent_actions_cache = None

        level_three = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_FOOD, 0, 0)
        )
        level_two = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_FOOD, 0, 1)
        )
        actions = self.env._avail_actions(0)
        self.assertNotIn(level_three, actions)
        self.assertIn(level_two, actions)

        observation = EntityObservationEncoder().encode(self.env, 0)
        self.assertFalse(
            observation["target_mask"][ActionType.BUY_FOOD, 0, 0]
        )
        self.assertTrue(
            observation["target_mask"][ActionType.BUY_FOOD, 0, 1]
        )

    def test_second_chocolate_target_disappears_after_reaching_level_three(self):
        self.env.agent = Player(
            shop=Shop(["chocolate", "chocolate"], turn=9),
            team=Team(["fish"]),
            turn=9,
            gold=6,
            record_actions=False,
        )
        self.env.agent.team[0].pet.set_total_experience(4)
        self.env.shop_agent = self.env.agent.shop
        self.env.team_agent = self.env.agent.team
        self.env.turn = 9
        self.env._agent_actions_cache = None

        first = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_FOOD, 0, 0)
        )
        self.assertIn(first, self.env._avail_actions(0))
        self.env.resolve_action(0, first)
        self.assertEqual(self.env.agent.team[0].pet.level, 3)

        remaining_chocolate_slots = [
            index
            for index, slot in enumerate(self.env.agent.shop)
            if slot.slot_type == "food"
            and slot.obj.name == "food-chocolate"
        ]
        self.assertEqual(len(remaining_chocolate_slots), 1)
        second = FlatActionCodec.encode(
            AutoregressiveAction(
                ActionType.BUY_FOOD, remaining_chocolate_slots[0], 0
            )
        )
        self.assertNotIn(second, self.env._avail_actions(0))
        observation = EntityObservationEncoder().encode(self.env, 0)
        self.assertFalse(
            observation["target_mask"][
                ActionType.BUY_FOOD, remaining_chocolate_slots[0], 0
            ]
        )

    def test_legal_action_cache_reuses_state_and_invalidates_after_action(self):
        self.env.agent.gold = 1
        self.env._agent_actions_cache = None
        first = self.env._avail_actions(0)
        second = self.env._avail_actions(0)
        self.assertIs(first, second)
        self.assertIn(FlatActionCodec.ROLL, first)

        self.env.resolve_action(0, FlatActionCodec.ROLL)
        self.assertIsNone(self.env._agent_actions_cache)
        after = self.env._avail_actions(0)
        self.assertIsNot(after, first)
        self.assertNotIn(FlatActionCodec.ROLL, after)

    def test_tier_two_one_food_slot_observation_and_mask(self):
        self.env.agent.shop = Shop(
            turn=3,
            seed_state=np.random.RandomState(303).get_state(),
        )
        self.env.agent.team = Team(["ant"])
        self.env.agent.gold = 10
        self.env._agent_actions_cache = None

        food_indices = [
            index
            for index, slot in enumerate(self.env.agent.shop)
            if slot.slot_type == "food"
        ]
        self.assertEqual(len(self.env.agent.shop), 4)
        self.assertEqual(food_indices, [3])

        observation = EntityObservationEncoder().encode(self.env, 0)
        legal_food_sources = list(
            np.flatnonzero(observation["source_mask"][ActionType.BUY_FOOD])
        )
        self.assertEqual(legal_food_sources, food_indices)
        first_shop_token = 1 + self.env.MAX_TEAM_PETS
        self.assertEqual(observation["item_ids"].shape, (21,))
        self.assertTrue(observation["entity_mask"].all())
        self.assertNotEqual(
            observation["item_ids"][first_shop_token + food_indices[0]],
            0,
        )
        self.assertTrue(
            np.all(
                observation["item_ids"][
                    first_shop_token + len(self.env.agent.shop) :
                    first_shop_token + MAX_SHOP_SLOTS
                ]
                == 0
            )
        )

    def test_sell_and_last_gold_roll_are_legal(self):
        self.env.agent.team = Team(["ant"])
        self.env.agent.gold = 1
        actions = self.env._avail_actions(agent_idx=0)
        self.assertIn(
            FlatActionCodec.encode(
                AutoregressiveAction(ActionType.SELL, 0)
            ),
            actions,
        )
        self.assertIn(FlatActionCodec.ROLL, actions)

    def test_freeze_unfreeze_and_move_are_exposed(self):
        freeze = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.FREEZE, 0)
        )
        self.assertIn(freeze, self.env._avail_actions(0))
        self.env.resolve_action(0, freeze)
        unfreeze = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.UNFREEZE, 0)
        )
        self.assertNotIn(unfreeze, self.env._avail_actions(0))
        self.env.agent.gold = max(self.env.agent.gold, 1)
        self.env.resolve_action(0, FlatActionCodec.ROLL)
        self.assertIn(unfreeze, self.env._avail_actions(0))

        self.env.agent.team = Team(["ant", "fish"])
        move = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.MOVE, 0, 1)
        )
        self.assertIn(move, self.env._avail_actions(0))
        self.env.resolve_action(0, move)
        self.assertEqual(self.env.agent.team[0].pet.name, "pet-fish")
        self.assertEqual(self.env.agent.team[1].pet.name, "pet-ant")

    def test_zero_gold_frozen_triple_copy_survives_next_turn_and_is_buyable(self):
        self.env.agent = Player(
            shop=Shop(
                ["ant", "apple", "fish", "beaver"],
                turn=2,
                seed_state=np.random.RandomState(909).get_state(),
                fixed_rules=True,
            ),
            team=Team(["fish"]),
            turn=2,
            gold=0,
            record_actions=False,
        )
        self.env.agent.team[0].pet.set_total_experience(1)
        self.env.shop_agent = self.env.agent.shop
        self.env.team_agent = self.env.agent.team
        self.env.turn = 2
        self.env._agent_actions_cache = None
        self.env._freeze_toggled_agent.clear()

        original_slot = self.env.agent.shop.slots[2]
        original_pet = original_slot.obj
        freeze = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.FREEZE, 2)
        )
        self.assertIn(freeze, self.env._avail_actions(0))
        self.env.resolve_action(0, freeze)
        self.assertTrue(original_slot.frozen)

        end = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.END_TURN)
        )
        self.env.resolve_action(0, end)
        self.env.agent.start_turn(winner=True)
        self.env.turn = 3
        self.env._agent_actions_cache = None
        self.env._freeze_toggled_agent.clear()

        self.assertIs(self.env.agent.shop.slots[0], original_slot)
        self.assertIs(self.env.agent.shop.slots[0].obj, original_pet)
        self.assertTrue(self.env.agent.shop.slots[0].frozen)
        observation = EntityObservationEncoder().encode(self.env, 0)
        first_shop_token = 1 + self.env.MAX_TEAM_PETS
        self.assertEqual(observation["numeric"][first_shop_token, 6], 1.0)
        buy_combine = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_PET, 0, 0)
        )
        self.assertIn(buy_combine, self.env._avail_actions(0))

    def test_turn_two_matching_pet_can_be_bought_separate_or_combined(self):
        self.env.agent = Player(
            shop=Shop(["ant", "fish", "mosquito", "apple"], turn=2),
            team=Team(["ant", "fish"]),
            turn=2,
            gold=3,
            record_actions=False,
        )
        self.env.shop_agent = self.env.agent.shop
        self.env.team_agent = self.env.agent.team
        self.env.turn = 2
        self.env._agent_actions_cache = None

        combine = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_PET, 0, 0)
        )
        separate = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_PET, 0, 2)
        )
        actions = self.env._avail_actions(0)
        self.assertIn(combine, actions)
        self.assertIn(separate, actions)

        self.env.resolve_action(0, separate)
        ants = [
            slot.pet for slot in self.env.agent.team
            if not slot.empty and slot.pet.name == "pet-ant"
        ]
        self.assertEqual(len(ants), 2)
        self.assertTrue(all(pet.level == 1 and pet.experience == 0 for pet in ants))

    def test_move_inserts_at_destination_and_shifts_intervening_pets(self):
        self.env.agent.team = Team(["ant", "fish", "crab"])
        self.env._seen_team_orders_agent = {
            self.env._team_order_key(self.env.agent)
        }
        action = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.MOVE, 0, 2)
        )
        self.env.resolve_action(0, action)
        self.assertEqual(
            [
                slot.pet.name
                for slot in self.env.agent.team
                if not slot.empty
            ],
            ["pet-fish", "pet-crab", "pet-ant"],
        )

    def test_linked_levelup_choices_are_observable_legal_and_mutually_exclusive(self):
        self.env.agent.shop = Shop(
            ["fish"],
            turn=3,
            seed_state=np.random.RandomState(23).get_state(),
        )
        self.env.agent.team = Team(["fish"])
        self.env.agent.gold = 10
        self.env.agent.shop.levelup()
        self.env._agent_actions_cache = None

        actions = self.env._avail_actions(0)
        for shop_index in (0, 1):
            buy = FlatActionCodec.encode(
                AutoregressiveAction(ActionType.BUY_PET, shop_index, 1)
            )
            self.assertIn(buy, actions)

        observation = EntityObservationEncoder().encode(self.env, 0)
        first_token = 1 + self.env.MAX_TEAM_PETS
        first = observation["numeric"][first_token]
        second = observation["numeric"][first_token + 1]
        self.assertEqual(first[11], 1.0)
        self.assertEqual(second[11], 1.0)
        self.assertEqual(first[12], second[12])
        self.assertEqual((first[13], second[13]), (0.0, 1.0))

        freeze = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.FREEZE, 0)
        )
        self.env.resolve_action(0, freeze)
        self.assertTrue(self.env.agent.shop[0].frozen)
        self.assertFalse(self.env.agent.shop[1].frozen)

        # The sibling remains independently freezeable, while the selected
        # choice cannot be immediately toggled back in the same shop state.
        actions = self.env._avail_actions(0)
        sibling_freeze = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.FREEZE, 1)
        )
        selected_unfreeze = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.UNFREEZE, 0)
        )
        self.assertIn(sibling_freeze, actions)
        self.assertNotIn(selected_unfreeze, actions)
        observation = EntityObservationEncoder().encode(self.env, 0)
        self.assertEqual(observation["numeric"][first_token, 6], 1.0)
        self.assertEqual(observation["numeric"][first_token + 1, 6], 0.0)

        self.env._agent_actions_cache = None
        buy = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.BUY_PET, 1, 1)
        )
        self.env.resolve_action(0, buy)
        self.assertFalse(
            any(
                slot.slot_type == "levelup"
                for slot in self.env.agent.shop
            )
        )

    def test_rolling_one_frozen_discover_removes_only_its_sibling(self):
        self.env.agent.shop = Shop(
            turn=3,
            seed_state=np.random.RandomState(231).get_state(),
        )
        self.env.agent.shop.levelup()
        first, second = self.env.agent.shop.slots[:2]
        self.env.agent.gold = 10
        self.env._agent_actions_cache = None

        self.env.resolve_action(
            0,
            FlatActionCodec.encode(
                AutoregressiveAction(ActionType.FREEZE, 0)
            ),
        )
        self.env.resolve_action(0, FlatActionCodec.ROLL)

        self.assertIn(first, self.env.agent.shop.slots)
        self.assertNotIn(second, self.env.agent.shop.slots)
        self.assertTrue(first.frozen)
        self.assertEqual(
            len(self.env.agent.shop),
            sum(get_shop_rules(3)[:2]),
        )
        observation = EntityObservationEncoder().encode(self.env, 0)
        first_shop_token = 1 + self.env.MAX_TEAM_PETS
        self.assertEqual(
            observation["numeric"][first_shop_token, 11], 1.0
        )
        self.assertEqual(
            int(observation["type_mask"][ActionType.UNFREEZE]), 1
        )

    def test_two_frozen_discovers_stay_linked_and_legal_after_roll(self):
        self.env.agent.shop = Shop(
            turn=3,
            seed_state=np.random.RandomState(232).get_state(),
        )
        self.env.agent.shop.levelup()
        self.env.agent.team = Team(["fish"])
        self.env.agent.gold = 10
        self.env._agent_actions_cache = None
        linked_group = self.env.agent.shop[0].reward_group

        for source in (0, 1):
            freeze = FlatActionCodec.encode(
                AutoregressiveAction(ActionType.FREEZE, source)
            )
            self.assertIn(freeze, self.env._avail_actions(0))
            self.env.resolve_action(0, freeze)
        self.env.resolve_action(0, FlatActionCodec.ROLL)

        rewards = [
            slot
            for slot in self.env.agent.shop
            if slot.reward_group == linked_group
        ]
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(slot.frozen for slot in rewards))
        observation = EntityObservationEncoder().encode(self.env, 0)
        actions = self.env._avail_actions(0)
        for source in (0, 1):
            buy = FlatActionCodec.encode(
                AutoregressiveAction(ActionType.BUY_PET, source, 1)
            )
            self.assertIn(buy, actions)
            self.assertTrue(
                observation["target_mask"][
                    ActionType.BUY_PET, source, 1
                ]
            )

        self.env.resolve_action(
            0,
            FlatActionCodec.encode(
                AutoregressiveAction(ActionType.BUY_PET, 1, 1)
            ),
        )
        self.assertFalse(
            any(
                slot.reward_group == linked_group
                for slot in self.env.agent.shop
            )
        )

    def test_levelup_overflow_preserves_ten_token_observation_and_masks(self):
        self.env.agent.shop = Shop(
            [
                "ant", "fish", "pig", "duck", "crab", "apple", "pear",
                "sushi", "pizza", "honey",
            ],
            turn=3,
            seed_state=np.random.RandomState(29).get_state(),
        )
        self.env.agent.shop.slots[-1].freeze()
        frozen = self.env.agent.shop.slots[-1]
        self.env.agent.team = Team(["fish"])
        self.env.agent.gold = 10
        self.env.agent.shop.levelup()
        self.env._agent_actions_cache = None

        self.assertEqual(len(self.env.agent.shop), 10)
        self.assertIn(frozen, self.env.agent.shop.slots)
        observation = EntityObservationEncoder().encode(self.env, 0)
        self.assertEqual(observation["item_ids"].shape, (21,))
        first_shop_token = 1 + self.env.MAX_TEAM_PETS
        self.assertEqual(
            observation["numeric"][first_shop_token, 11], 1.0
        )
        self.assertEqual(
            observation["numeric"][first_shop_token + 1, 11], 1.0
        )

        for shop_index in (0, 1):
            action = AutoregressiveAction(
                ActionType.BUY_PET, shop_index, 1
            )
            flat = FlatActionCodec.encode(action)
            self.assertIn(flat, self.env._avail_actions(0))
            self.assertTrue(
                observation["target_mask"][
                    ActionType.BUY_PET, shop_index, 1
                ]
            )

    def test_end_turn_does_not_return_an_ended_agent(self):
        self.env.agent.gold = 0
        observation, _, _, _, _ = self.env.step(
            self.env.ACTION_BASE_NUM["end_turn"]
        )
        self.assertFalse(self.env.agent.turn_ended)
        self.assertTrue(self.env.observation_space.contains(observation))

    def test_end_turn_requires_spending_all_gold(self):
        end_turn = self.env.ACTION_BASE_NUM["end_turn"]
        roll = self.env.ACTION_BASE_NUM["roll"]

        self.env.agent.gold = 2
        actions = self.env._avail_actions(0)
        self.assertNotIn(end_turn, actions)
        self.assertIn(roll, actions)

        self.env.resolve_action(0, roll)
        self.assertEqual(self.env.agent.gold, 1)
        self.assertNotIn(end_turn, self.env._avail_actions(0))
        self.env.resolve_action(0, roll)
        self.assertEqual(self.env.agent.gold, 0)
        self.assertIn(end_turn, self.env._avail_actions(0))

        self.env.agent.gold = 10
        self.env.shop_action_count_agent = 50
        self.assertEqual(
            set(self.env._avail_actions(0)),
            {end_turn},
        )

    def test_reset_seed_reproduces_independent_shop_streams(self):
        first = SuperAutoPetsEnv(manual_battles=True)
        second = SuperAutoPetsEnv(manual_battles=True)
        first.reset(seed=2028)
        second.reset(seed=2028)

        def names(shop):
            return tuple(slot.obj.name for slot in shop)

        self.assertEqual(names(first.agent.shop), names(second.agent.shop))
        self.assertEqual(
            names(first.opponent.shop), names(second.opponent.shop)
        )
        self.assertNotEqual(names(first.agent.shop), names(first.opponent.shop))

        opponent_before = first.opponent.shop.state
        first.agent.roll()
        self.assertEqual(first.opponent.shop.state, opponent_before)
        second.agent.roll()
        self.assertEqual(names(first.agent.shop), names(second.agent.shop))

        first.opponent.roll()
        second.opponent.roll()
        self.assertEqual(
            names(first.opponent.shop), names(second.opponent.shop)
        )

    def test_move_mask_blocks_revisited_orders_and_caps_at_two(self):
        self.env.agent.team = Team(["ant", "fish", "crab"])
        self.env.reorder_count_agent = 0
        self.env._seen_team_orders_agent = {
            self.env._team_order_key(self.env.agent)
        }
        first = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.MOVE, 0, 1)
        )
        self.assertIn(first, self.env._avail_actions(0))
        self.env.resolve_action(0, first)

        # Swapping the same two physical pets back would revisit the original
        # arrangement and is therefore dominated/illegal.
        self.assertNotIn(first, self.env._avail_actions(0))
        second = FlatActionCodec.encode(
            AutoregressiveAction(ActionType.MOVE, 0, 2)
        )
        self.assertIn(second, self.env._avail_actions(0))
        self.env.resolve_action(0, second)
        self.assertEqual(self.env.reorder_count_agent, 2)
        self.assertFalse(
            any(
                FlatActionCodec.decode(action).action_type == ActionType.MOVE
                for action in self.env._avail_actions(0)
            )
        )

    def test_move_history_resets_with_environment(self):
        self.env.agent.team = Team(["ant", "fish"])
        self.env.reorder_count_agent = 2
        self.env._seen_team_orders_agent = {
            self.env._team_order_key(self.env.agent)
        }
        self.env.reset(seed=818)
        self.assertEqual(self.env.reorder_count_agent, 0)
        self.assertEqual(
            self.env._seen_team_orders_agent,
            {self.env._team_order_key(self.env.agent)},
        )

    def test_last_battle_loss_survives_start_turn_for_snail(self):
        self.env.agent = Player(
            shop=self.env.agent.shop,
            team=Team(["ant", "snail"]),
            lf_winner=False,
        )
        self.env.agent.start_turn(winner=self.env.agent.lf_winner)
        self.assertIs(self.env.agent.lf_winner, False)
        self.env.agent.end_turn()
        self.assertEqual(self.env.agent.team[0].pet.attack, 3)

        self.env.agent.last_battle_outcome = -1
        observation = EntityObservationEncoder().encode(self.env, 0)
        self.assertEqual(observation["numeric"][0, 1], -1.0)
        self.assertEqual(observation["numeric"][0, 2], 1.0)

    def test_global_entity_exposes_public_opponent_score_by_perspective(self):
        self.env.agent.lives = 4
        self.env.agent.wins = 3
        self.env.opponent.lives = 2
        self.env.opponent.wins = 7
        encoder = EntityObservationEncoder()

        agent = encoder.encode(self.env, 0)["numeric"][0]
        opponent = encoder.encode(self.env, 1)["numeric"][0]
        self.assertAlmostEqual(agent[9], 2 / 6)
        self.assertAlmostEqual(agent[10], 7 / 10)
        self.assertAlmostEqual(agent[13], 4 / 6)
        self.assertAlmostEqual(agent[14], 3 / 10)
        self.assertAlmostEqual(opponent[9], 4 / 6)
        self.assertAlmostEqual(opponent[10], 3 / 10)
        self.assertAlmostEqual(opponent[13], 2 / 6)
        self.assertAlmostEqual(opponent[14], 7 / 10)

    def test_entity_observation_exposes_action_and_toggle_history(self):
        self.env.agent.team = Team(["ant", "fish"])
        self.env._seen_team_orders_agent = {
            self.env._team_order_key(self.env.agent)
        }
        self.env.reorder_count_agent = 1
        self.env.shop_action_count_agent = 17
        toggled_object = self.env.agent.shop[1].obj
        self.env._freeze_toggled_agent = {id(toggled_object)}

        observation = EntityObservationEncoder().encode(self.env, 0)
        global_features = observation["numeric"][0]
        self.assertEqual(global_features[0], 0.5)
        self.assertEqual(global_features[5], 17 / 50)
        self.assertEqual(global_features[6], 0.1)
        self.assertEqual(global_features[7], 1 / 3)

        first_shop_token = 1 + self.env.MAX_TEAM_PETS
        self.assertEqual(observation["numeric"][first_shop_token, 8], 0.0)
        self.assertEqual(observation["numeric"][first_shop_token + 1, 8], 1.0)

    def test_turn_two_restores_one_heart_but_preserves_snail_loss(self):
        self.env.agent = Player(
            shop=Shop([]), team=Team(["ant", "snail"]), lives=6
        )
        self.env.opponent = Player(shop=Shop([]), team=Team(["fish"]), lives=6)
        self.env.turn = 2

        self.env._player_fight_outcome(1)
        self.assertEqual((self.env.agent.lives, self.env.opponent.lives), (6, 6))
        self.assertIs(self.env.agent.lf_winner, False)
        self.assertEqual(self.env.agent.last_battle_outcome, -1)

        self.env.agent.end_turn()
        self.assertEqual(self.env.agent.team[0].pet.attack, 3)

        self.env.agent.lives = 5
        self.env.opponent.lives = 5
        self.env._player_fight_outcome(1)
        self.assertEqual((self.env.agent.lives, self.env.opponent.lives), (5, 6))

    def test_turn_one_loss_does_not_restore_a_heart(self):
        self.env.turn = 1
        self.env.agent.lives = 6
        self.env.opponent.lives = 6
        self.env._player_fight_outcome(1)
        self.assertEqual((self.env.agent.lives, self.env.opponent.lives), (5, 6))

    def test_masked_random_rollout(self):
        observation, _ = self.env.reset(seed=7)
        for _ in range(250):
            valid = np.flatnonzero(self.env.action_masks())
            action = int(self.env.rs_random.choice(valid))
            observation, _, terminated, truncated, _ = self.env.step(action)
            self.assertTrue(self.env.observation_space.contains(observation))
            self.assertFalse(self.env.agent.turn_ended)
            if terminated or truncated:
                observation, _ = self.env.reset()

    def test_seeded_evaluation_is_reproducible_but_players_are_independent(self):
        def shops(seed):
            self.env.reset(seed=seed)
            agent = [slot.obj.name for slot in self.env.agent.shop]
            opponent = [slot.obj.name for slot in self.env.opponent.shop]
            return agent, opponent

        first = shops(1234)
        second = shops(1234)
        different_seed = shops(4321)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different_seed)
        self.assertNotEqual(first[0], first[1])


if __name__ == "__main__":
    unittest.main()
