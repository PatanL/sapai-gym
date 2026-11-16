import hashlib
import unittest

import numpy as np

from sapai import Player, Shop, Team
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.rl import ActionType, AutoregressiveAction
from sapai_gym1.rl.guide_teacher import (
    GUIDE_PATH,
    GUIDE_SHA256,
    GUIDE_VERSION,
    LinePlan,
    TEACHER_SHA256,
    TurtleGuideTeacher,
)


class TestTurtleGuideTeacherV11(unittest.TestCase):
    def setUp(self):
        self.teacher = TurtleGuideTeacher(np.random.default_rng(123))

    def test_version_hash_matches_canonical_prose(self):
        self.assertEqual(GUIDE_VERSION, "turtle-p1-guide-v12-20260904")
        self.assertTrue(GUIDE_PATH.exists())
        self.assertEqual(
            GUIDE_SHA256,
            hashlib.sha256(GUIDE_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(self.teacher.name, GUIDE_VERSION)
        self.assertEqual(self.teacher.guide_sha256, GUIDE_SHA256)
        self.assertEqual(
            TEACHER_SHA256,
            hashlib.sha256(
                (GUIDE_PATH.parent / "sapai_gym1/rl/guide_teacher.py").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(self.teacher.executable_sha256, TEACHER_SHA256)

    def test_discover_freeze_prefers_one_choice_before_clogging_second(self):
        shop = Shop(["giraffe", "camel"], turn=4)
        for slot in shop:
            slot.slot_type = "levelup"
            slot.reward_group = 1
        player = Player(shop=shop, team=Team(["fish"]))
        player.gold = 0

        first_scores = [
            self.teacher._score_freeze(player, source)
            for source in range(2)
        ]
        preferred = int(np.argmax(first_scores))
        self.assertGreater(first_scores[preferred], 16.0)

        player.freeze(preferred)
        sibling = 1 - preferred
        second_score = self.teacher._score_freeze(player, sibling)
        self.assertLess(second_score, 16.0)

    def test_affordable_freeze_keeps_pacing_and_salad_exceptions(self):
        player = Player(
            shop=Shop(["ant"], turn=4),
            team=Team(["ant", "fish", "beaver", "horse"]),
            turn=4,
        )
        player.gold = 3
        ordinary = self.teacher._score_freeze(player, 0)
        self.assertLess(ordinary, 0.0)

        player.team[0].pet.set_total_experience(1)
        tier_up = self.teacher._score_freeze(player, 0)
        self.assertGreater(tier_up, ordinary + 10)

        player.turn = 5
        player.team = Team(["fish", "beaver", "horse"])
        salad_isolation = self.teacher._score_freeze(player, 0)
        self.assertGreater(salad_isolation, ordinary)

        player.turn = 7
        player.team = Team(
            ["fish", "beaver", "horse", "mosquito", "otter"]
        )
        late_full_board = self.teacher._score_freeze(player, 0)
        self.assertGreater(late_full_board, ordinary)

    def test_move_waits_for_zero_gold_unless_pill_is_available(self):
        player = Player(
            shop=Shop(["ant", "apple"], turn=6),
            team=Team(["fish", "snail"]),
            turn=6,
        )
        player.gold = 3
        early = self.teacher._score_move(player, 0, 1)
        player.gold = 0
        final = self.teacher._score_move(player, 0, 1)
        self.assertGreater(final, early + 10)

        player.gold = 3
        player.shop = Shop(
            ["sleeping-pill", "apple"], turn=6, fixed_rules=True
        )
        pill_setup = self.teacher._score_move(player, 0, 1)
        self.assertEqual(pill_setup, final)

    def test_line_plan_persists_after_a_temporary_piece_disappears(self):
        core = SuperAutoPetsEnv(manual_battles=True)
        core.agent = Player(
            shop=Shop([]), team=Team(["elephant", "blowfish", "wolverine"]), turn=6
        )
        first = self.teacher._update_plan(core, 0, core.agent, None)
        self.assertEqual(first.line_name, "elephant hurt engine")
        first_confidence = first.confidence

        core.agent.team = Team(["elephant", "ant", "wolverine"])
        second = self.teacher._update_plan(core, 0, core.agent, None)
        self.assertIs(first, second)
        self.assertEqual(second.line_name, "elephant hurt engine")
        self.assertGreaterEqual(second.confidence, first_confidence)
        self.assertTrue(second.required_hits)
        self.assertIsNotNone(second.sell_slot)

    def test_teacher_uses_last_observed_not_private_current_opponent(self):
        core = SuperAutoPetsEnv(manual_battles=True)
        core.previous_opponent_state = Team(["rhino"])
        core.opponent.team = Team(["ant"])
        self.assertIs(
            self.teacher._known_opponent_team(core, 0),
            core.previous_opponent_state,
        )

    def test_plan_tracks_public_opponent_score_by_perspective(self):
        core = SuperAutoPetsEnv(manual_battles=True)
        core.agent.lives = 5
        core.agent.wins = 2
        core.opponent.lives = 1
        core.opponent.wins = 4
        agent_plan = self.teacher._update_plan(
            core, 0, core.agent, None
        )
        opponent_plan = self.teacher._update_plan(
            core, 1, core.opponent, None
        )
        self.assertEqual(
            (agent_plan.opponent_lives, agent_plan.opponent_wins), (1, 4)
        )
        self.assertEqual(agent_plan.tempo_pressure, 2.0)
        self.assertEqual(
            (opponent_plan.opponent_lives, opponent_plan.opponent_wins),
            (5, 2),
        )
        self.assertEqual(opponent_plan.tempo_pressure, 0.0)

    def test_low_opponent_lives_favors_tempo_over_slow_scaling(self):
        player = Player(
            shop=Shop(["deer", "dragon", "pear", "canned-food"], turn=9),
            team=Team(["ant", "fish"]),
            turn=9,
        )
        neutral = LinePlan(
            line_name="generic stat-tech", tempo_pressure=0.0
        )
        pressured = LinePlan(
            line_name="generic stat-tech", tempo_pressure=2.0,
            opponent_lives=1,
        )
        self.assertGreater(
            self.teacher._item_value(player, "pet-deer", pressured),
            self.teacher._item_value(player, "pet-deer", neutral) + 4,
        )
        self.assertLess(
            self.teacher._item_value(player, "pet-dragon", pressured),
            self.teacher._item_value(player, "pet-dragon", neutral) - 2,
        )
        self.assertGreater(
            self.teacher._item_value(player, "food-pear", pressured),
            self.teacher._item_value(player, "food-pear", neutral) + 4,
        )
        self.assertLess(
            self.teacher._item_value(player, "food-canned-food", pressured),
            self.teacher._item_value(player, "food-canned-food", neutral) - 2,
        )

    def test_cow_is_near_mandatory_on_food_scaling_line(self):
        player = Player(
            shop=Shop(["cow", "deer"], turn=7),
            team=Team(["rabbit", "ant", "fish", "turtle"]),
            turn=7,
        )
        plan = LinePlan(
            line_name="rabbit / cat food",
            confidence=0.8,
            committed_slots=(0, 1, 2),
            flexible_slots=(3,),
            sell_slot=3,
            required_hits=("pet-cow",),
        )
        cow = self.teacher._score_buy_pet(
            player,
            AutoregressiveAction(ActionType.BUY_PET, 0, 4),
            plan,
            None,
        )
        deer = self.teacher._score_buy_pet(
            player,
            AutoregressiveAction(ActionType.BUY_PET, 1, 4),
            plan,
            None,
        )
        self.assertGreater(cow, deer + 10)

    def test_full_board_cow_plan_prefers_designated_sell_slot(self):
        player = Player(
            shop=Shop(["cow"], turn=7),
            team=Team(["rabbit", "cat", "fish", "turtle", "ant"]),
            turn=7,
        )
        plan = LinePlan(
            line_name="rabbit / cat food",
            confidence=0.8,
            committed_slots=(0, 1, 2, 3),
            flexible_slots=(4,),
            sell_slot=4,
            required_hits=("pet-cow",),
        )
        intended = self.teacher._score_sell(player, 4, plan)
        committed = self.teacher._score_sell(player, 1, plan)
        self.assertGreater(intended, committed + 15)

    def test_cow_consumes_free_milk_before_cycle_sell(self):
        player = Player(
            shop=Shop(["milk"], turn=7), team=Team(["cow", "ant"]), turn=7
        )
        player.shop[0].cost = 0
        plan = LinePlan(line_name="generic stat-tech", sell_slot=0)
        before_milk = self.teacher._score_sell(player, 0, plan)
        player.shop.slots = []
        player.shop._nslots = 0
        after_milk = self.teacher._score_sell(player, 0, plan)
        self.assertGreater(after_milk, before_milk + 20)

    def test_elephant_hurt_engine_scores_ordered_composition(self):
        ordered = Team(["elephant", "blowfish", "wolverine", "deer"])
        broken = Team(["blowfish", "elephant", "wolverine", "deer"])
        ordered_score = self.teacher._composition_value(
            ordered, "elephant hurt engine", ()
        )
        broken_score = self.teacher._composition_value(
            broken, "elephant hurt engine", ()
        )
        self.assertGreater(ordered_score, broken_score + 15)

    def test_mammoth_mushroom_tiger_is_premium_and_skunk_favorable(self):
        team = Team(["mammoth", "tiger"])
        team[0].pet.status = "status-extra-life"
        plain = self.teacher._composition_value(
            team, "tiger-mammoth-mushroom", ()
        )
        skunk = self.teacher._composition_value(
            team,
            "tiger-mammoth-mushroom",
            ("skunk-accelerates-mammoth",),
        )
        reversed_team = Team(["tiger", "mammoth"])
        reversed_team[1].pet.status = "status-extra-life"
        reversed_score = self.teacher._composition_value(
            reversed_team, "tiger-mammoth-mushroom", ()
        )
        self.assertGreater(plain, reversed_score + 20)
        self.assertGreater(skunk, plain)

    def test_partial_mammoth_or_tiger_can_establish_persistent_line(self):
        for pet_name in ("mammoth", "tiger"):
            with self.subTest(pet=pet_name):
                core = SuperAutoPetsEnv(manual_battles=True)
                core.agent = Player(
                    shop=Shop([]), team=Team([pet_name]), turn=9
                )
                plan = self.teacher._update_plan(
                    core, 0, core.agent, None
                )
                self.assertEqual(
                    plan.line_name, "tiger-mammoth-mushroom"
                )
                self.assertIn(
                    "food-mushroom", plan.required_hits
                )

    def test_mammoth_tiger_is_reduced_by_rhino_and_elephant_blowfish(self):
        team = Team(["mammoth", "tiger"])
        team[0].pet.status = "status-extra-life"
        plain = self.teacher._composition_value(
            team, "tiger-mammoth-mushroom", ()
        )
        countered = self.teacher._composition_value(
            team,
            "tiger-mammoth-mushroom",
            (
                "rhino-vs-summons",
                "elephant-blowfish-vs-fragile-engine",
            ),
        )
        self.assertLess(countered, plain - 15)

    def test_deer_turn_seven_default_and_rhino_penalty(self):
        player = Player(
            shop=Shop(["deer"], turn=7), team=Team(["deer", "ant"]), turn=7
        )
        favorable = LinePlan(
            line_name="generic stat-tech", matchup_counters=()
        )
        countered = LinePlan(
            line_name="generic stat-tech",
            matchup_counters=("rhino-vs-summons",),
        )
        default_value = self.teacher._item_value(player, "pet-deer", favorable)
        rhino_value = self.teacher._item_value(player, "pet-deer", countered)
        self.assertGreater(default_value, rhino_value + 15)

    def test_deer_prefers_front_or_mid_front(self):
        front = Team(["deer", "ant", "fish"])
        back = Team(["ant", "fish", "deer"])
        self.assertGreater(
            self.teacher._composition_value(front, "generic stat-tech", ()),
            self.teacher._composition_value(back, "generic stat-tech", ()),
        )

    def test_each_requested_opponent_counter_is_detected(self):
        player = Player(shop=Shop([]), team=Team(["deer", "tiger"]), turn=9)
        player.team[0].pet._attack = 30
        player.team[0].pet._health = 30
        cases = {
            "rhino-vs-summons": Team(["rhino"]),
            "skunk-vs-tall-stats": Team(["skunk"]),
            "elephant-blowfish-vs-fragile-engine": Team(
                ["elephant", "blowfish"]
            ),
            "crocodile-vs-backline-tiger": Team(["crocodile"]),
            "leopard-vs-weak-melon": Team(["leopard"]),
            "scorpion-vs-tall-front": Team(["scorpion"]),
        }
        for expected, opponent in cases.items():
            with self.subTest(counter=expected):
                counters = self.teacher._matchup_counters(
                    player, opponent, "generic stat-tech"
                )
                self.assertIn(expected, counters)

    def test_turn_two_six_lives_biases_roll_toward_free_loss_setup(self):
        player = Player(shop=Shop(["ant"]), team=Team(["fish"]), turn=2, lives=6)
        plan = LinePlan(line_name="sell-pet / triple economy")
        at_six = self.teacher._score_roll(player, 0, plan)
        player.lives = 5
        at_five = self.teacher._score_roll(player, 0, plan)
        self.assertGreater(at_six, at_five)

    def test_turn_three_four_scaling_apple_targets(self):
        plan = LinePlan(line_name="generic stat-tech")

        tier_three = Player(shop=Shop([]), team=Team(["giraffe"]), turn=3)
        self.assertEqual(
            self.teacher._scaling_apple_bonus(
                tier_three, tier_three.team[0].pet, plan
            ),
            14.0,
        )

        pair = Player(
            shop=Shop([]), team=Team(["peacock", "peacock"]), turn=4
        )
        self.assertEqual(
            self.teacher._scaling_apple_bonus(pair, pair.team[0].pet, plan),
            12.0,
        )

        big_plan = LinePlan(line_name="big tier-1")
        big = Player(shop=Shop([]), team=Team(["ant"]), turn=3)
        self.assertEqual(
            self.teacher._scaling_apple_bonus(big, big.team[0].pet, big_plan),
            10.0,
        )

    def test_scaling_apple_rule_excludes_horse_cricket_and_dog(self):
        horse_plan = LinePlan(line_name="horse-cricket summon tempo")
        horse = Player(shop=Shop([]), team=Team(["giraffe"]), turn=3)
        self.assertEqual(
            self.teacher._scaling_apple_bonus(
                horse, horse.team[0].pet, horse_plan
            ),
            0.0,
        )
        dog = Player(shop=Shop([]), team=Team(["dog", "giraffe"]), turn=4)
        generic = LinePlan(line_name="generic stat-tech")
        self.assertEqual(
            self.teacher._scaling_apple_bonus(dog, dog.team[1].pet, generic),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
