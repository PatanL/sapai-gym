import json
from pathlib import Path
import tempfile
import unittest

from sapai import Food, Pet, Player, Shop, Team

from sapai_gym1.rl.game_trace import (
    action_snapshot,
    audit_state,
    pet_snapshot,
    player_snapshot,
    render_game_html,
    state_changes,
)
from sapai_gym1.rl import ActionType, AutoregressiveAction


class TestGameVisualizer(unittest.TestCase):
    def test_move_caption_describes_insertion_not_a_swap(self):
        player = Player(team=Team(['ant', 'beaver', 'cricket', 'duck', 'fish']))
        action = AutoregressiveAction(ActionType.MOVE, 0, 4)
        description = action_snapshot(player, action)['description']
        self.assertIn('Move Ant from team 0 to team 4', description)
        self.assertIn('shifting intervening pets', description)
        self.assertNotIn('Swap', description)
        player.swap(0, 4)
        self.assertEqual([slot.pet.name for slot in player.team],
                         ['pet-beaver', 'pet-cricket', 'pet-duck', 'pet-fish', 'pet-ant'])

    def test_generated_pet_snapshot_has_gameplay_tier_one(self):
        bus = Pet("bus")
        bus._attack = 5
        bus._health = 3
        self.assertEqual(bus.tier, 1)
        self.assertEqual(pet_snapshot(bus)["tier"], 1)

    def make_player(self):
        shop = Shop(
            slots=[
                Pet("ant"),
                Pet("ox"),
                Food("apple"),
                Pet("deer"),
            ],
            turn=7,
        )
        shop.slots[0].frozen = True
        shop.slots[1].slot_type = "levelup"
        shop.slots[3].slot_type = "levelup"
        shop.slots[1].reward_group = 4
        shop.slots[3].reward_group = 4
        team = Team([Pet("fish"), Pet("camel")])
        return Player(shop=shop, team=team, turn=7, gold=4)

    def test_snapshot_preserves_debug_fields(self):
        player = self.make_player()
        player.team[1].pet.ability_counter = 2
        snapshot = player_snapshot(player)
        self.assertEqual(snapshot["gold"], 4)
        self.assertTrue(snapshot["shop"][0]["frozen"])
        self.assertEqual(snapshot["shop"][1]["rewardGroup"], 4)
        self.assertEqual(snapshot["shop"][3]["rewardGroup"], 4)
        self.assertEqual(snapshot["team"][1]["abilityCounter"], 2)
        self.assertEqual(snapshot["team"][0]["trigger"], "LevelUp")

    def test_changes_show_exact_stat_and_shop_mutations(self):
        player = self.make_player()
        before_player = player_snapshot(player)
        player.gold = 1
        player.team[0].pet._health += 2
        player.shop.slots[0].frozen = False
        after_player = player_snapshot(player)
        before = {"player": before_player}
        after = {"player": after_player}
        changes = state_changes(before, after)
        self.assertIn("Gold: 4 → 1", changes)
        self.assertTrue(any("Fish 2/3" in row and "Fish 2/5" in row for row in changes))
        self.assertTrue(any("Shop 0:" in row and "frozen" in row for row in changes))

    def test_audit_flags_malformed_discover_and_unspent_end(self):
        player = self.make_player()
        player.shop.slots[0].reward_group = 4
        before = {
            "player": player_snapshot(player),
            "actionCount": 7,
        }
        after = json.loads(json.dumps(before))
        warnings = audit_state(
            {"type": "END_TURN", "cost": 0}, before, after
        )
        texts = [warning["text"] for warning in warnings]
        self.assertTrue(any("Discover #4 has 3 choices" in text for text in texts))
        self.assertTrue(any("unspent gold" in text for text in texts))

    def test_single_frozen_discover_survives_roll_without_warning(self):
        player = self.make_player()
        before = {
            "player": player_snapshot(player),
            "actionCount": 3,
        }
        player.shop.slots[1].frozen = True
        player.shop.slots.pop(3)
        after = {
            "player": player_snapshot(player),
            "actionCount": 4,
        }
        warnings = audit_state(
            {"type": "ROLL", "cost": 0}, before, after
        )
        self.assertFalse(
            any("Discover #4" in warning["text"] for warning in warnings)
        )

    def test_renderer_embeds_trace_and_controls(self):
        player = self.make_player()
        state = {
            "player": player_snapshot(player),
            "opponentScore": {"lives": 6, "wins": 0},
            "lastOpponentTeam": None,
            "actionCount": 0,
            "reorderCount": 0,
        }
        trace = {
            "metadata": {
                "title": "Inspector test",
                "seed": 1,
                "simulatorRevision": "test",
            },
            "frames": [
                {
                    "turn": 1,
                    "phase": "shop",
                    "label": "Shop opens",
                    "before": state,
                    "after": state,
                    "action": None,
                    "changes": [],
                    "warnings": [],
                    "policy": None,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "game.html"
            render_game_html(trace, output)
            document = output.read_text()
        self.assertIn("Inspector test", document)
        self.assertIn('id="prev"', document)
        self.assertIn('id="trace"', document)
        self.assertIn("Physical slots 0", document.replace("physical", "Physical"))

    def test_sell_gold_audit_includes_pig_and_cake_value(self):
        pig = Pet("pig")
        pig.level = 2
        pig.sell_value = 4
        player = Player(shop=Shop([]), team=Team([pig]), gold=1)
        action = action_snapshot(
            player,
            AutoregressiveAction(ActionType.SELL, 0),
        )
        before = {"player": player_snapshot(player), "actionCount": 1}
        player.sell(0)
        after = {"player": player_snapshot(player), "actionCount": 2}
        self.assertEqual(audit_state(action, before, after), [])

    def test_freeze_audit_detects_missing_state_change(self):
        player = self.make_player()
        player.shop.slots[0].frozen = False
        action = action_snapshot(
            player,
            AutoregressiveAction(ActionType.FREEZE, 0),
        )
        before = {"player": player_snapshot(player), "actionCount": 1}
        after = json.loads(json.dumps(before))
        warnings = audit_state(action, before, after)
        self.assertTrue(any("frozen=True" in row["text"] for row in warnings))


if __name__ == "__main__":
    unittest.main()
