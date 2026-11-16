import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sapai_gym1.replay import (
    ReplayCode,
    ReplayParser,
    ReplayReconstructor,
    semantic_state_to_observation,
    collate_replay_sequences,
)
from sapai_gym1.replay.client import SapLibraryClient


CALCULATOR_ROOT = Path(__file__).resolve().parents[2] / "SAP-Calculator"


def nested(value):
    return json.dumps(value, separators=(",", ":"))


class TestReplayIngestion(unittest.TestCase):
    def test_replay_code_parser(self):
        self.assertEqual(ReplayCode.parse("abc").participation_id, "abc")
        self.assertEqual(ReplayCode.parse('{"Pid":"abc","T":4}').turn, 4)

    def test_player_search_builds_bounded_cached_query(self):
        with tempfile.TemporaryDirectory() as directory:
            client = SapLibraryClient(directory)
            with mock.patch.object(
                client, "_get_json", return_value={"results": []}
            ) as get_json:
                client.search_player("player", page=3, page_size=500)
        get_json.assert_called_once_with(
            "/search",
            {
                "playerId": "player",
                "matchType": "private",
                "version": "current",
                "page": 3,
                "pageSize": 100,
                "sort": "created_at",
                "order": "desc",
            },
        )

    def test_validated_end_state_and_quarantine(self):
        board_id = "board"
        pet_id = {"BoId": board_id, "Uni": 1}
        unknown_food_id = {"BoId": board_id, "Uni": 99}
        build = {
            "Bor": {
                "Tur": 1,
                "GoSp": 10,
                "Sta": 6,
                "LiMa": 6,
                "Hash": 123,
                "Mins": {
                    "Items": [
                        {
                            "Enu": 3,
                            "Id": pet_id,
                            "Lvl": 1,
                            "At": {"Perm": 3},
                            "Hp": {"Perm": 2},
                        },
                        None,
                        None,
                        None,
                        None,
                    ]
                },
                "MiSh": [],
                "SpSh": [],
            }
        }
        actions = [
            {
                "Type": 6,
                "Turn": 1,
                "Request": nested({"MinionId": pet_id}),
                "Response": nested({}),
            },
            {
                "Type": 8,
                "Turn": 1,
                "Request": nested({"SpellId": unknown_food_id, "Aim": pet_id}),
                "Response": nested({}),
            },
            {
                "Type": 11,
                "Turn": 1,
                "Request": nested({}),
                "Response": nested({"Data": {"Hash": 123}}),
            },
            {
                "Type": 0,
                "Turn": 1,
                "Build": nested(build),
                "Battle": nested({"Outcome": 1}),
                "Request": nested({}),
                "Response": nested({}),
            },
        ]
        bundle = {
            "replayId": "fixture",
            "metadata": {
                "id": "fixture",
                "pack": "Turtle",
                "opponent_pack": "Turtle",
                "player_name": "must-not-survive",
            },
            "perspectives": {
                "perspectives": [
                    {
                        "role": "player",
                        "participationId": "pid",
                        "raw": {"Actions": actions, "Outcome": 1},
                    }
                ]
            },
            "turns": {"turns": [{"turn": 1}]},
        }
        report = ReplayReconstructor(ReplayParser(CALCULATOR_ROOT)).reconstruct(bundle)
        self.assertEqual(len(report.examples), 1)
        self.assertEqual(report.examples[0].state["boardHash"], 123)
        self.assertEqual(report.examples[0].state["team"][0]["sapaiId"], "pet-beaver")
        self.assertEqual(len(report.quarantine), 1)
        self.assertEqual(report.quarantine[0].reason, "unresolved_object_identity")
        self.assertNotIn("player_name", report.metadata)

        with tempfile.TemporaryDirectory() as directory:
            report.write(directory)
            self.assertTrue(Path(directory, "fixture.report.json").exists())
            self.assertTrue(Path(directory, "fixture.examples.jsonl").exists())

    def test_replay_absolute_experience_matches_live_entity_feature(self):
        state = {
            "turn": 5,
            "gold": 0,
            "team": [
                {
                    "sapaiId": "pet-fish",
                    "attack": 4,
                    "health": 5,
                    "level": 2,
                    "experience": 2,
                }
            ],
            "shop": [],
        }
        observation = semantic_state_to_observation(state)
        self.assertAlmostEqual(observation["numeric"][1, 2], 2 / 3)
        self.assertAlmostEqual(observation["numeric"][1, 3], 2 / 5)

    def test_overlapping_pet_food_enums_resolve_by_action_context(self):
        parser = ReplayParser(CALCULATOR_ROOT)
        board_id = "overlap"
        source_id = {"BoId": board_id, "Uni": 1}
        target_id = {"BoId": board_id, "Uni": 2}
        perspective = {
            "raw": {
                "Actions": [
                    {
                        "Type": 8,
                        "Turn": 1,
                        "Request": nested(
                            {"SpellId": source_id, "Aim": target_id}
                        ),
                        "Response": nested({}),
                        "Build": nested(
                            {
                                "Bor": {
                                    "SpSh": [{"Id": source_id, "Enu": 3}],
                                    "Mins": {
                                        "Items": [
                                            {"Id": target_id, "Enu": 3}
                                        ]
                                    },
                                }
                            }
                        ),
                    }
                ]
            }
        }
        actions = parser.parse_perspective(perspective)
        catalog = parser.build_object_catalog(actions)
        identity = ReplayReconstructor(parser)._resolve_action_identity(
            actions[0], catalog
        )
        self.assertTrue(identity["resolved"])
        self.assertEqual(identity["objects"]["source"]["name"], "Baguette")
        self.assertEqual(identity["objects"]["target"]["name"], "Beaver")

    def test_untargeted_random_food_does_not_require_an_aim_identity(self):
        parser = ReplayParser(CALCULATOR_ROOT)
        food_id = {"BoId": "random-food", "Uni": 1}
        perspective = {
            "raw": {
                "Actions": [
                    {
                        "Type": 8,
                        "Turn": 5,
                        "Request": nested({"SpellId": food_id}),
                        "Response": nested(
                            {
                                "Event": {
                                    "Event": {
                                        "Spell": {"Id": food_id, "Enu": 63}
                                    }
                                }
                            }
                        ),
                    }
                ]
            }
        }
        actions = parser.parse_perspective(perspective)
        catalog = parser.build_object_catalog(actions)
        identity = ReplayReconstructor(parser)._resolve_action_identity(
            actions[0], catalog
        )
        self.assertTrue(identity["resolved"])
        self.assertEqual(identity["objects"]["source"]["name"], "Pizza")
        self.assertNotIn("target", identity["objects"])

    def test_combine_propagates_known_target_identity_to_anonymous_source(self):
        parser = ReplayParser(CALCULATOR_ROOT)
        source_id = {"BoId": "combine", "Uni": 1}
        target_id = {"BoId": "combine", "Uni": 2}
        perspective = {
            "raw": {
                "Actions": [
                    {
                        "Type": 7,
                        "Turn": 3,
                        "Request": nested(
                            {
                                "SourceMinionId": source_id,
                                "TargetMinionId": target_id,
                            }
                        ),
                        "Response": nested({}),
                        "Build": nested(
                            {
                                "Bor": {
                                    "Mins": {
                                        "Items": [
                                            {"Id": target_id, "Enu": 70}
                                        ]
                                    }
                                }
                            }
                        ),
                    }
                ]
            }
        }
        actions = parser.parse_perspective(perspective)
        catalog = parser.build_object_catalog(actions)
        identity = ReplayReconstructor(parser)._resolve_action_identity(
            actions[0], catalog
        )
        self.assertTrue(identity["resolved"])
        self.assertEqual(identity["objects"]["source"]["name"], "Skunk")
        self.assertEqual(identity["objects"]["target"]["name"], "Skunk")

    def test_recurrent_sequence_collation_preserves_lengths(self):
        observation = semantic_state_to_observation(
            {"turn": 1, "gold": 10, "team": [], "shop": []}
        )
        item = {
            "observation": observation,
            "action": np.array([0, 10, 5], dtype=np.int64),
            "battle_outcome": 2,
            "return": np.float32(0),
            "weight": np.float32(0.25),
            "initial_memory": np.asarray([1.0, 2.0], dtype=np.float32),
            "replay_id": "a",
        }
        batch = collate_replay_sequences([[item, item], [item]])
        self.assertEqual(tuple(batch["actions"].shape), (2, 2, 3))
        self.assertEqual(batch["sequence_lengths"].tolist(), [2, 1])
        self.assertEqual(
            batch["valid_mask"].tolist(), [[True, True], [True, False]]
        )
        self.assertEqual(
            batch["example_weights"].tolist(),
            [[0.25, 0.25], [0.25, 0.0]],
        )
        np.testing.assert_array_equal(
            batch["initial_memories"][0], item["initial_memory"]
        )


if __name__ == "__main__":
    unittest.main()
