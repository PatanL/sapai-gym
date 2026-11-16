import json
from pathlib import Path
import tempfile
import unittest
import sys

from sapai import Team
from sapai_gym1.rl.opponent_reservoir import (
    OpponentTrajectoryReservoir,
    RESERVOIR_SCHEMA_VERSION,
    team_distance,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_opponent_trajectory_reservoirs import merge_reservoirs  # noqa: E402


class OpponentTrajectoryReservoirTest(unittest.TestCase):
    @staticmethod
    def board(turn, names):
        return {"turn": turn, "team": Team(names).state}

    def test_matches_public_anchor_and_returns_coherent_future(self):
        payload = {
            "schemaVersion": RESERVOIR_SCHEMA_VERSION,
            "metadata": {"simulatorRevision": "test"},
            "trajectories": [
                {
                    "id": "close",
                    "boards": [
                        self.board(1, ["ant"]),
                        self.board(2, ["fish"]),
                        self.board(3, ["giraffe", "fish"]),
                        self.board(4, ["giraffe", "peacock"]),
                    ],
                },
                {
                    "id": "far",
                    "boards": [
                        self.board(1, ["dragon"]),
                        self.board(2, ["leopard"]),
                        self.board(3, ["tiger"]),
                        self.board(4, ["snake"]),
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reservoir.json"
            path.write_text(json.dumps(payload))
            reservoir = OpponentTrajectoryReservoir.load(path)
        matches = reservoir.match(
            Team(["ant"]), current_turn=2, horizon=3, limit=2
        )
        self.assertEqual([match.trajectory_id for match in matches], ["close", "far"])
        self.assertEqual(
            [
                next(slot.pet.name for slot in team if not slot.empty)
                for team in matches[0].teams
            ],
            ["pet-fish", "pet-giraffe", "pet-giraffe"],
        )

    def test_distance_penalizes_identity_and_position(self):
        observed = Team(["ant", "fish"])
        self.assertEqual(team_distance(observed, Team(["ant", "fish"])), 0.0)
        self.assertGreater(
            team_distance(observed, Team(["fish", "ant"])), 0.0
        )

    def test_matching_conditions_on_revealed_enemy_board(self):
        trajectories = [
            {
                "id": "saw-fish",
                "boards": [
                    {
                        **self.board(3, ["ant"]),
                        "opponentTeam": Team(["fish"]).state,
                    },
                    {
                        **self.board(4, ["rhino"]),
                        "opponentTeam": Team(["deer"]).state,
                    },
                ],
            },
            {
                "id": "saw-dragon",
                "boards": [
                    {
                        **self.board(3, ["ant"]),
                        "opponentTeam": Team(["dragon"]).state,
                    },
                    {
                        **self.board(4, ["skunk"]),
                        "opponentTeam": Team(["hippo"]).state,
                    },
                ],
            },
        ]
        reservoir = OpponentTrajectoryReservoir(trajectories)
        fish_match = reservoir.match(
            Team(["ant"]),
            observed_enemy=Team(["fish"]),
            current_turn=4,
            horizon=1,
            limit=1,
        )
        dragon_match = reservoir.match(
            Team(["ant"]),
            observed_enemy=Team(["dragon"]),
            current_turn=4,
            horizon=1,
            limit=1,
        )
        self.assertEqual(fish_match[0].trajectory_id, "saw-fish")
        self.assertEqual(dragon_match[0].trajectory_id, "saw-dragon")

    def test_matching_uses_two_board_public_history(self):
        trajectories = [
            {
                "id": "fish-history",
                "boards": [
                    self.board(2, ["fish"]),
                    self.board(3, ["ant"]),
                    self.board(4, ["rhino"]),
                ],
            },
            {
                "id": "dragon-history",
                "boards": [
                    self.board(2, ["dragon"]),
                    self.board(3, ["ant"]),
                    self.board(4, ["rhino"]),
                ],
            },
        ]
        reservoir = OpponentTrajectoryReservoir(trajectories)
        matches = reservoir.match(
            Team(["ant"]),
            older_observed=Team(["dragon"]),
            current_turn=4,
            horizon=1,
            limit=2,
        )
        self.assertEqual(matches[0].trajectory_id, "dragon-history")

    def test_merges_diverse_sources_only_under_one_runtime_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, training_revision in enumerate(("new", "old")):
                path = Path(directory) / f"source-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schemaVersion": RESERVOIR_SCHEMA_VERSION,
                            "metadata": {
                                "simulatorRevision": "runtime",
                                "checkpoint": f"checkpoint-{index}",
                                "checkpointTrainingRevision": training_revision,
                                "games": 1,
                                "seed": index,
                            },
                            "trajectories": [
                                {
                                    "id": "game-0-p0",
                                    "boards": [self.board(1, ["ant"])],
                                }
                            ],
                        }
                    )
                )
                paths.append(path)
            merged = merge_reservoirs(paths)
            self.assertEqual(merged["metadata"]["games"], 2)
            self.assertEqual(merged["metadata"]["trajectories"], 2)
            self.assertEqual(
                [row["checkpointTrainingRevision"] for row in merged["metadata"]["components"]],
                ["new", "old"],
            )
            self.assertEqual(
                {row["id"] for row in merged["trajectories"]},
                {"source-00-game-0-p0", "source-01-game-0-p0"},
            )

            for trajectory in merged["trajectories"]:
                trajectory["boards"].append(self.board(2, ["fish"]))
            reservoir = OpponentTrajectoryReservoir(merged["trajectories"])
            diverse = reservoir.match(
                Team(["ant"]), current_turn=2, horizon=1, limit=2
            )
            self.assertEqual(
                {match.source_index for match in diverse}, {0, 1}
            )

            weighted = OpponentTrajectoryReservoir(
                [
                    {
                        **trajectory,
                        "id": f"{trajectory['id']}-{copy}",
                    }
                    for trajectory in merged["trajectories"]
                    for copy in range(20)
                ],
                metadata={
                    "components": [
                        {"sourceIndex": 0, "games": 3},
                        {"sourceIndex": 1, "games": 1},
                    ]
                },
            )
            weighted_matches = weighted.match(
                Team(["ant"]), current_turn=2, horizon=1, limit=8
            )
            source_counts = {
                source: sum(
                    match.source_index == source
                    for match in weighted_matches
                )
                for source in (0, 1)
            }
            self.assertEqual(source_counts, {0: 6, 1: 2})

            bad = json.loads(paths[1].read_text())
            bad["metadata"]["simulatorRevision"] = "different-runtime"
            paths[1].write_text(json.dumps(bad))
            with self.assertRaisesRegex(SystemExit, "incompatible simulator"):
                merge_reservoirs(paths)


if __name__ == "__main__":
    unittest.main()
