import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestProgressionSummary(unittest.TestCase):
    def test_writes_sorted_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for candidate, opponent, score in (
                (3000, 2000, 0.55),
                (2000, 1000, 0.52),
            ):
                report = {
                    "checkpointUpdate": candidate,
                    "opponentCheckpointUpdate": opponent,
                    "games": 200,
                    "outcomes": {"win": 110, "draw": 0, "loss": 90},
                    "pairedMeanScore": score,
                    "pairedScore95CI": [score - 0.05, score + 0.05],
                    "seed": candidate,
                    "runtimeSimulatorRevision": "test-v16",
                }
                (root / f"paired-{candidate:06d}-vs-{opponent:06d}.json").write_text(
                    json.dumps(report)
                )
            behavior = {
                "checkpointUpdate": 2000,
                "opponentCheckpoint": "/runs/example/update-001000.pt",
                "games": 100,
                "matchWinRate": 0.56,
                "meanFinalTeamStats": 90.0,
                "meanSellsPerMatch": 4.5,
                "meanPermanentlyStattedPets": 3.0,
                "meanTierOneTwoPetsAfterTurnEight": 2.0,
                "fractionEndTurnsWithFullBoard": 0.8,
                "meanUnspentGold": 0.0,
                "discoverGroupsSeen": 300,
                "discoverPurchases": 150,
                "discoverFreezeActions": 20,
            }
            (root / "behavior-v2-002000-vs-001000.json").write_text(
                json.dumps(behavior)
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_v16_progression.py"
            subprocess.run(
                [sys.executable, str(script), str(root)],
                check=True,
                capture_output=True,
                text=True,
            )
            trend = json.loads((root / "trend.json").read_text())
            self.assertEqual(
                [row["candidateUpdate"] for row in trend["comparisons"]],
                [2000, 3000],
            )
            self.assertEqual(trend["behavior"][0]["candidateUpdate"], 2000)
            markdown = (root / "TREND.md").read_text()
            self.assertIn("| 2000 | 1000 | fixed |", markdown)
            self.assertIn("52.0%", markdown)
            self.assertIn("56.0%", markdown)


if __name__ == "__main__":
    unittest.main()
