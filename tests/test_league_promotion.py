import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.promote_league_checkpoint import validate_reports


class LeaguePromotionGateTest(unittest.TestCase):
    def _report(self, score=0.67):
        return {
            "checkpoint": "/workspace/runs/test/update-001000.pt",
            "checkpointUpdate": 1000,
            "checkpointTransitions": 1234,
            "checkpointSimulatorRevision": "revision",
            "runtimeSimulatorRevision": "revision",
            "opponentCheckpointSimulatorRevision": "revision",
            "pairs": 200,
            "pairedMeanScore": score,
            "pairedScore95CI": [0.58, 0.76],
        }

    def _args(self, paths):
        return SimpleNamespace(
            reports=paths,
            minimum_score=0.65,
            minimum_pairs=100,
            minimum_ci_low=0.50,
            allow_opponent_training_revision_mismatch=False,
        )

    def test_requires_every_panel_report_to_clear_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, score in enumerate((0.67, 0.66)):
                path = Path(directory) / f"report-{index}.json"
                path.write_text(json.dumps(self._report(score)))
                paths.append(path)
            reports, checkpoint, revision = validate_reports(
                self._args(paths)
            )
            self.assertEqual(len(reports), 2)
            self.assertEqual(
                checkpoint, "/workspace/runs/test/update-001000.pt"
            )
            self.assertEqual(revision, "revision")

            paths[1].write_text(json.dumps(self._report(0.64)))
            with self.assertRaisesRegex(SystemExit, "score 0.640"):
                validate_reports(self._args(paths))

    def test_old_opponent_revision_requires_explicit_runtime_safe_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = self._report()
            report["opponentCheckpointSimulatorRevision"] = "older-revision"
            path.write_text(json.dumps(report))
            args = self._args([path])
            with self.assertRaisesRegex(
                SystemExit, "opponent training revision mismatch"
            ):
                validate_reports(args)
            args.allow_opponent_training_revision_mismatch = True
            reports, _, revision = validate_reports(args)
            self.assertEqual(len(reports), 1)
            self.assertEqual(revision, "revision")

    def test_candidate_runtime_revision_mismatch_is_never_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = self._report()
            report["runtimeSimulatorRevision"] = "wrong-runtime"
            path.write_text(json.dumps(report))
            args = self._args([path])
            args.allow_opponent_training_revision_mismatch = True
            with self.assertRaisesRegex(
                SystemExit, "candidate/runtime simulator mismatch"
            ):
                validate_reports(args)


if __name__ == "__main__":
    unittest.main()
