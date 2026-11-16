from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch

from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy, ParallelSelfPlayEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_paired_policy as evaluation


def test_natural_death_takes_precedence_over_a_simultaneous_turn_limit():
    core = SimpleNamespace(agent=SimpleNamespace(lives=0, wins=2),
                           opponent=SimpleNamespace(lives=4, wins=6),
                           turn=25, MAX_TURN=25, MAX_REORDERS_PER_TURN=2)
    record = evaluation.evaluation_game_record(core, 1, 123, 1, 'win')
    assert record['terminationReason'] == 'lives'
    assert record['candidateSide'] == 1 and record['livesBySeat'] == [0, 4]
    core.agent.lives = 3
    assert evaluation.evaluation_game_record(core, 1, 123, 1, 'win')['terminationReason'] == 'turn_limit'


def test_termination_logging_does_not_change_paired_evaluation(monkeypatch):
    def short_environment():
        env = ParallelSelfPlayEnv()
        env.core.MAX_TURN = 2
        return env

    monkeypatch.setattr(evaluation, 'ParallelSelfPlayEnv', short_environment)
    encoder = ParallelSelfPlayEnv().encoder
    torch.manual_seed(3121)
    policy = EntityTransformerPolicy(EntityTransformerConfig(
        num_items=encoder.num_items, num_statuses=encoder.num_statuses,
        d_model=32, nhead=4, num_layers=1, dim_feedforward=64, memory_dim=24,
    )).eval()
    args = (policy, policy, np.array([81, 82]), torch.device('cpu'))
    plain = evaluation.run_batched_checkpoint_evaluation(*args, batch_size=4, progress_every=0)
    records = []
    logged = evaluation.run_batched_checkpoint_evaluation(*args, batch_size=4, progress_every=0, game_records=records)
    assert plain == logged
    assert len(records) == 4
    assert all(record['terminationReason'] == 'turn_limit' for record in records)
    assert {(record['seed'], record['candidateSide']) for record in records} == {(81,0),(81,1),(82,0),(82,1)}


def test_interrupted_report_publication_preserves_the_previous_complete_file(tmp_path, monkeypatch):
    path = tmp_path / 'evaluation.json'
    evaluation.write_evaluation_report(path, {'completed': 1})
    def interrupted(source, target):
        raise OSError('simulated publication interruption')
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', interrupted)
        with pytest.raises(OSError, match='interruption'):
            evaluation.write_evaluation_report(path, {'completed': 2})
    assert json.loads(path.read_text()) == {'completed': 1}
    evaluation.write_evaluation_report(path, {'completed': 2})
    assert json.loads(path.read_text()) == {'completed': 2}
    assert not path.with_suffix('.json.tmp').exists()
