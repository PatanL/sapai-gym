from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from sapai_gym1.rl.historical_evaluation import (
    HistoricalOpponentEvaluator, POLICY_OWNERSHIP_VERSION, restore_historical_decision,
)
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy, FlatActionCodec, ParallelSelfPlayEnv
from sapai_gym1.rl.decision_manifest import decision_record, restore_decision

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from search_state_mining import mine_match_opportunity


class TaggedPolicy:
    def __init__(self, tag, dimension):
        self.tag = tag
        self.config = SimpleNamespace(memory_dim=dimension)
        self.calls = []

    def __call__(self, observations, memory, deterministic):
        assert deterministic
        assert memory.shape[1] == self.config.memory_dim
        self.calls.append(memory.clone())
        return SimpleNamespace(
            actions=torch.tensor([[self.tag, 10, 5]]).repeat(len(memory), 1),
            memory=memory + self.tag,
        )


@pytest.mark.parametrize('acting_side', [0, 1])
def test_interleaved_requests_keep_network_and_memory_assignment(acting_side):
    main, opponent = TaggedPolicy(1, 3), TaggedPolicy(2, 7)
    evaluator = HistoricalOpponentEvaluator(main, opponent, acting_side=acting_side, batch_size=2)
    requests = []
    for index, side in enumerate([1, 0, 0, 1, 1]):
        dimension = 3 if side == acting_side else 7
        requests.append((index, side, {'x': np.array([index])}, torch.full((dimension,), float(index))))
    choices = evaluator._choose_many(requests)
    for request, (action, memory) in zip(requests, choices):
        tag = 1 if request[1] == acting_side else 2
        assert action.action_type == tag
        torch.testing.assert_close(memory, request[3] + tag)
    assert main.calls and opponent.calls


def small_policy(seed, memory_dim):
    torch.manual_seed(seed)
    encoder = ParallelSelfPlayEnv().encoder
    return EntityTransformerPolicy(EntityTransformerConfig(
        num_items=encoder.num_items, num_statuses=encoder.num_statuses,
        d_model=32, nhead=4, num_layers=1, dim_feedforward=64, memory_dim=memory_dim,
    )).eval()


@pytest.mark.parametrize('acting_side', [0, 1])
def test_historical_mining_replay_and_real_continuation(acting_side):
    main, opponent = small_policy(661, 24), small_policy(663, 32)
    selected = mine_match_opportunity(
        main, np.random.default_rng(917), torch.device('cpu'), 2, 2, 'funded',
        opponent_policy=opponent, acting_side=acting_side, with_replay=True,
    )
    assert selected is not None
    env, memories, side, reasons, replay = selected
    assert side == acting_side
    assert memories[side].shape == (24,)
    assert memories[1 - side].shape == (32,)
    record = decision_record(env, memories, side, reasons, replay,
                             search_seed=1, validation_seed=2, truth_seeds=[3, 4])
    owners = ['other-network', 'other-network']
    owners[side] = 'main-network'
    record.update(policyOwnershipVersion=POLICY_OWNERSHIP_VERSION,
                  policySha256BySeat=owners, checkpointSha256='main-network',
                  opponentCheckpointSha256='other-network')
    restore_historical_decision(record, 'main-network', 'other-network')
    with pytest.raises(ValueError, match='ownership'):
        restore_historical_decision(record, 'main-network', 'wrong-network')
    restored, replay_memories, _, _ = restore_decision(record)
    evaluator = HistoricalOpponentEvaluator(main, opponent, acting_side=side, horizon_battles=1)
    action = FlatActionCodec.decode(next(iter(restored.core._avail_actions(side))))
    first = evaluator.evaluate(restored, replay_memories, side, [action], [3, 4])
    second = evaluator.evaluate(restored, replay_memories, side, [action], [3, 4])
    assert first == second
    assert all(result.battles == 1 for result in first)
    with pytest.raises(ValueError, match='assignment'):
        evaluator.evaluate(restored, replay_memories, 1 - side, [action], [3, 4])
    with pytest.raises(ValueError, match='recurrent state'):
        evaluator.evaluate(restored, list(reversed(replay_memories)), side, [action], [3, 4])
