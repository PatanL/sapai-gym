from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction
from sapai_gym1.rl.cached_tree_evaluation import CachedTreeOpponentEvaluator, PublicObservationView
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy, ParallelSelfPlayEnv
from sapai_gym1.rl.decision_manifest import observation_hash
from sapai_gym1.rl.historical_evaluation import HistoricalOpponentEvaluator


def observation():
    return {
        'x': np.array([0.]), 'type_mask': np.ones(9, dtype=np.int8),
        'source_mask': np.ones((9, 11), dtype=np.int8),
        'target_mask': np.ones((9, 11, 6), dtype=np.int8),
    }


class PublicOnlyTree:
    def __init__(self, policy, action):
        self.evaluator = SimpleNamespace(policy=policy)
        self.nodes = {}
        self.action = action
        self.calls = 0

    def tree_action(self, state, memory):
        assert isinstance(state, PublicObservationView)
        assert not hasattr(state, 'core')
        with pytest.raises(ValueError):
            state.observation()['x'][0] = 99.
        self.calls += 1
        return (self.action, torch.full_like(memory, 99.)) if self.action else None


@pytest.mark.parametrize('side', [0, 1])
def test_cached_actions_only_affect_own_side_and_keep_live_memory(side):
    policy = SimpleNamespace(config=SimpleNamespace(memory_dim=2))
    raw = AutoregressiveAction(ActionType.ROLL)
    planned = AutoregressiveAction(ActionType.FREEZE, 0)
    tree = PublicOnlyTree(policy, planned)
    evaluator = CachedTreeOpponentEvaluator(policy, policy, tree=tree, acting_side=side)
    obs, memory, next_memory = observation(), torch.zeros(2), torch.ones(2)
    assert evaluator._continuation_choice(obs, memory, 1 - side, raw, next_memory) == (raw, next_memory)
    assert tree.calls == 0
    action, returned_memory = evaluator._continuation_choice(obs, memory, side, raw, next_memory)
    assert action == planned and returned_memory is next_memory
    assert evaluator.counters['cacheHits'] == evaluator.counters['changedActions'] == 1
    assert obs['x'][0] == 0.
    obs['target_mask'][6, 0, 5] = 0
    with pytest.raises(RuntimeError, match='legal masks'):
        evaluator._continuation_choice(obs, memory, side, raw, next_memory)


def test_cache_miss_falls_back_and_foreign_policy_is_rejected():
    policy = SimpleNamespace(config=SimpleNamespace(memory_dim=2))
    other = SimpleNamespace(config=SimpleNamespace(memory_dim=2))
    tree = PublicOnlyTree(policy, None)
    with pytest.raises(ValueError, match='different policy'):
        CachedTreeOpponentEvaluator(other, other, tree=tree, acting_side=0)
    evaluator = CachedTreeOpponentEvaluator(policy, other, tree=tree, acting_side=0)
    raw = AutoregressiveAction(ActionType.ROLL)
    memory = torch.zeros(2)
    assert evaluator._continuation_choice(observation(), memory, 0, raw, memory) == (raw, memory)
    assert not evaluator.counters['cacheHits']


def test_tiny_memory_differences_are_counted_not_silently_reused():
    policy = SimpleNamespace(config=SimpleNamespace(memory_dim=2))
    tree = PublicOnlyTree(policy, None)
    obs = observation()
    tree.nodes['node'] = SimpleNamespace(visits=1, observation=obs, memory=torch.zeros(2))
    evaluator = CachedTreeOpponentEvaluator(policy, policy, tree=tree, acting_side=0)
    raw = AutoregressiveAction(ActionType.ROLL)
    memory = torch.full((2,), 1e-7)
    assert evaluator._continuation_choice(obs, memory, 0, raw, memory)[0] == raw
    assert evaluator.counters['observationMatchesWithoutExactMemory'] == 1
    assert evaluator.counters['nearNumericalMemoryMisses'] == 1


def test_opt_in_numerical_match_requires_agreement_and_exact_public_state():
    policy = SimpleNamespace(config=SimpleNamespace(memory_dim=2))
    raw = AutoregressiveAction(ActionType.ROLL)
    planned = AutoregressiveAction(ActionType.FREEZE, 0)
    tree = PublicOnlyTree(policy, None)
    obs = observation()

    def node(action, offset=0.):
        return SimpleNamespace(visits=4, observation=obs, memory=torch.full((2,), offset),
                               priors={action: 1.}, edges={action: SimpleNamespace(visits=4, mean_value=1.)})

    tree.nodes = {'one': node(planned)}
    evaluator = CachedTreeOpponentEvaluator(policy, policy, tree=tree, acting_side=0,
                                            numerical_memory_tolerance=2e-6)
    memory = torch.full((2,), 5e-7)
    assert evaluator._continuation_choice(obs, memory, 0, raw, memory)[0] == planned
    assert evaluator.counters['numericalCacheHits'] == 1
    distant = torch.full((2,), 2e-5)
    assert evaluator._continuation_choice(obs, distant, 0, raw, distant)[0] == raw
    changed = observation()
    changed['x'][0] = 1.
    assert evaluator._continuation_choice(changed, memory, 0, raw, memory)[0] == raw
    tree.nodes['two'] = node(raw, 1e-7)
    ambiguous = CachedTreeOpponentEvaluator(policy, policy, tree=tree, acting_side=0,
                                            numerical_memory_tolerance=2e-6)
    assert ambiguous._continuation_choice(obs, memory, 0, raw, memory)[0] == raw
    assert ambiguous.counters['ambiguousNumericalMatches'] == 1
    with pytest.raises(ValueError, match='tolerance'):
        CachedTreeOpponentEvaluator(policy, policy, tree=tree, acting_side=0,
                                     numerical_memory_tolerance=1.)


@pytest.mark.parametrize('side', [0, 1])
def test_no_hit_execution_matches_raw_and_never_overrides_forced_root(side):
    env = ParallelSelfPlayEnv()
    env.reset(seed=1501)
    env.core.MAX_TURN = 2
    torch.manual_seed(1502)
    policy = EntityTransformerPolicy(EntityTransformerConfig(
        num_items=env.encoder.num_items, num_statuses=env.encoder.num_statuses,
        d_model=32, nhead=4, num_layers=1, dim_feedforward=64, memory_dim=24,
    )).eval()
    seen = []

    class RecordingTree:
        evaluator = SimpleNamespace(policy=policy)
        nodes = {}

        def tree_action(self, state, memory):
            assert isinstance(state, PublicObservationView)
            seen.append(observation_hash(state.observation()))
            return None

    memories = [policy.initial_memory(1)[0] for _ in range(2)]
    args = (env, memories, side, [AutoregressiveAction(ActionType.ROLL)], [501, 502])
    baseline = HistoricalOpponentEvaluator(policy, policy, acting_side=side, horizon_battles=1)
    planned = CachedTreeOpponentEvaluator(policy, policy, tree=RecordingTree(),
                                          acting_side=side, horizon_battles=1)
    assert baseline.evaluate(*args) == planned.evaluate(*args)
    assert seen
    assert observation_hash(env.observations()[side]) not in seen
    assert not planned.counters['changedActions']
