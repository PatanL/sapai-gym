import numpy as np
import pytest
import torch

from sapai_gym1.rl.puct_search import PUCTConfig, StochasticPUCTSearch
from test_search_horizon import ACTION, HorizonState, ConstantValue


@pytest.mark.parametrize('limit,steps', [(None, 5), (0, 0), (1, 1), (3, 3)])
def test_tree_leaf_limit_does_not_shorten_confirmation(limit, steps):
    config = PUCTConfig(horizon_battles=5, max_rollout_steps=180,
                        tree_leaf_rollout_steps=limit, leaf_value_weight=0.5,
                        selection_only=True)
    planner = StochasticPUCTSearch(ConstantValue(), config)
    state = HorizonState()
    result = planner._policy_rollout(state, torch.zeros(2), np.random.default_rng(1), 5)
    expected = sum(0.9**step for step in range(steps)) + 0.9**steps * 5
    assert result == pytest.approx(expected)
    assert state.battles == steps
    initial = HorizonState()
    planner._expand(initial, torch.zeros(2))
    confirmation = planner._confirmation_rollouts(initial, torch.zeros(2), [ACTION], [11, 12], use_tree=False)
    full = sum(0.9**step for step in range(5)) + 0.9**5 * 5
    assert [row['total'] for row in confirmation] == pytest.approx([full, full])
    assert all(len(row['battle_results']) == 5 for row in confirmation)


def test_zero_leaf_reuses_expansion_value_and_bounds_total_simulation_work():
    class CountingValue(ConstantValue):
        def value(self, state, memory):
            raise AssertionError('Newly expanded value-only leaves must not repeat inference')

    planner = StochasticPUCTSearch(CountingValue(), PUCTConfig(
        tree_leaf_rollout_steps=0, selection_only=True, max_simulator_transitions=1,
    ))
    initial = HorizonState()
    root, _ = planner._expand(initial, torch.zeros(2))
    # A generous limit permits one root edge and one newly evaluated leaf.
    from dataclasses import replace
    planner.config = replace(planner.config, max_simulator_transitions=10)
    assert planner._simulate(initial, torch.zeros(2), 100)
    assert planner.simulator_transitions == 1
    assert root.edges[ACTION].mean_value == pytest.approx(1 + 0.9 * 5)


def test_short_leaves_still_respect_global_transition_budget():
    planner = StochasticPUCTSearch(ConstantValue(), PUCTConfig(
        tree_leaf_rollout_steps=0, selection_only=True, max_simulator_transitions=3,
    ))
    initial = HorizonState()
    planner._expand(initial, torch.zeros(2))
    for _ in range(10):
        planner._simulate(initial, torch.zeros(2), 100)
        assert planner.simulator_transitions <= 3


@pytest.mark.parametrize('limit', [-1, 1.5, True])
def test_invalid_leaf_budget(limit):
    with pytest.raises(ValueError, match='tree_leaf_rollout_steps'):
        StochasticPUCTSearch(ConstantValue(), PUCTConfig(tree_leaf_rollout_steps=limit))
