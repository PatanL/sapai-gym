"""Search and confirmation must optimize the same discounted horizon."""

import numpy as np
import pytest
import torch

from sapai_gym1.rl import ActionType, AutoregressiveAction
from sapai_gym1.rl.puct_search import PUCTConfig, StochasticPUCTSearch
from sapai_gym1.rl.rollout_search import (
    CounterfactualRolloutSearch, PolicyEvaluation, RolloutSearchConfig,
    SearchTransition,
)


ACTION = AutoregressiveAction(ActionType.END_TURN)


class HorizonState:
    def __init__(self, battles=0, terminal_at=None):
        self.battles = battles
        self.terminal_at = terminal_at
        self.terminated = False
        self.opponent_board_belief = None

    def set_opponent_board_belief(self, belief):
        self.opponent_board_belief = belief

    def clone(self):
        result = HorizonState(self.battles, self.terminal_at)
        result.terminated = self.terminated
        return result

    def observation(self):
        return {"battle": np.array([self.battles])}

    def legal_actions(self):
        return [ACTION]

    def step(self, action, rng):
        assert action == ACTION
        self.battles += 1
        self.terminated = self.battles == self.terminal_at
        return SearchTransition(1.0, 0.0 if self.terminated else 0.9, True, 0)


class ConstantValue:
    def evaluate(self, state, memory):
        return PolicyEvaluation({ACTION: 1.0}, 10.0, torch.zeros(2))

    def choose(self, state, memory, deterministic=True):
        return ACTION, torch.zeros(2), 10.0

    def value(self, state, memory):
        assert not state.terminated, "Terminal states must never bootstrap"
        return 10.0


def test_cached_tree_node_restores_its_belief_for_a_new_rollout():
    planner = StochasticPUCTSearch(ConstantValue(), PUCTConfig(selection_only=True))
    state = HorizonState()
    memory = torch.zeros(2)
    node, _ = planner._expand(state, memory)
    marker = object()
    node.opponent_belief = marker
    fresh = state.clone()
    assert fresh.opponent_board_belief is None
    reused, expanded = planner._expand(fresh, memory)
    assert reused is node and not expanded
    assert fresh.opponent_board_belief is marker


@pytest.mark.parametrize("initial_battles", [0, 3])
@pytest.mark.parametrize("terminal", [False, True])
def test_root_tree_and_confirmation_have_the_same_horizon(initial_battles, terminal):
    initial = HorizonState(initial_battles, initial_battles + 2 if terminal else None)
    evaluator = ConstantValue()
    memory = torch.zeros(2)
    expected = 1 + 0.9 + (0 if terminal else 0.9**2 * 0.5 * 10)
    root = CounterfactualRolloutSearch(
        evaluator, RolloutSearchConfig(horizon_battles=2, leaf_value_weight=0.5)
    )
    root.simulator_transitions = 0
    records = root._rollouts(initial, [ACTION], memory, [13, 17])
    assert [record["total"] for record in records] == pytest.approx([expected] * 2)
    assert [record["battles"] for record in records] == [initial_battles + 2] * 2
    monte_carlo = root.rollout_state_value(
        initial, memory, rollouts=2, horizon_battles=2, seed=13
    )
    assert monte_carlo.returns == pytest.approx([1.9] * 2)

    tree = StochasticPUCTSearch(
        evaluator,
        PUCTConfig(horizon_battles=2, leaf_value_weight=0.5, selection_only=True),
    )
    node, _ = tree._expand(initial, memory)
    # First traversal uses a leaf rollout; later traversals reach the horizon
    # in the tree itself. Both paths must include exactly one discounted tail.
    for _ in range(4):
        tree._simulate(initial, memory, world_seed=13)
        assert node.edges[ACTION].mean_value == pytest.approx(expected)
    for use_tree in (False, True):
        confirmation = tree._confirmation_rollouts(
            initial, memory, [ACTION], [13, 17], use_tree=use_tree
        )
        assert [record["total"] for record in confirmation] == pytest.approx([expected] * 2)
