import numpy as np
import pytest
import torch

from sapai_gym1.rl import ActionType, AutoregressiveAction
from sapai_gym1.rl.puct_search import PUCTConfig, StochasticPUCTSearch
from sapai_gym1.rl.rollout_search import PolicyEvaluation, SearchTransition


SETUP = AutoregressiveAction(ActionType.SELL, 0)
WAIT = AutoregressiveAction(ActionType.ROLL)
TRY = AutoregressiveAction(ActionType.BUY_PET, 0, 0)


class DelayedLoss:
    """The base policy loses after a long chain; TRY at phase one wins."""
    def __init__(self, phase=0):
        self.phase = phase
        self.battles = 0
        self.terminated = False

    def clone(self):
        cloned = type(self)(self.phase)
        cloned.battles, cloned.terminated = self.battles, self.terminated
        return cloned

    def redeterminize(self, seed):
        pass

    def observation(self):
        return {'phase': np.array([self.phase], dtype=np.int64)}

    def legal_actions(self):
        return [SETUP] if self.phase == 0 else [WAIT, TRY] if self.phase == 1 else [WAIT]

    def step(self, action, rng):
        assert action in self.legal_actions()
        if self.phase == 1 and action == TRY:
            self.terminated, self.battles = True, 1
            return SearchTransition(2., 0., True, 0)
        self.phase += 1
        if self.phase == 1000:
            self.terminated, self.battles = True, 1
            return SearchTransition(-2., 0., True, 1)
        return SearchTransition(0., 1., False)


class ExactBasePolicyValue:
    """V=-2 is exact for the base policy, which always takes WAIT."""
    def evaluate(self, state, memory):
        priors = ({SETUP: 1.} if state.phase == 0 else
                  {WAIT: 0.85, TRY: 0.15} if state.phase == 1 else {WAIT: 1.})
        return PolicyEvaluation(priors, -2., torch.zeros(2))

    def choose(self, state, memory, deterministic=True):
        return (SETUP if state.phase == 0 else WAIT), torch.zeros(2), -2.

    def value(self, state, memory):
        assert not state.terminated
        return -2.


def search(weight):
    planner = StochasticPUCTSearch(ExactBasePolicyValue(), PUCTConfig(
        max_simulations=1000, max_simulator_transitions=40,
        tree_leaf_rollout_steps=0, leaf_value_weight=0.5,
        unvisited_value_weight=weight, cpuct=2., prior_floor=0.,
        selection_only=True, min_root_visits=2,
    ))
    result = planner.search(DelayedLoss(), torch.zeros(2), seed=31)
    assert result.simulator_transitions <= 40
    return planner


def test_consistent_first_play_scale_can_uncover_a_low_prior_winning_followup():
    legacy, scaled = search(None), search(0.5)
    child = DelayedLoss(1)
    assert legacy.tree_action(child, torch.zeros(2))[0] == WAIT
    assert scaled.tree_action(child, torch.zeros(2))[0] == TRY


@pytest.mark.parametrize('weight', [-1., 1.1, float('nan'), True])
def test_invalid_first_play_weights(weight):
    with pytest.raises(ValueError, match='unvisited_value_weight'):
        StochasticPUCTSearch(ExactBasePolicyValue(), PUCTConfig(unvisited_value_weight=weight))
