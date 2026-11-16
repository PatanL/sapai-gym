from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sapai_gym1.rl.training import BatchedSelfPlayCollector
from sapai_gym1.rl.selfplay_env import SelfPlayStep


class TaggedPolicy(torch.nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.config = SimpleNamespace(memory_dim=2)
        self.tag = tag
        self.inputs = []

    def forward(self, observations, memory, *, entropy_mode_override=None):
        assert entropy_mode_override == 'legacy_path'
        self.inputs.append(memory.clone())
        return SimpleNamespace(
            actions=torch.tensor([[0, 10, 5]]).repeat(len(memory), 1),
            memory=memory + self.tag,
            log_prob=torch.zeros(len(memory)),
        )


class TwoBattleGame:
    def reset(self, seed=None):
        self.battles = 0
        return self.observations()

    def observations(self):
        return tuple({'x': np.array([self.battles], dtype=np.float32)} for _ in range(2))

    def active_players(self):
        return True, True

    def step(self, first, second):
        self.battles += 1
        return SelfPlayStep(self.observations(), (0., 0.), True, self.battles == 2, 2)


@pytest.mark.parametrize('replacement_is_mirror', [False, True])
def test_frozen_opponent_is_retained_until_match_end(replacement_is_mirror):
    main, old, new = TaggedPolicy(1), TaggedPolicy(7), TaggedPolicy(8)
    collector = BatchedSelfPlayCollector(main, 1, 'cpu')
    game = TwoBattleGame()
    collector.envs = [game]
    collector.observations = [game.reset()]
    collector.set_opponent_policy(old)
    assert collector.game_opponents == [old]
    assert not collector.learn_from_resident_sides
    collector.collect(1)
    assert collector.game_started == [True]
    replacement = None if replacement_is_mirror else new
    collector.set_opponent_policy(replacement)
    assert collector.game_opponents == [old]
    assert len(old.inputs) == 1
    collector.collect(1)
    assert len(old.inputs) == 2
    torch.testing.assert_close(old.inputs[1], torch.full((1, 2), 7.))
    assert not new.inputs
    assert collector.game_opponents == [replacement]
    assert collector.game_started == [False]
    collector.collect(1)
    assert len(old.inputs) == 2
    if replacement_is_mirror:
        # Both sides start the new game under main, with fresh hidden states.
        torch.testing.assert_close(main.inputs[-1], torch.zeros((2, 2)))
    else:
        torch.testing.assert_close(new.inputs[0], torch.zeros((1, 2)))
