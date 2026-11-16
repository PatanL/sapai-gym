"""A league rotation cannot change the opponent halfway through a match."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sapai_gym1.rl.distributed import DistributedSelfPlayRuntime


class TaggedPolicy(torch.nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = torch.nn.Parameter(torch.tensor(float(tag)))
        self.config = SimpleNamespace(memory_dim=2)

    def forward(self, observations, memory, *, entropy_mode_override=None):
        assert entropy_mode_override == 'legacy_path'
        size = memory.shape[0]
        return SimpleNamespace(
            actions=torch.tensor([[0, 10, 5]]).repeat(size, 1),
            log_prob=torch.zeros(size),
            memory=memory + self.tag,
        )


def test_retired_policy_serves_its_leased_game_and_is_released_afterward():
    runtime = DistributedSelfPlayRuntime(
        TaggedPolicy(1), num_actors=1, games_per_actor=1,
        sample_observation={"x": np.zeros(1, dtype=np.float32)},
        device="cpu", queue_size=2,
    )
    try:
        old, new = TaggedPolicy(7), TaggedPolicy(8)
        runtime.set_policy_pool({7: old}, main_probability=0)
        router = runtime.policy_router
        rng = np.random.default_rng(131)
        assert router.sample(rng, lease_index=0) == 7
        shared = runtime.actor_buffers[0]
        # An opponent that has ended its shop still owns its lease while it
        # waits for battle and the next turn, even with no inference request.
        shared.active.zero_()
        runtime.set_policy_pool({8: new}, main_probability=0)
        assert runtime.batcher.resident_policies[7] is old
        assert list(router.policy_ids[:int(router.count[0])]) == [8]
        with pytest.raises(ValueError, match="leased"):
            runtime.set_policy_pool({7: TaggedPolicy(70)}, main_probability=0)
        shared.active[1] = 1
        shared.policy_ids[1] = 7
        shared.memories[1].fill_(100)
        runtime.batcher._serve([0])
        torch.testing.assert_close(shared.next_memories[1], torch.tensor([107., 107.]))
        assert int(shared.policy_versions[1]) == -1
        # The next match replaces the old lease atomically with its sample.
        assert router.sample(rng, lease_index=0) == 8
        runtime.set_policy_pool({8: new}, main_probability=0)
        assert 7 not in runtime.batcher.resident_policies
    finally:
        runtime.trajectory_ring.close()
        runtime.result_queue.cancel_join_thread()
        runtime.result_queue.close()
