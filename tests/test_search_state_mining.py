from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy, ParallelSelfPlayEnv
from sapai_gym1.rl.decision_manifest import decision_record, restore_decision, observation_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_state_mining import mine_match_opportunity, matches_gold_band


@pytest.mark.parametrize("band", ["zero", "low", "funded"])
def test_mining_reaches_each_gold_band_with_matching_snapshot_memory(band):
    torch.manual_seed(973)
    encoder = ParallelSelfPlayEnv().encoder
    policy = EntityTransformerPolicy(EntityTransformerConfig(
        num_items=encoder.num_items, num_statuses=encoder.num_statuses,
        d_model=32, nhead=4, num_layers=1, dim_feedforward=64, memory_dim=24,
    )).eval()
    rng = np.random.default_rng(39111)
    selected = None
    for _ in range(6):
        selected = mine_match_opportunity(
            policy, rng, torch.device("cpu"), 2, 2, band, with_replay=True
        )
        if selected is not None:
            break
    assert selected is not None
    env, memories, side, reasons, replay = selected
    player = env.core.agent if side == 0 else env.core.opponent
    assert env.core.turn == 2
    assert not player.turn_ended
    assert matches_gold_band(player.gold, band)
    assert reasons
    assert all(memory.shape == (24,) and torch.isfinite(memory).all() for memory in memories)
    if band == "zero":
        assert "zero-gold-freeze-decision" in reasons
        assert (env.core.shop_action_count_agent if side == 0 else env.core.shop_action_count_opponent) > 0
    # The returned decision remains actionable after mining the rest of the
    # match, and its cloned legality masks survive a second clone.
    cloned = env.clone()
    for original, copied in zip(env.observations(), cloned.observations()):
        for field in ("type_mask", "source_mask", "target_mask"):
            np.testing.assert_array_equal(original[field], copied[field])
    record = decision_record(
        env, memories, side, reasons, replay,
        search_seed=1, validation_seed=2, truth_seeds=[3, 4],
    )
    restored, restored_memories, restored_side, _ = restore_decision(record)
    assert restored_side == side
    assert [observation_hash(obs) for obs in restored.observations()] == record["observationHashes"]
    for left, right in zip(memories, restored_memories):
        torch.testing.assert_close(left, right)
    corrupted = {**record, "observationHashes": ["wrong", "wrong"]}
    with pytest.raises(ValueError, match="differs"):
        restore_decision(corrupted)
