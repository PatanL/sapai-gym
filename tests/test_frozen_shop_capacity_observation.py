import numpy as np

from sapai import Food, Shop
from sapai_gym1.rl import ParallelSelfPlayEnv, FlatActionCodec


def test_full_frozen_food_shop_has_ten_tokens_and_exact_legal_masks_after_roll():
    env = ParallelSelfPlayEnv(); env.reset(seed=51)
    player = env.core.agent
    player.shop = Shop([Food('apple') for _ in range(10)], turn=11, fixed_rules=False,
                       seed_state=np.random.RandomState(11).get_state())
    player._connect_objects()
    env.core.shop_agent = player.shop
    for slot in player.shop:
        slot.freeze()
    player.roll()
    assert len(player.shop) == 10
    obs = env.observations()[0]
    assert obs['item_ids'].shape == (21,)
    assert sum(obs['entity_mask'][6:16]) == 10
    legal = {FlatActionCodec.decode(index).as_tuple() for index in env.core._avail_actions(0)}
    encoded = {tuple(map(int, row)) for row in np.argwhere(obs['target_mask'])}
    assert encoded == legal
