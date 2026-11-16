from pathlib import Path
import sys

from sapai import Team, Shop, Pet
from sapai_gym1.rl import ParallelSelfPlayEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from search_state_mining import opening_freeze_reasons


def state(experience, offers):
    env = ParallelSelfPlayEnv()
    env.reset(seed=301)
    pet = Pet('fish')
    pet.experience = experience
    env.core.agent.team = Team([pet])
    env.core.agent.shop = Shop([Pet(name) for name in offers], turn=1, fixed_rules=True)
    env.core.agent.gold = 0
    env.core.agent._connect_objects()
    env.core.team_agent = env.core.agent.team
    env.core.shop_agent = env.core.agent.shop
    env.core._agent_actions_cache = None
    return env.core


def test_levelup_and_two_copy_opportunities_are_distinguished():
    assert opening_freeze_reasons(state(1, ['fish']), 0) == ('opening-levelup-copy-available',)
    assert opening_freeze_reasons(state(0, ['fish', 'fish']), 0) == ('opening-two-shop-copies-available',)
    assert not opening_freeze_reasons(state(0, ['fish']), 0)


def test_only_currently_legal_unfrozen_choices_are_selected():
    core = state(1, ['fish'])
    core._freeze_toggled_agent.add(id(core.agent.shop[0].obj))
    core._agent_actions_cache = None
    assert not opening_freeze_reasons(core, 0)
    core = state(1, ['fish'])
    core.agent.shop[0].frozen = True
    assert not opening_freeze_reasons(core, 0)
    core = state(1, ['fish'])
    core.agent.gold = 1
    assert not opening_freeze_reasons(core, 0)
