import importlib
import numpy as np
import pytest
from sapai import Team
from sapai_gym1.rl.battle_replay import capture_battle_playback


@pytest.mark.parametrize('front,back', [('pet-deer','pet-ox'),('pet-ant','pet-tiger'),('pet-hedgehog','pet-camel')])
@pytest.mark.parametrize('seed',[12,123,901])
def test_capture_preserves_outcome_teams_and_rng(front,back,seed):
    module=importlib.import_module('sapai_gym1.rl.selfplay_env')
    engine=importlib.import_module('sapai.battle')
    original=module.Battle; phase=engine.battle_phase
    teams=(Team([front,back]),Team(['pet-fish','pet-blowfish','pet-cricket']))
    baseline=original(*teams,seed=seed,fast_training=True);result=baseline.battle()
    with capture_battle_playback() as records:
        captured=module.Battle(*teams,seed=seed,fast_training=True)
        assert captured.battle()==result
    assert module.Battle is original and engine.battle_phase is phase
    assert len(records)==1 and records[0]['events'][0]['kind']=='initial'
    assert records[0]['events'][-1]['kind']=='complete'
    assert baseline.t0.state==captured.t0.state
    assert baseline.t1.state==captured.t1.state
    for a,b in zip(baseline.rs.get_state(),captured.rs.get_state()):
        np.testing.assert_equal(a,b)


def test_capture_restores_hooks_after_exception():
    module=importlib.import_module('sapai_gym1.rl.selfplay_env');old=module.Battle
    with pytest.raises(RuntimeError):
        with capture_battle_playback():
            raise RuntimeError('cancelled')
    assert module.Battle is old
