import pytest
from sapai_gym1.replay.align_constraints import compatible,monotone_filter,align

def test_global_monotone_filter_preserves_ambiguity():
    assert monotone_filter([[1,5],[],[3],[2,4,6]])==[[1],[],[3],[4,6]]
    assert monotone_filter([[1,2],[2,3]])==[[1,2],[2,3]]
    assert monotone_filter([[5],[3]]) is None
    assert monotone_filter([[],[]])==[[],[]]

def state():
    return {'teamTopologyKnown':True,'gold':3,'turn':2,'teamIds':['id',None,None,None,None],
            'teamIdentityAnnotations':[{'name':'Ant'},None,None,None,None],
            'teamExperience':[0,None,None,None,None]}

def scene():
    return {'turn':2,'gold':3,'candidates':[{'bank':'team','screenSlot':0,'occupancy':'stats_detected',
             'species':'pet-ant','level':1,'experience':0,'levelConfidence':1}]}

def test_checked_fields_not_full_state_certification():
    m=compatible(state(),scene())
    assert m and not m['trainingApproved']
    assert 'attack_health_values' in m['unverifiedFields']
    assert compatible({**state(),'gold':4},scene()) is None
    assert compatible({**state(),'teamTopologyKnown':False},scene()) is None
    assert compatible({**state(),'phase':'battle_not_shop'},scene()) is None

def test_experience_is_absolute_not_relative():
    s=state();s['teamExperience'][0]=3
    f=scene();f['candidates'][0].update(level=2,experience=1)
    assert compatible(s,f)
    f['candidates'][0]['experience']=0
    assert compatible(s,f) is None

def test_wrong_player_rejected():
    with pytest.raises(ValueError):align({'participationId':'a'},{'job':{'participationId':'b'}})
