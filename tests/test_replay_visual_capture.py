from pathlib import Path
import cv2
import pytest
from sapai_gym1.replay.visual_capture import ReplayScreen, coverage, response_shape

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'runs/expert-replay-ingest-20260926-v1/UI_PROFILE.json'

def test_profile_hashes_and_positive_negative_controls():
    screen=ReplayScreen(ROOT,PROFILE)
    menu=cv2.imread(str(ROOT/'runs/expert-replay-ingest-20260926-v1/browser-initial.png'))
    assert screen.visible(menu,'main_history')
    assert not screen.visible(menu,'shop_controls')
    assert not screen.visible(menu,'watch_replay')
    for name,ref in screen.profile['references'].items():
        im=cv2.imread(str(ROOT/ref['file']))
        assert screen.visible(im,name),name

def row(turn):
    return {'observation':{'shop':True,'turn':{'value':turn,'confidence':1}}}

@pytest.mark.parametrize('last',[3,11,14,20])
def test_no_hardcoded_last_turn(last):
    expected=list(range(1,last+1))
    frames=[row(t) for t in expected for _ in range(2)]
    assert coverage(expected,frames,True)['turnCoverageComplete']
    assert not coverage(expected,frames,False)['turnCoverageComplete']
    assert not coverage(expected,frames[:-1],True)['turnCoverageComplete']
    assert not coverage(expected,frames,True)['actionCoverageValidated']

def test_no_empty_completion():
    assert not coverage([],[],True)['turnCoverageComplete']

def test_network_diagnostics_do_not_capture_values():
    import json
    shape=response_shape({'Token':'secret-value','Replay':{'Actions':[{'name':'private-value'}]}})
    encoded=json.dumps(shape)
    assert 'secret-value' not in encoded and 'private-value' not in encoded
    assert shape['Replay']['Actions']['count']==1
