import json
from pathlib import Path
import shutil
import cv2
import pytest

from sapai_gym1.replay.archive import atomic_json,sha_file
from sapai_gym1.replay.visual_capture import ReplayScreen,coverage
from sapai_gym1.replay.visual_parse import parse_capture,fingerprint

ROOT=Path(__file__).resolve().parents[2]


def test_fingerprint_separates_zero_gold_stat_change():
    row={'observation':{'turn':{'value':4},'gold':{'value':0}}}
    item={'attack':2,'health':3,'level':1,'experience':0}
    assert fingerprint(row,[item])!=fingerprint(row,[{**item,'health':4}])


def test_actual_vision_cpu_smoke_no_training_export(tmp_path):
    screen=ReplayScreen(ROOT,ROOT/'runs/expert-replay-ingest-20260926-v1/UI_PROFILE.json')
    capture=tmp_path/'capture';capture.mkdir()
    source=ROOT/'runs/expert-replay-reconstruction-20260925-v1/capture-v5/frame-00006.jpg'
    im=cv2.imread(str(source));observation=screen.inspect(im)
    assert observation['shop']
    frames=[]
    for i in range(2):
        path=capture/f'frame-{i:05d}.jpg';shutil.copyfile(source,path)
        frames.append({'frame':path.name,'sha256':sha_file(path),'seconds':i*.2,'observation':observation})
    index={'job':{'fixture':True},'frames':frames,'endReason':'fixture_excerpt',
           'coverage':coverage([1],frames,False)}
    atomic_json(capture/'INDEX.json',index)
    report=parse_capture(ROOT,capture,tmp_path/'parsed',max_keyframes=1)
    assert report['stats']['numericScenes']==1
    assert report['stats']['identityScenes']==1
    assert not report['trainingApproved']
    assert not report['captureCoverage']['turnCoverageComplete']
    assert all(not i.get('replayDomainCalibrated',False) for s in report['scenes'] for i in s['candidates'])
    with pytest.raises(ValueError):parse_capture(ROOT,capture,tmp_path/'parsed')
    frames[0]['sha256']='corrupt'
    atomic_json(capture/'INDEX.json',{**index,'frames':frames})
    with pytest.raises(ValueError,match='Frame hash changed'):
        parse_capture(ROOT,capture,tmp_path/'corrupt')


def test_running_capture_rejected(tmp_path):
    capture=tmp_path/'capture';capture.mkdir()
    atomic_json(capture/'INDEX.json',{'endReason':'running'})
    with pytest.raises(ValueError,match='still running'):
        parse_capture(ROOT,capture,tmp_path/'parsed')
