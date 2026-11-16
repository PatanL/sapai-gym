from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from v44_preflight import validate_evaluation, validate_smoke, parent_ready, SIMULATOR, source_inventory, check_source, smoke_speed_ratio
from monitor_v44 import phase_name


def complete_report(opponent='opponent.pt'):
    records = [{'pairIndex': index, 'seed': index + 91, 'candidateSide': side,
                'candidateResult': 'win' if side == 0 else 'loss', 'terminationReason': 'lives'}
               for index in range(100) for side in (0, 1)]
    return {'checkpointUpdate': 2250, 'games': 200, 'pairs': 100,
            'runtimeSimulatorRevision': SIMULATOR, 'checkpointSimulatorRevision': SIMULATOR,
            'opponentCheckpoint': opponent, 'pairSeeds': list(range(91, 191)), 'gameResults': records,
            'outcomes': dict(Counter(r['candidateResult'] for r in records)),
            'pairedMeanScore': .5, 'pairedScore95CI': [.45, .55], 'terminationCounts': {'lives': 200}}


def test_complete_report_requires_all_paired_seats_and_matching_rules():
    report = complete_report()
    validate_evaluation(report, 2250, 'opponent.pt')
    for change in ('missing', 'duplicate', 'wrong_rules', 'wrong_score', 'wrong_opponent'):
        bad = copy.deepcopy(report)
        if change == 'missing':
            bad['gameResults'].pop()
        elif change == 'duplicate':
            bad['gameResults'][0] = bad['gameResults'][1]
        elif change == 'wrong_rules':
            bad['runtimeSimulatorRevision'] = 'old-rules'
        elif change == 'wrong_score':
            bad['pairedMeanScore'] = .75
        else:
            bad['opponentCheckpoint'] = 'wrong.pt'
        with pytest.raises(ValueError):
            validate_evaluation(bad, 2250, 'opponent.pt')


def test_parent_waits_for_complete_final_evaluations_and_durable_league(tmp_path):
    assert not parent_ready(tmp_path)
    (tmp_path / 'update-002250.pt').write_bytes(b'checkpoint fixture; tensor load is the GPU smoke gate')
    (tmp_path / 'league.json').write_text(json.dumps({'members': [{'member_id': 'historical-002250'}]}))
    (tmp_path / 'evaluations').mkdir()
    paths = {
        'parent': 'turtle-league-v38-chocolate-legality-recalibration-from-v36-100/update-000250.pt',
        'v31': 'turtle-league-v31-search-integration-from-v25-300/update-001000.pt',
    }
    for panel, path in paths.items():
        (tmp_path / 'evaluations' / f'paired-2250-vs-{panel}.json').write_text(json.dumps(complete_report(path)))
    assert parent_ready(tmp_path)
    (tmp_path / 'league.json').write_text(json.dumps({'members': []}))
    with pytest.raises(ValueError, match='league entry'):
        parent_ready(tmp_path)


def test_smoke_requires_finite_completed_steps_and_real_joint_toggles():
    log = ('action_schema=primitive-plus-end-freeze-v1-20260905\n'
           'update=2253 loss=0.1 policy=-.01 value=.2 entropy=1.3 grad=.3 end_toggles=4\n'
           'finished updates=2253 transitions=100 parameters=14346925 device=cuda\n')
    validate_smoke(log, 2253, joint=True)
    for broken in (log.replace('end_toggles=4', 'end_toggles=0'), log.replace('loss=0.1', 'loss=nan'),
                   log.replace('finished updates=2253', 'stopped before checkpoint'), log.replace('update=2253', 'update=2252')):
        with pytest.raises(ValueError):
            validate_smoke(broken, 2253, joint=True)


def test_health_monitor_distinguishes_verified_wait_from_missing_observation():
    assert phase_name(set(), 'active', 'active') == 'waiting_for_parent'
    assert phase_name(set(), None, None) == 'unknown'
    assert phase_name({'sap-v44-control'}, 'active', 'inactive') == 'control'
    assert phase_name(set(), 'failed', 'inactive') == 'failed'


def test_source_inventory_rejects_code_or_data_drift(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    module = source / 'engine.py'
    module.write_text('RULE = 1\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(source_inventory(source)))
    check_source(source, manifest)
    module.write_text('RULE = 2\n')
    with pytest.raises(ValueError, match='mixed-source'):
        check_source(source, manifest)


def test_gpu_speed_guard_uses_recent_unique_updates_not_repeated_log_lines():
    control = 'update=2251 update_fps=1\nupdate=2252 update_fps=800\nupdate=2253 update_fps=820\n'
    joint = 'update=2252 update_fps=2\nupdate=2252 update_fps=780\nupdate=2253 update_fps=800\n'
    assert smoke_speed_ratio(control, joint) == pytest.approx(790 / 810)
    with pytest.raises(ValueError, match='Insufficient'):
        smoke_speed_ratio(control, 'update=2253 update_fps=800\n')
