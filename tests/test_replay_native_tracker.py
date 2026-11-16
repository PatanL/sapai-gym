from copy import deepcopy
import json
from pathlib import Path
import pytest
from sapai_gym1.replay.evidence import EvidenceParser
from sapai_gym1.replay.native_tracker import NativeTopologyTracker

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def evidence():
    p=ROOT/'runs/expert-replay-reconstruction-20260925-v1'
    selection=json.loads((p/'SELECTION.json').read_text())
    return EvidenceParser(ROOT/'SAP-Calculator')({'replayId':selection['metadata']['id'],'metadata':selection['metadata'],
        'perspectives':json.loads((p/'perspectives.json').read_text()),'turns':json.loads((p/'turns.json').read_text())})['perspectives'][0]

def test_fixture_topology_matches_hand_checked_trace(evidence):
    result=NativeTopologyTracker(evidence).run()
    truth=json.loads((ROOT/'runs/expert-replay-action-reconstruction-20260925-v1/FINAL.json').read_text())
    expected={a['index']:a for a in truth['actions']}
    checked=0
    for row in result['actions']:
        if row['index'] not in expected:continue
        for key in ('before','after'):
            state=row[key]
            if not state['teamTopologyKnown']:continue
            ids=[int(x.split(':')[-1]) if x else None for x in state['teamIds']]
            assert ids==[p['id'] if p else None for p in expected[row['index']][key]['teamBackToFront']],(row['index'],key)
            checked+=1
            assert not state['currentStatsKnown']
            if state['gold'] is not None:
                assert state['gold']==expected[row['index']][key]['gold'],(row['index'],key)
    assert checked>100
    assert result['summary']['mismatchedEndpoints']==0
    assert {r['index'] for r in result['actions'] if r['action']=='buy_and_combine'}=={66,104,113,116}
    assert result['summary']['completePolicyObservations']==0

def test_pill_not_assumed_empty_and_end_is_after_only(evidence):
    rows={r['index']:r for r in NativeTopologyTracker(evidence).run()['actions']}
    assert not rows[88]['after']['teamTopologyKnown']
    end=rows[94]
    assert end['after']['phase']=='battle_not_shop'
    assert any(d.get('notBeforeEndState') for d in end['details'] if isinstance(d,dict))

def test_freeze_order_not_assumed_screen_order(evidence):
    rows=NativeTopologyTracker(evidence).run()['actions']
    assert all(not r['before']['shopScreenPositionsKnown'] for r in rows)
    assert all(x['screenPosition'] is None for r in rows for x in r['before']['shopObjects'].values())

def test_conflicting_order_roster_not_silently_repaired(evidence):
    tracker=NativeTopologyTracker(evidence)
    tracker.team=['board:1',None,None,None,None]
    tracker.apply_orders([{'MinionId':{'BoId':'board','Uni':2},'Point':{'x':1}}],5)
    assert not tracker.topology_known
    assert tracker.issues[0]['reason']=='order_batch_roster_conflict'

def test_input_unchanged_and_no_simulation_import(evidence):
    original=deepcopy(evidence)
    NativeTopologyTracker(evidence).run()
    assert evidence==original

def test_food_price_corruption_trips_spending_check(evidence):
    food=next(a for a in evidence['actions'] if a.get('consumedFood'))
    food['consumedFood']['price']+=1
    result=NativeTopologyTracker(evidence).run()
    assert result['summary']['mismatchedSpendingEndpoints']>=1

def test_ownership_proof_does_not_use_shop_identity_as_team_membership(evidence):
    tracker=NativeTopologyTracker(evidence)
    # These four source IDs disappear in actual full-board buy/merge events;
    # a shop object and buy response alone do not prove team membership.
    for index in (66,104,113,116):
        action=next(a for a in evidence['actions'] if a['index']==index)
        assert action['sourceId'] not in tracker.owned_proofs
        assert tracker.purchase_kind_proof(action)['kind']!='separate'

def test_later_roster_absence_after_sale_is_not_a_merge_proof(evidence):
    tracker=NativeTopologyTracker(evidence)
    tracker.evidence['actions']=[
        {'index':1,'type':6,'sourceId':'unseen','reorderBatch':None},
        {'index':2,'type':9,'reorderBatch':None},
        {'index':3,'type':5,'reorderBatch':[], 'evidence':{'test':True}}]
    assert tracker.purchase_kind_proof(tracker.evidence['actions'][0])['kind']=='unknown'
