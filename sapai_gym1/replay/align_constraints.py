"""Monotone native/visual candidate alignment with explicit ambiguity.

Matching gold, roster, species and XP is NOT proof that unverified numeric
stats/perks/shop offers are correct. No arbitrary maximum-probability path or
training labels are selected here. All surviving compatible scenes are retained.
"""
from __future__ import annotations

from collections import defaultdict


def compatible(state,scene):
    if state.get('phase')=='battle_not_shop':return None
    if not state['teamTopologyKnown'] or state['gold'] is None:return None
    if state['turn']!=scene['turn'] or state['gold']!=scene['gold']:return None
    slots={x['screenSlot']:x for x in scene['candidates'] if x['bank']=='team'}
    matches=['turn','gold'];unknown=[]
    for slot,uid in enumerate(state['teamIds']):
        visible=slots.get(slot,{})
        detected=visible.get('occupancy')=='stats_detected'
        if uid is None:
            if detected:return None
            # Absence of readable badges is not an independent empty-slot proof.
            unknown.append(f'team[{slot}].visual_empty_proof')
            continue
        if not detected:return None
        matches.append(f'team[{slot}].visible_stat_badges')
        identity=state['teamIdentityAnnotations'][slot]
        species=visible.get('species')
        if species and identity and identity['name']:
            expected='pet-'+identity['name'].lower().replace(' ','-')
            if expected!=species:return None
            matches.append(f'team[{slot}].species_candidate')
        else:unknown.append(f'team[{slot}].species_pixels')
        xp=state['teamExperience'][slot]
        lv=visible.get('level');progress=visible.get('experience')
        if xp is not None and lv in (1,2,3) and progress is not None and visible.get('levelConfidence',0)>=.75:
            if xp!={1:0,2:2,3:5}[lv]+progress:return None
            matches.append(f'team[{slot}].experience_candidate')
        else:unknown.append(f'team[{slot}].experience_pixels')
    return {'matchedFields':matches,'unverifiedFields':unknown+['attack_health_values','perks','full_shop_inventory','freeze_screen_mapping'],
            'trainingApproved':False}


def monotone_filter(candidate_lists):
    """Retain values occurring in some nondecreasing assignment, skip empty rows.

If nonempty rows contradict each other's temporal order, return no asserted
alignment rather than silently discard one of the conflicting constraints.
"""
    active=[(i,sorted(set(xs))) for i,xs in enumerate(candidate_lists) if xs]
    earliest=[];previous=-1
    for _,xs in active:
        legal=[x for x in xs if x>=previous]
        if not legal:return None
        previous=legal[0];earliest.append(previous)
    latest=[0]*len(active);following=float('inf')
    for k in range(len(active)-1,-1,-1):
        legal=[x for x in active[k][1] if x<=following]
        if not legal:return None
        following=legal[-1];latest[k]=following
    result=[[] for _ in candidate_lists]
    for k,(original,xs) in enumerate(active):
        low=earliest[k-1] if k else -1
        high=latest[k+1] if k+1<len(active) else float('inf')
        result[original]=[x for x in xs if low<=x<=high]
    return result


def align(trace,visual):
    if trace['participationId']!=visual['job']['participationId']:
        raise ValueError('Different participation IDs')
    bad_turns={c['turn'] for c in trace.get('endpointChecks',[]) if any(c.get(k) is False for k in ('match','experienceMatch','goldMatch','spendingMatch'))}
    scene_by_id={s['scene']:s for s in visual['scenes'] if s['turn']>0}
    byturn=defaultdict(list)
    for scene in scene_by_id.values():byturn[scene['turn']].append(scene)
    turns=defaultdict(list)
    for action in trace['actions']:
        if action['action']!='start_turn':turns[action['turn']].append(action)
    output=[];conflicts=[]
    for turn,actions in sorted(turns.items()):
        boundaries=[]
        for action in actions:
            for key in ('before','after'):
                matches={}
                for scene in (byturn[turn] if turn not in bad_turns else []):
                    match=compatible(action[key],scene)
                    if match:matches[scene['scene']]=match
                boundaries.append({'index':action['index'],'side':key,'matches':matches})
        filtered=monotone_filter([list(b['matches']) for b in boundaries])
        if turn in bad_turns:
            filtered=None;conflicts.append({'turn':turn,'reason':'native_endpoint_constraint_failed'})
        elif filtered is None:conflicts.append({'turn':turn,'reason':'no_joint_monotone_assignment_for_nonempty_boundaries'})
        for i,action in enumerate(actions):
            row={'index':action['index'],'turn':turn,'action':action['action'],
                 'sourceId':action['sourceId'],'sourceIdentityAnnotation':action['sourceIdentityAnnotation'],
                 'nativeActionEvidence':action['nativeActionEvidence'],'trainingApproved':False}
            for offset,key in enumerate(('before','after')):
                boundary=boundaries[2*i+offset]
                surviving=filtered[2*i+offset] if filtered is not None else []
                candidates=[]
                for sid in surviving:
                    scene=scene_by_id[sid]
                    candidates.append({'scene':sid,'firstFrame':scene['firstFrame'],'lastFrame':scene['lastFrame'],
                                       'identityFrame':scene.get('identityFrame'),**boundary['matches'][sid]})
                row[key]={'nativeFacts':action[key],'rawCompatibleScenes':sorted(boundary['matches']),
                          'candidates':candidates,'status':('temporal_conflict' if filtered is None else
                            'unmatched' if not candidates else 'unique_by_checked_fields_only' if len(candidates)==1 else 'ambiguous')}
            if action['after'].get('phase')=='battle_not_shop':
                row['after']['nativeEndEvidence']=[d for d in action['details'] if isinstance(d,dict) and 'afterEndAnchor' in d]
            output.append(row)
    return {'schema':'sap-native-visual-alignment-candidates-v1','participationId':trace['participationId'],
            'actions':output,'temporalConflicts':conflicts,'trainingApproved':False,
            'summary':{'transactions':len(output),'withBeforeCandidates':sum(bool(a['before']['candidates']) for a in output),
                       'withAfterCandidates':sum(bool(a['after']['candidates']) for a in output),
                       'uniqueBeforeByCheckedFieldsOnly':sum(a['before']['status']=='unique_by_checked_fields_only' for a in output),
                       'turnsWithTemporalConflicts':len(conflicts),'fullyValidatedBeforeAfterPairs':0},
            'warning':'These are compatibility constraints, not exact action frames or validated complete observations. No stats are copied from later native endpoints into earlier frames.'}
