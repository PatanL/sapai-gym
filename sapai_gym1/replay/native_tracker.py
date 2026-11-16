"""Causal topology/inventory constraints for native version-48 Turtle replays.

This is not a second game simulator. It tracks object IDs and explicit batches,
while invalidating facts when shop abilities or ambiguous insertions can change
them. It never substitutes base pet stats for unseen purchase stats. Native END
boards are AFTER-END anchors, never pre-END policy observations.
"""
from __future__ import annotations

from copy import deepcopy

from .parser import decode_nested_json, object_key


SCHEMA = "sap-native-topology-gold-xp-constraints-v4"


def level(experience):
    return 1 if experience < 2 else 2 if experience < 5 else 3


def board_xp(board):
    return {object_key(item['Id']):int(item.get('Exp',0))
            for item in (board.get('Mins') or {}).get('Items',[]) or [] if item}


def board_team(board):
    result=[None]*5
    for item in (board.get('Mins') or {}).get('Items',[]) or []:
        if item is None:continue
        slot=int((item.get('Poi') or {}).get('x',0))
        uid=object_key(item.get('Id'))
        if uid is None or not 0<=slot<5 or result[slot] is not None:
            raise ValueError('Invalid native team topology')
        result[slot]=uid
    return result


class NativeTopologyTracker:
    def __init__(self,evidence):
        self.evidence=evidence
        self.catalog=deepcopy(evidence['identityCatalog'])
        self.team=[None]*5
        self.topology_known=True
        self.shop={}
        self.shop_universe_known=False
        self.identity_constraints=[]
        self.issues=[]
        self.turn=0
        self.xp={}
        self.gold=None
        self.spent=0
        self.owned_proofs={}
        for action in evidence['actions']:
            ids=[]
            if action['type']==7:ids.extend([action['sourceId'],action['targetId']])
            if action['type']==9:ids.append(action['sourceId'])
            if action['type']==8:ids.append(action['targetId'])
            ids.extend(object_key(x.get('MinionId')) for x in action['reorderBatch'] or [])
            for uid in ids:
                if uid:self.owned_proofs.setdefault(uid,[]).append(action['evidence'])
        for snapshot in evidence['serializedBoards']:
            for uid in [x for x in board_team(snapshot['board']) if x]:
                self.owned_proofs.setdefault(uid,[]).append(snapshot['evidence'])

    def purchase_kind_proof(self,action):
        source=action['sourceId']
        if source in self.owned_proofs:
            return {'kind':'separate','reason':'purchased_ID_is_explicitly_observed_as_owned',
                    'evidence':self.owned_proofs[source],
                    'annotationOnly':'retrospective ownership classification, not future stats or hidden opponent features'}
        # With no intervening roster-changing operation, absence at the next
        # explicit roster proves the purchased ID was consumed by a merge.
        for later in self.evidence['actions']:
            if later['index']<=action['index']:continue
            roster=None;proof=None
            if later['reorderBatch'] is not None:
                roster=[object_key(x.get('MinionId')) for x in later['reorderBatch']]
                proof=later['evidence']
            elif later['type']==11:
                end=next((e for e in self.evidence['endpoints'] if e['turn']==later['turn'] and e['hashMatches']),None)
                snap=next((s for s in self.evidence['serializedBoards'] if end and s['actionIndex']==end['snapshotIndex']),None)
                if snap:roster=board_team(snap['board']);proof=snap['evidence']
            if roster is not None:
                return {'kind':'separate' if source in roster else 'combine',
                        'reason':'next_explicit_roster_without_intervening_removal_or_purchase',
                        'evidence':proof,'annotationOnly':'retrospective action classification'}
            food=later.get('consumedFood')
            if later['type'] not in (5,8,12) or (later['type']==8 and (not food or food['name']=='Sleeping Pill')):
                break
        return {'kind':'unknown','reason':'no_unambiguous_ownership_proof'}

    def identity(self,uid):
        info=self.catalog.get(uid)
        if not info or info.get('ambiguousOrTransformed'):
            return None
        return {"name":info.get('name'),"kind":info['kind'],"enum":info['enum'],
                "annotationEvidence":deepcopy(info['evidence']),
                "mayUseLaterIntrinsicIdentityEvidence":True}

    def snapshot(self):
        return {"turn":self.turn,"teamOrder":"native_back_to_front",
                "teamIds":list(self.team) if self.topology_known else None,
                "lastPartialTeamIds":list(self.team),"teamTopologyKnown":self.topology_known,
                "teamIdentityAnnotations":[self.identity(uid) if uid else None for uid in self.team],
                "teamExperience":[self.xp.get(uid) if uid else None for uid in self.team],
                "shopObjects":deepcopy(self.shop),"shopObjectUniverseKnown":self.shop_universe_known,
                "shopScreenPositionsKnown":False,"gold":self.gold,"goldSpent":self.spent,"currentStatsKnown":False,
                "completePolicyObservation":False}

    def spend(self,amount,index):
        self.spent+=amount
        if self.gold is not None:
            self.gold-=amount
            if self.gold<0:
                self.issues.append({'index':index,'reason':'derived_negative_gold','gold':self.gold})
                self.gold=None

    def merge_xp(self,source,target,shop=False):
        sx=0 if shop else self.xp.get(source)
        tx=self.xp.get(target)
        self.xp[target]=min(5,tx+sx+1) if sx is not None and tx is not None else None

    def annotate_same_species(self,source,target,index):
        self.identity_constraints.append({"source":source,"target":target,"index":index,
                                          "reason":"full_team_occupied_destination_successful_buy_pet"})
        known=self.catalog.get(target)
        if known and not known.get('ambiguousOrTransformed') and source not in self.catalog:
            self.catalog[source]={**deepcopy(known),"derivedViaFullBoardBuyCombine":index}

    def apply_orders(self,orders,index):
        if orders is None:return
        arranged=[None]*5
        seen=set()
        for entry in orders:
            uid=object_key(entry.get('MinionId'))
            slot=int((entry.get('Point') or {}).get('x',0))
            if uid is None or uid in seen or not 0<=slot<5 or arranged[slot] is not None:
                raise ValueError(f'Invalid BoardOrders at {index}')
            seen.add(uid);arranged[slot]=uid
        # In the checked v48 fixture BoardOrders is a complete nonempty team
        # configuration. A conflicting known roster is NOT silently repaired.
        if self.topology_known and seen != {x for x in self.team if x}:
            self.issues.append({"index":index,"reason":"order_batch_roster_conflict",
                                "before":self.team,"batch":arranged})
            self.topology_known=False
            return
        self.team=arranged;self.topology_known=True

    def apply_freezes(self,batch,index):
        if batch is None:return
        objects={}
        for entry in batch:
            uid=object_key(entry.get('ItemId'))
            if uid is None or uid in objects:raise ValueError(f'Invalid freeze batch at {index}')
            objects[uid]={"id":uid,"frozen":bool(entry.get('Freeze',False)),
                          "identityAnnotation":self.identity(uid),"evidenceActionIndex":index,
                          "screenPosition":None}
        self.shop=objects
        self.shop_universe_known=True

    def uncertain_shop(self,keep_frozen=True):
        self.shop={uid:item for uid,item in self.shop.items() if keep_frozen and item.get('frozen')}
        self.shop_universe_known=False

    def run(self):
        genesis=next(s for s in self.evidence['serializedBoards'] if s['actionIndex']==-1)['board']
        self.team=board_team(genesis)
        self.xp=board_xp(genesis)
        self.gold=int(genesis.get('Go',0))
        if genesis.get('FrRo',0) or genesis.get('FuRo',0):
            raise ValueError('Free-roll replay requires a separately verified ledger profile')
        for bank in ('MiSh','SpSh'):
            for item in genesis.get(bank,[]) or []:
                uid=object_key(item.get('Id'))
                self.shop[uid]={"id":uid,"frozen":bool(item.get('Fro',False)),"identityAnnotation":self.identity(uid),
                                "evidence":"genesis","screenPosition":None}
        self.shop_universe_known=True
        endpoint={s['turn']:s for s in self.evidence['serializedBoards'] if s['actionIndex']>=0}
        valid_end={e['turn']:e for e in self.evidence['endpoints'] if e['hashMatches']}
        records=[];checks=[]
        for a in self.evidence['actions']:
            kind,index,turn=a['type'],a['index'],a['turn']
            if kind not in (4,5,6,7,8,9,11):
                if kind not in (0,1,2,12):
                    self.topology_known=False
                    self.uncertain_shop(False)
                    self.issues.append({'index':index,'reason':'unhandled_native_action','type':kind})
                continue
            if turn!=self.turn:
                if self.turn>0:self.uncertain_shop()
                self.turn=turn
            self.apply_orders(a['reorderBatch'],index)
            self.apply_freezes(a['freezeBatch'],index)
            before=self.snapshot()
            source,target=a['sourceId'],a['targetId']
            semantic=a['action'];details=[]
            if kind==4:
                self.spent=0
                self.gold=10 if self.topology_known else None
                for uid in [x for x in self.team if x]:
                    identity=self.identity(uid)
                    if identity is None:
                        self.gold=None;break
                    if identity['name']=='Swan':
                        if self.xp.get(uid) is None:self.gold=None;break
                        if self.gold is not None:self.gold+=level(self.xp[uid])
                details.append('v48_turtle_turn_gold_10_plus_swan_levels; validated against spending/end gold')
            elif kind==5:
                self.spend(1,index)
                self.uncertain_shop()
            elif kind==6:
                self.spend(3,index)
                if source is None:raise ValueError(f'Purchase without native source at {index}')
                # Even if its placement is unknown, a fresh Turtle shop copy
                # starts at0XP. Its ID may later be consumed; this entry is not
                # an assertion that the ID survived onto the team.
                self.xp[source]=0
                kind_proof=self.purchase_kind_proof(a)
                details.append({'purchaseKindProof':kind_proof})
                self.shop.pop(source,None)
                slot=int((a.get('destination') or {}).get('x',0))
                if not 0<=slot<5:raise ValueError('Purchase coordinate out of bounds')
                if not self.topology_known:
                    details.append('purchase_with_unknown_team_topology')
                    if kind_proof['kind']!='separate':
                        identity=self.identity(source)
                        for uid in [x for x in self.team if x]:
                            other=self.identity(uid)
                            if not identity or not other or identity['name']==other['name']:
                                self.xp[uid]=None
                elif self.team[slot] is None:
                    self.team[slot]=source
                    self.xp[source]=0
                elif all(self.team) or kind_proof['kind']=='combine':
                    retained=self.team[slot]
                    old=self.identity(retained);new=self.identity(source)
                    if old and new and old['name']!=new['name']:
                        self.issues.append({"index":index,"reason":"full_board_buy_species_conflict","target":old,"source":new})
                        self.topology_known=False
                    else:
                        self.annotate_same_species(source,retained,index)
                        self.merge_xp(source,retained,shop=True)
                        semantic='buy_and_combine'
                        details.append({"retainedTargetId":retained,"consumedShopId":source})
                else:
                    # A merge and insertion/shift may both be possible. Do not
                    # choose one based on the next endpoint or policy preference.
                    self.topology_known=False
                    if kind_proof['kind']!='separate':self.xp[self.team[slot]]=None
                    details.append('occupied_destination_with_free_space_ambiguous_merge_or_insert')
                # Buy abilities can generate food; a level-up can generate pets.
                # We cannot assert an otherwise complete stock universe here.
                self.shop_universe_known=False
            elif kind==7:
                # Source/target IDs explicitly identify a combine even when
                # their positions are unknown (e.g. after a summoning Pill).
                self.merge_xp(source,target)
                if source in self.team and target in self.team and self.topology_known:
                    self.team[self.team.index(source)]=None
                else:
                    self.topology_known=False
                    details.append('combine_roster_not_fully_observed')
                self.shop_universe_known=False
            elif kind==9:
                identity=self.identity(source)
                experience=self.xp.get(source)
                if identity is None or experience is None:self.gold=None
                elif self.gold is not None:
                    self.gold+=level(experience)*(2 if identity['name']=='Pig' else 1)
                if source in self.team:self.team[self.team.index(source)]=None
                elif self.topology_known:
                    self.issues.append({"index":index,"reason":"sold_id_absent_from_team"})
                    self.topology_known=False
                self.shop_universe_known=False
            elif kind==8:
                self.shop.pop(source,None)
                food=a.get('consumedFood')
                if food is None:self.gold=None
                else:
                    self.spend(int(food['price']),index)
                    if food['name']=='Chocolate' and target:
                        xp=self.xp.get(target)
                        self.xp[target]=min(5,xp+1) if xp is not None else None
                if not food or food['name']=='Sleeping Pill':
                    # A faint may summon or resurrect; don't invent empty slots
                    # or look at the postbattle board to label them prematurely.
                    self.topology_known=False
                    details.append('food_or_faint_may_change_team_topology')
            elif kind==11 and turn in endpoint and turn in valid_end:
                anchor=endpoint[turn]
                native=board_team(anchor['board'])
                native_xp=board_xp(anchor['board'])
                checks.append({"turn":turn,"index":index,"predictionKnown":self.topology_known,
                               "predicted":list(self.team) if self.topology_known else None,"native":native,
                               "match":self.team==native if self.topology_known else None,
                               "experienceMatch":all(self.xp.get(uid)==value for uid,value in native_xp.items()) if all(self.xp.get(uid) is not None for uid in native_xp) else None,
                               "goldMatch":self.gold==int(anchor['board'].get('Go',0)) if self.gold is not None else None,
                               "spendingMatch":self.spent==int(anchor['board'].get('GoSp',0)),
                               "predictedGold":self.gold,"nativeGold":int(anchor['board'].get('Go',0)),
                               "predictedSpending":self.spent,"nativeSpending":int(anchor['board'].get('GoSp',0))})
                self.team=native;self.topology_known=True
                self.xp.update(native_xp)
                self.gold=int(anchor['board'].get('Go',0))
                details.append({"afterEndAnchor":anchor['evidence'],"hashReceiptMatched":True,
                                "notBeforeEndState":True})
            after=self.snapshot()
            if kind==11:after['phase']='battle_not_shop'
            records.append({"index":index,"turn":turn,"action":semantic,"sourceId":source,"targetId":target,
                            "sourceIdentityAnnotation":self.identity(source),"before":before,"after":after,
                            "details":details,"nativeActionEvidence":a['evidence'],"trainingApproved":False})
        return {"schema":SCHEMA,"participationId":self.evidence['participationId'],"name":self.evidence['name'],
                "actions":records,"endpointChecks":checks,"issues":self.issues,"identityConstraints":self.identity_constraints,
                "identityCatalog":self.catalog,"trainingApproved":False,
                "summary":{"records":len(records),"knownBeforeTopologies":sum(r['before']['teamTopologyKnown'] for r in records),
                           "buyCombinesIdentified":sum(r['action']=='buy_and_combine' for r in records),
                           "checkedEndpoints":sum(c['match'] is not None for c in checks),
                           "mismatchedEndpoints":sum(c['match'] is False for c in checks),
                           "checkedExperienceEndpoints":sum(c['experienceMatch'] is not None for c in checks),
                           "mismatchedExperienceEndpoints":sum(c['experienceMatch'] is False for c in checks),
                           "checkedGoldEndpoints":sum(c['goldMatch'] is not None for c in checks),
                           "mismatchedGoldEndpoints":sum(c['goldMatch'] is False for c in checks),
                           "mismatchedSpendingEndpoints":sum(c['spendingMatch'] is False for c in checks),
                           "completePolicyObservations":0}}
