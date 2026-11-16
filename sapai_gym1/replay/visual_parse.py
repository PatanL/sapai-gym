"""CPU visual candidates and temporal scene grouping, NOT certified labels.

Every raw capture frame is retained. Consecutive frames with identical numeric
observations are grouped; multiple identity keyframes are retained to avoid
silently assuming identical stats mean identical pets or unchanged freezes.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .archive import atomic_json, immutable_json, sha_file, utc


class VisualCandidates:
    def __init__(self, root):
        from sapai_gym1.deployment import stat_ocr
        from sapai_gym1.deployment.pet_classifier import OnnxPetClassifier
        from sapai_gym1.deployment.sprite_catalog import SpriteCatalog
        from sapai_gym1.deployment.sprite_matcher import SpriteMatcher
        self.root=Path(root)
        self.ocr=stat_ocr
        self.digits=stat_ocr.DigitTemplateRecognizer(self.root/'runs/live-vision/stat-digits-v1.npz')
        self.pets=OnnxPetClassifier(self.root/'runs/live-vision/pet-mobilenet-v1.onnx')
        self.foods=SpriteMatcher(SpriteCatalog.from_replay_bot(self.root/'sap-replay-bot'))
        self.food_cache={}
        cv2.setNumThreads(1)

    def numeric(self, im):
        values=[]
        for bank,count in [('team',5),('shopPet',7)]:
            for slot in range(count):
                x=384+144*slot
                item={'bank':bank,'screenSlot':slot,'centerX':x,'attack':None,'health':None,'level':None,'experience':None,
                      'occupancy':'unknown_or_empty','perk':None,'perkStatus':'unresolved'}
                try:
                    a,h=self.digits.read_pair(im,x/1920,shop=bank=='shopPet')
                    item.update(attack=a.value,health=h.value,statConfidence=min(a.confidence,h.confidence),occupancy='stats_detected')
                except ValueError:
                    pass
                if bank=='team' and item['attack'] is not None:
                    try:
                        level=self.ocr.read_level_experience(im,x/1920)
                        item.update(level=level.level,experience=level.experience,levelConfidence=level.confidence)
                    except ValueError:
                        pass
                values.append(item)
        return values

    def identities(self, im, items):
        crops=[];present=[]
        for item in items:
            if item['occupancy']!='stats_detected':continue
            x=item['centerX'];shop=item['bank']=='shopPet'
            y1,y2=(625,765) if shop else (350,490)
            crops.append(im[y1:y2,x-90:x+90]);present.append(item)
        if crops:
            for item,prediction in zip(present,self.pets.predict(crops)):
                item['identityCandidate']=prediction.to_dict()
                item['species']=prediction.item_id if prediction.accepted else None
                item['replayDomainCalibrated']=False
        foods=[]
        for x in (1392,1536):
            badge=im[570:635,x-82:x-14]
            if float((badge.mean(axis=2)<45).mean()) <= .05:continue
            crop=im[570:790,x-115:x+115]
            key=hashlib.sha256(crop.tobytes()).hexdigest()
            if key not in self.food_cache:
                self.food_cache[key]=self.foods.match_food(crop).to_dict()
            prediction=self.food_cache[key]
            foods.append({'centerX':x,'identityCandidate':prediction,'species':prediction['best']['visual_id'] if prediction['accepted'] else None,
                          'frozen':None,'frozenStatus':'requires_native_batch_alignment_or_visual_review',
                          'price':None,'replayDomainCalibrated':False})
        return foods


def fingerprint(row,items):
    obs=row['observation']
    return (obs['turn']['value'],obs['gold']['value'],tuple((x['attack'],x['health'],x['level'],x['experience']) for x in items))


def parse_capture(root, directory, output, max_keyframes=400):
    root,directory,output=Path(root),Path(directory),Path(output)
    if output.exists():
        raise ValueError('Choose a new visual-parse output; old candidates are immutable')
    output.mkdir(parents=True)
    index_path=directory/'INDEX.json';index=json.loads(index_path.read_text())
    if index['endReason']=='running':raise ValueError('Capture still running')
    source_sha=sha_file(index_path)
    parser=VisualCandidates(root);started=time.monotonic()
    scenes=[];last_key=None;current=None;review=[]
    for i,row in enumerate(index['frames']):
        obs=row['observation']
        if not obs.get('shop') or not obs.get('turn') or not obs.get('gold'):
            last_key=None;current=None;continue
        path=directory/row['frame']
        if sha_file(path)!=row['sha256']:raise ValueError(f'Frame hash changed: {path}')
        im=cv2.imread(str(path));items=parser.numeric(im)
        key=fingerprint(row,items)
        if key!=last_key:
            current={'scene':len(scenes),'turn':obs['turn']['value'],'gold':obs['gold']['value'],
                     'firstFrame':row['frame'],'lastFrame':row['frame'],'frameCount':0,'candidates':items,
                     'frames':[],'seconds':row['seconds'],'observation':obs}
            scenes.append(current);last_key=key
        current['frameCount']+=1;current['lastFrame']=row['frame'];current['frames'].append(row['frame'])
        if i%150==0:
            print(json.dumps({'parsedNumericFrames':i,'scenes':len(scenes)}),flush=True)
    # All scenes remain in the output, even unstable/transient ones. Spend the
    # bounded classifier budget on multi-frame states, uniformly across turns.
    eligible=[s for s in scenes if s['frameCount']>=2]
    if len(eligible)>max_keyframes:
        positions=np.linspace(0,len(eligible)-1,max_keyframes,dtype=int)
        selected=[eligible[int(i)] for i in positions]
    else:selected=eligible
    for n,scene in enumerate(selected):
        path=directory/scene['lastFrame'];im=cv2.imread(str(path))
        scene['foods']=parser.identities(im,scene['candidates'])
        scene['identityFrame']=scene['lastFrame']
        scene['freezePerkPriceComplete']=False
        for item in scene['candidates']:
            if item['occupancy']=='stats_detected' and not item.get('species'):
                review.append({'scene':scene['scene'],'frame':scene['identityFrame'],'bank':item['bank'],'slot':item['screenSlot'],'reason':'uncertain_pet_identity'})
        if n%10==0:
            atomic_json(output/'PROGRESS.json',{'phase':'identity_candidates','completed':n,'total':len(selected),'updatedAt':utc()})
            print(json.dumps({'identityScenes':n,'total':len(selected)}),flush=True)
    report={'schema':'sap-replay-visual-candidates-v1','sourceIndex':str(index_path),'sourceIndexSHA256':source_sha,
            'job':index['job'],'captureCoverage':index['coverage'],'scenes':scenes,'reviewItems':review,
            'stats':{'rawFrames':len(index['frames']),'numericScenes':len(scenes),'identityScenes':len(selected),
                     'seconds':time.monotonic()-started,'sceneCountsByTurn':dict(Counter(s['turn'] for s in scenes))},
            'limitations':['OCR and classifier scores are uncalibrated on replay cosmetics',
                'unchanged numeric fingerprint does not prove unchanged species, perks or freezes; all raw frames retained',
                'profile covers seven pet anchors and two food anchors; crowded alternate layouts require review',
                'unknown_or_empty is not an empty-slot label','no action/frame one-to-one alignment or full-state certification yet'],
            'trainingApproved':False,'modelSHA256':sha_file(root/'runs/live-vision/pet-mobilenet-v1.onnx'),
            'digitsSHA256':sha_file(root/'runs/live-vision/stat-digits-v1.npz')}
    immutable_json(output/'CANDIDATES.json',report)
    atomic_json(output/'PROGRESS.json',{'phase':'complete_candidates_not_training_labels','stats':report['stats'],'updatedAt':utc()})
    return report
