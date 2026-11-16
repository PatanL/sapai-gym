#!/usr/bin/env python3
"""Run deterministic native ID constraints without changing archive evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sapai_gym1.replay.archive import atomic_json,immutable_json,sha_file
from sapai_gym1.replay.native_tracker import NativeTopologyTracker

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    status=json.loads((a.corpus/'STATUS.json').read_text());results=[]
    for match in status['matches']:
        path=a.corpus/match['directory']/'EVIDENCE.json'
        if sha_file(path)!=match['evidenceSHA256']:raise ValueError('Evidence hash changed')
        evidence=json.loads(path.read_text())
        for perspective in evidence['perspectives']:
            result=NativeTopologyTracker(perspective).run()
            result.update(matchId=match['matchId'],sourceEvidenceSHA256=match['evidenceSHA256'],split=match['split'])
            dest=a.output/match['matchId']/(perspective['participationId']+'.json')
            immutable_json(dest,result)
            results.append({'matchId':match['matchId'],'participationId':perspective['participationId'],
                            'file':str(dest.relative_to(a.output)), 'sha256':sha_file(dest),
                            'summary':result['summary'],'issues':len(result['issues'])})
    report={'schema':'sap-native-id-tracker-batch-v1','perspectives':len(results),'results':results,
            'summary':{key:sum(r['summary'][key] for r in results) for key in results[0]['summary']},
            'sourceSHA256':sha_file(Path(sys.modules[NativeTopologyTracker.__module__].__file__)),
            'trainingApproved':False}
    immutable_json(a.output/'REPORT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':main()
