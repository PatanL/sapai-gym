#!/usr/bin/env python3
"""Bounded CPU annotation queue for completed captures; preserves failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sapai_gym1.replay.archive import atomic_json, lock, sha_file, utc
from sapai_gym1.replay.visual_parse import parse_capture


def service_state(name):
    result=subprocess.run(['systemctl','--user','show',name,'-p','ActiveState','-p','InvocationID'],check=True,capture_output=True,text=True,timeout=10)
    return dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--visual',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capture-service',required=True)
    p.add_argument('--capture-invocation',required=True)
    p.add_argument('--restore-browser-invocation')
    p.add_argument('--seconds',type=int,default=4500)
    a=p.parse_args()
    if not a.capture_service.startswith('sap-expert-visual-') or '/' in a.capture_service:
        p.error('Unexpected capture service')
    a.output.mkdir(parents=True,exist_ok=True)
    completed={};deadline=time.monotonic()+a.seconds
    with lock(a.output/'worker.lock'):
        try:
            while time.monotonic()<deadline:
                state=service_state(a.capture_service)
                if state.get('InvocationID') not in ('',a.capture_invocation):
                    raise RuntimeError('Capture service invocation changed; refusing to follow a different run')
                live=state.get('ActiveState') in ('active','activating','deactivating','reloading')
                for result_path in sorted(a.visual.glob('*/RESULT.json')):
                    result=json.loads(result_path.read_text());pid=result['job']['participationId']
                    result_sha=sha_file(result_path)
                    if pid in completed and completed[pid].get('resultSHA256')==result_sha:continue
                    attempt=a.visual/result['attempt']
                    if not result.get('coverage',{}).get('turnCoverageComplete'):
                        completed[pid]={'status':'capture_incomplete','evidence':str(result_path),'resultSHA256':result_sha};continue
                    destination=a.output/pid
                    try:
                        if (destination/'CANDIDATES.json').exists():
                            report=json.loads((destination/'CANDIDATES.json').read_text())
                            if report['sourceIndexSHA256']!=sha_file(attempt/'INDEX.json'):
                                raise ValueError('Completed annotation belongs to a different capture')
                        else:
                            report=parse_capture(a.root,attempt,destination)
                        completed[pid]={'status':'visual_candidates_not_validated','stats':report['stats'],
                                        'report':str(destination/'CANDIDATES.json')}
                    except Exception as exc:
                        completed[pid]={'status':'parse_failed','error':f'{type(exc).__name__}: {exc}','attempt':str(attempt)}
                    completed[pid]['resultSHA256']=result_sha
                    atomic_json(a.output/'STATUS.json',{'phase':'running','results':completed,'trainingApproved':False,'updatedAt':utc()})
                if not live:break
                time.sleep(10)
            atomic_json(a.output/'STATUS.json',{'phase':'finished_or_time_bound','results':completed,'trainingApproved':False,'updatedAt':utc()})
        finally:
            # This run temporarily raised the software-rendering browser from
            # 1 to 2 cores. Never alter a replacement browser invocation.
            if a.restore_browser_invocation:
                browser=service_state('sap-live-browser.service')
                if browser.get('InvocationID')==a.restore_browser_invocation:
                    subprocess.run(['systemctl','--user','set-property','--runtime','sap-live-browser.service','CPUQuota=100%'],check=True,timeout=10)


if __name__=='__main__':main()
