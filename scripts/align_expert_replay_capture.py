#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sapai_gym1.replay.archive import immutable_json,sha_file
from sapai_gym1.replay.align_constraints import align
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Native/visual candidate alignment; not training labels')
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--visual',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=align(json.loads(a.trace.read_text()),json.loads(a.visual.read_text()))
    result.update(traceSHA256=sha_file(a.trace),visualSHA256=sha_file(a.visual),sourceSHA256=sha_file(Path(sys.modules[align.__module__].__file__)))
    immutable_json(a.output,result);print(json.dumps(result['summary'],indent=2))
