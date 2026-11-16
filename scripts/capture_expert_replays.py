#!/usr/bin/env python3
"""Consume a bounded public replay queue through the existing local browser."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sapai_gym1.replay.visual_capture import run_queue

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--max-jobs",type=int,default=4)
    p.add_argument("--participation-id",action='append')
    p.add_argument("--reload-client",action='store_true')
    a=p.parse_args()
    if not 1<=a.max_jobs<=80:p.error("max-jobs must be 1..80")
    run_queue(a.root,a.corpus,a.output,a.profile,a.max_jobs,a.participation_id,a.reload_client)
