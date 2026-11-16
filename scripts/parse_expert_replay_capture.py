#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sapai_gym1.replay.visual_parse import parse_capture
if __name__=='__main__':
    p=argparse.ArgumentParser(description='CPU replay visual candidates; no training export')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-keyframes',type=int,default=400)
    a=p.parse_args()
    if not 1<=a.max_keyframes<=2000:p.error('max-keyframes outside 1..2000')
    parse_capture(a.root,a.capture,a.output,a.max_keyframes)
