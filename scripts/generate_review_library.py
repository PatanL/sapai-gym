#!/usr/bin/env python3
"""Bounded sequential CPU replay library, separate from training/evaluation."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--latest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--seed', type=int, default=90612000)
    args=parser.parse_args()
    if not 1 <= args.games <= 20 or not args.prefix.replace('-', '').isalnum():
        raise ValueError('Use a bounded library and safe unique filename prefix')
    root=args.output_directory;root.mkdir(parents=True,exist_ok=True)
    manifest=root/f'{args.prefix}.json'
    if manifest.exists():raise ValueError('Never overwrite a previous library')
    data={'schema':'cpu-review-library-v1','latest':str(args.latest),'baseline':str(args.baseline),
          'checkpointHashes':{'latest':sha(args.latest),'baseline':sha(args.baseline)},
          'startedUtc':time.time(),'inference':'deterministic CPU, one worker',
          'selection':'10 consecutive seeds, alternating seats; no win filtering',
          'experimentalLatest':True,'games':[],'complete':False}
    for game in range(args.games):
        data['games'].append({'game':game+1,'seed':args.seed+game,'latestSeat':game%2,
                             'views':{},'status':'pending'})
    def save():
        manifest.write_text(json.dumps(data,indent=2)+'\n')
        rows=[]
        for game in data['games']:
            links=' · '.join(f'<a href="{html.escape(v["html"])}">P{side}{" — latest" if int(side)==game["latestSeat"] else " — baseline"}</a>' for side,v in game['views'].items())
            result=next(iter(game['views'].values()),{}).get('result','')
            rows.append(f'<tr><td>Game {game["game"]}</td><td>{game["seed"]}</td><td>{links or "Queued"}</td><td>{html.escape(result)}</td><td>{html.escape(game["status"])}</td></tr>')
        page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAP livestream review library</title><style>body{font:18px system-ui;background:#fffbed;color:#222;max-width:1200px;margin:40px auto;padding:20px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:16px;border-bottom:1px solid #ddd}a{color:#1657a0}p{line-height:1.6}</style>
<h1>SAP — 10 games to review</h1><p>Latest experimental update 9200 vs v45 update 6250. Consecutive seeds, alternating seats, no cherry-picking. Both views replay the same seeded match.</p>
<p>Choose a game, then a ⚔ battle in its sidebar. Use <b>Play battle</b>, speed, the event slider, or <b>Save this moment</b> to flag a suspected bug. Playback shows recorded simulator trigger/phase snapshots, not a separate battle implementation.</p>
<p>Games appear here as the single CPU worker finishes. Refresh for new links. No GPU is used. This is a review sample, not a strength evaluation.</p>
<table><thead><tr><th>Game</th><th>Seed</th><th>Perspectives</th><th>Result</th><th>Status</th></tr></thead><tbody>'''+''.join(rows)+'</tbody></table>'
        temporary=root/f'{args.prefix}.html.tmp';temporary.write_text(page);temporary.replace(root/f'{args.prefix}.html')
    save()
    try:
        for game in data['games']:
            checkpoints=[args.latest,args.baseline] if game['latestSeat']==0 else [args.baseline,args.latest]
            for side in (0,1):
                game['status']=f'generating P{side}';save()
                stem=f'{args.prefix}-game{game["game"]:02d}-p{side}'
                output=root/f'{stem}.html';trace=root/f'{stem}.json'
                subprocess.run([sys.executable,str(Path(__file__).with_name('visualize_entity_game.py')),
                    str(checkpoints[0]),'--opponent-checkpoint',str(checkpoints[1]),
                    '--seed',str(game['seed']),'--perspective',f'p{side}','--threads','1',
                    '--output',str(output),'--trace-json',str(trace)],check=True)
                payload=json.loads(trace.read_text())
                game['views'][str(side)]={'html':output.name,'trace':trace.name,'result':payload['summary']['resultLabel']}
                if side==1:
                    other=json.loads((root/game['views']['0']['trace']).read_text())
                    if other['summary']['result']!=payload['summary']['result']:
                        raise RuntimeError('Perspective replays diverged')
                    left=[f['battle']['playback'] for f in other['frames'] if f['phase']=='battle']
                    right=[f['battle']['playback'] for f in payload['frames'] if f['phase']=='battle']
                    if left!=right:raise RuntimeError('Battle playback differs across perspectives')
                save()
            game['status']='ready';save()
        data['complete']=True;data['finishedUtc']=time.time();save()
    except Exception as error:
        data['failure']=str(error);save();raise


if __name__=='__main__':main()
