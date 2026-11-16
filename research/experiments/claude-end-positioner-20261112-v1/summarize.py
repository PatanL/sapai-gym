"""Summarise a (possibly split) paired evaluation: python3 summarize.py out/eval-v3-*.jsonl"""
import glob, json, sys, statistics
import random

files = [f for pat in sys.argv[1:] for f in glob.glob(pat) if not f.endswith('-turns.jsonl')]
rows = [json.loads(l) for f in files for l in open(f)]
by = {}
for r in rows: by.setdefault(r['seed'], {})[r['plannerSide']] = r
pairs = [v for v in by.values() if len(v) == 2]
scores = [(v[0]['score'] + v[1]['score']) / 2 for v in pairs]
rng = random.Random(0)
boot = sorted(sum(rng.choice(scores) for _ in scores) / len(scores) for _ in range(5000)) if scores else [0.]
pw = sum(r['plannerWins'] for r in rows); rw = sum(r['rawWins'] for r in rows)
print(json.dumps(dict(files=len(files), games=len(rows), pairs=len(pairs), meanMatchScore=round(sum(scores) / len(scores), 4) if scores else None,
                      paired95=[round(boot[int(.025 * len(boot))], 4), round(boot[int(.975 * len(boot)) - 1], 4)],
                      pairWon=scores.count(1.), pairSplit=scores.count(.5), pairLost=scores.count(0.),
                      gameScore=round(sum(r['score'] for r in rows) / max(1, len(rows)), 4), battleShare=round(pw / max(1, pw + rw), 4),
                      meanSeconds=round(statistics.mean(r['seconds'] for r in rows), 1) if rows else None), indent=1))
turns = [json.loads(l) for f in files for l in open(f.replace('.jsonl', '-turns.jsonl'))] if all(
    __import__('os').path.exists(f.replace('.jsonl', '-turns.jsonl')) for f in files) else []
if turns:
    ov = [t for t in turns if t['override']]
    d = [t['realChosen'] - t['realGreedy'] for t in ov]
    print(json.dumps(dict(turns=len(turns), overrides=len(ov), overrideRealGain=round(statistics.mean(d), 4) if d else None,
                          overrideRealGainSE=round(statistics.stdev(d) / len(d) ** .5, 4) if len(d) > 1 else None,
                          meanRealActual=round(statistics.mean(t['realActual'] for t in turns), 4))))
