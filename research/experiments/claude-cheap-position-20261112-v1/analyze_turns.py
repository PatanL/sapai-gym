import json, statistics, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
ov=[r for r in rows if r['override']]
def m(xs): return sum(xs)/len(xs) if xs else float('nan')
print('turns',len(rows),'overrides',len(ov))
for name,rs in (('ALL',rows),('OVR',ov)):
    print('%s pred greedy %.3f chosen %.3f | real greedy %.3f chosen %.3f actual %.3f'%(name,m([r['predGreedy'] for r in rs]),m([r['predChosen'] for r in rs]),m([r['realGreedy'] for r in rs]),m([r['realChosen'] for r in rs]),m([r['realActual'] for r in rs])))
d=[r['realChosen']-r['realGreedy'] for r in ov]
if len(d)>1: print('override real gain mean %.4f sd %.3f se %.4f'%(m(d),statistics.stdev(d),statistics.stdev(d)/len(d)**.5))
pd=[r['predChosen']-r['predGreedy'] for r in ov]
if len(pd)>1: print('override predicted gain mean %.4f; corr(pred gain, real gain) %.3f'%(m(pd), statistics.correlation(pd,d) if len(d)>2 else float('nan')))
