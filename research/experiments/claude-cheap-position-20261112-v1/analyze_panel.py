"""Cost vs share of the oracle's positioning gain captured, per setting (diag rows written by --diag-oracle).
Cost = wall clock of one END positioning decision, excluding the privileged diagnostic scoring.
Usage: python3 analyze_panel.py ref half small one prev"""
import json, statistics as st, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent / 'out'
PARTS = ('paths', 'sim', 'screen', 'apply', 'exact')
rows = []
for name in sys.argv[1:]:
    f = OUT / f'{name}.jsonl'
    if not f.exists(): print(name, 'missing'); continue
    games = [json.loads(l) for l in f.read_text().splitlines()]
    d = [x for g in games for x in (g.get('diag') or [])]
    if not d: print(name, 'no diag rows'); continue
    m = lambda k: st.mean(x[k] for x in d)
    cap = (m('realChosen') - m('realCurrent')) / max(1e-9, m('realBest') - m('realCurrent'))
    gain = [x['realChosen'] - x['realCurrent'] for x in d]
    tot = sorted(sum(x['timing'].get(p, 0.) for p in PARTS) for x in d)
    q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
    part = {p: round(st.mean(x['timing'].get(p, 0.) for x in d), 1) for p in PARTS}
    battles = st.mean(x['timing'].get('nScreenBattles', 0) + x['timing'].get('nExactBattles', 0) for x in d)
    under50 = sum(t <= 50 for t in tot) / len(tot)
    r = dict(setting=name, games=len(games), decisions=len(d), captured=round(cap, 2), gain=round(st.mean(gain), 3),
             se=round(st.stdev(gain) / len(gain) ** .5, 3), ms_median=round(q(tot, .5)), ms_p90=round(q(tot, .9)),
             ms_mean=round(st.mean(tot)), under50=round(under50, 2), battles=round(battles), parts_ms=part,
             score=round(st.mean(g['score'] for g in games), 3))
    rows.append(r)
    print(f"{name:6s} games {r['games']:2d} dec {r['decisions']:3d} | captured {r['captured']:.2f} gain {r['gain']:+.3f} ±{r['se']:.3f} | "
          f"ms median {r['ms_median']:5d} p90 {r['ms_p90']:5d} mean {r['ms_mean']:5d} | <=50ms {r['under50']:.2f} | battles {r['battles']:4d} | {part}")
(OUT / 'panel-summary.json').write_text(json.dumps(rows, indent=1))
