import json, statistics, sys
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    d = [x for r in rows for x in (r.get('diag') or [])]
    if not d: print(f, 'no diag rows'); continue
    m = lambda k: statistics.mean(x[k] for x in d)
    gain = [x['realChosen'] - x['realCurrent'] for x in d]
    print(f'{f}: games {len(rows)} decisions {len(d)} | real current {m("realCurrent"):.3f} chosen {m("realChosen"):.3f} '
          f'best {m("realBest"):.3f} | captured {(m("realChosen") - m("realCurrent")) / max(1e-9, m("realBest") - m("realCurrent")):.2f} '
          f'| real gain/decision {statistics.mean(gain):.3f} (se {statistics.stdev(gain) / len(gain) ** .5:.3f}) | sim chosen {m("simChosen"):.3f}')
