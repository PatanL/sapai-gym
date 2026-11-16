from plan_common import install
install(cuda=False)
from sapai import data
import json, pickle, collections
D = data.data if hasattr(data, 'data') else data
D = D if isinstance(D, dict) else vars(D)
print('keys', list(D.keys())[:12])
pets = sorted(D['pets'].keys())
statuses = sorted(D['statuses'].keys()) if 'statuses' in D else []
lib = pickle.load(open('out/library.pkl', 'rb'))
seen_status = collections.Counter(); seen_names = collections.Counter()
for t, rows in lib.items():
    for r in rows:
        for sl in r['board']['team']:
            p = sl.get('pet') or {}
            seen_names[p.get('name')] += 1; seen_status[p.get('status')] += 1
print(json.dumps(dict(npets=len(pets), nstatus=len(statuses), statuses=statuses[:40], seenStatus=seen_status.most_common(30), seenNames=len(seen_names))))
json.dump(dict(pets=pets, statuses=statuses), open('out/vocab.json', 'w'))
