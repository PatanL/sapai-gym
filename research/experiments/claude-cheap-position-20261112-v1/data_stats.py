import glob, torch, collections
files = glob.glob('out/distill-v1/game-*.pt'); n = f = 0; types = collections.Counter(); turns = collections.Counter()
for p in files:
    for r in torch.load(p, weights_only=False):
        n += 1
        if r['forced']:
            f += 1; types[r['action']['primitive'][0]] += 1; turns[r['turn']] += 1
print(f'files {len(files)} decisions {n} forced {f} ({f / max(1, n):.3f}) forced action types {dict(types)} forced by turn {dict(sorted(turns.items()))}')
