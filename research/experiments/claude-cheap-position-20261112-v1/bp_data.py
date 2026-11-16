"""Battle-predictor training data: (board A, board B) -> outcome counts over simulated battles.

A: library boards and planner-like variants (reordered, a pet dropped, a pet swapped in from another board of
the same turn, small stat changes). B: library boards from the same or a nearby turn. Labels are win/draw/loss
counts from independent battle seeds (A's perspective). A held-out calibration set uses 64 battles per pair.
"""
import copy, json, multiprocessing as mp, pickle, random, sys, time
import numpy as np
from plan_common import install, OUT

PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LIB = None
EMPTY_SLOT = None


def occupied(state):
    return [i for i, sl in enumerate(state['team']) if (sl.get('pet') or {}).get('name') not in (None, 'pet-none')]


def mutate(state, turn, rng):
    s = copy.deepcopy(state); occ = occupied(s)
    r = rng.random()
    if r < 0.30 and len(occ) >= 2:   # reorder pets
        pets = [s['team'][i] for i in occ]; rng.shuffle(pets)
        for i, p in zip(occ, pets): s['team'][i] = p
    elif r < 0.45 and len(occ) >= 2:  # drop a pet: remove its slot and append an empty slot at the back
        i = rng.choice(occ); del s['team'][i]; s['team'].append(copy.deepcopy(EMPTY_SLOT))
    elif r < 0.70:                    # swap in a pet from another board of this turn
        other = rng.choice(LIB[turn]); oo = occupied(other)
        if oo and occ:
            s['team'][rng.choice(occ)] = copy.deepcopy(other['team'][rng.choice(oo)])
    elif r < 0.85 and occ:            # stat nudges
        for i in occ:
            p = s['team'][i]['pet']; p['attack'] = max(1, p['attack'] + rng.randint(-2, 3)); p['health'] = max(1, p['health'] + rng.randint(-2, 3))
    return s


def _work(args):
    seed, n, battles = args
    from planner import library_team, safe_battle
    from sapai import Battle
    from bp_common import encode
    rng = random.Random(seed); turns = [t for t in LIB if len(LIB[t]) >= 20]
    A = ([], [], []); B = ([], [], []); Y = []
    for _ in range(n):
        t = rng.choice(turns)
        a = rng.choice(LIB[t]); a = mutate(a, t, rng) if rng.random() < 0.6 else a
        tb = t if rng.random() < 0.8 else min(turns, key=lambda x: abs(x - (t + rng.choice((-2, -1, 1, 2)))))
        b = rng.choice(LIB[tb])
        counts = [0, 0, 0]
        for k in range(battles):
            s = rng.getrandbits(31)
            r = safe_battle(library_team(a, s), library_team(b, s + 1), s)
            if r is None: continue  # engine loop: skip this battle
            counts[0 if r == 0 else (1 if r == 2 else 2)] += 1
        if sum(counts) == 0: continue
        for dst, src in ((A, encode(a)), (B, encode(b))):
            for d, x in zip(dst, src): d.append(x)
        Y.append(counts)
    return [np.stack(x) for x in A], [np.stack(x) for x in B], np.asarray(Y, np.int16)


def main():
    global LIB, EMPTY_SLOT
    install(cuda=False)
    from planner import strip_rng
    raw = pickle.load(open(OUT / 'library.pkl', 'rb'))
    LIB = {t: [strip_rng(r['board']) for r in rows] for t, rows in raw.items()}; del raw
    EMPTY_SLOT = next(sl for rows in LIB.values() for b in rows for sl in b['team'] if (sl.get('pet') or {}).get('name') in (None, 'pet-none'))
    pool = mp.get_context('fork').Pool(WORKERS)
    began = time.time()
    for name, total, battles, chunk in (('bp-calib', 2000, 64, 125), ('bp-train', PAIRS, 1, 2000)):
        parts = pool.map(_work, [(hash((name, i)) & 0xffffffff, min(chunk, total - i * chunk), battles) for i in range((total + chunk - 1) // chunk)])
        A = [np.concatenate([p[0][j] for p in parts]) for j in range(3)]
        B = [np.concatenate([p[1][j] for p in parts]) for j in range(3)]
        Y = np.concatenate([p[2] for p in parts])
        np.savez_compressed(OUT / f'{name}.npz', a_ids=A[0], a_st=A[1], a_num=A[2], b_ids=B[0], b_st=B[1], b_num=B[2], y=Y)
        print(json.dumps(dict(set=name, pairs=len(Y), battles=battles, seconds=round(time.time() - began, 1),
                              aWinRate=float(Y[:, 0].sum() / Y.sum()), drawRate=float(Y[:, 1].sum() / Y.sum()))), flush=True)


if __name__ == '__main__':
    main()
