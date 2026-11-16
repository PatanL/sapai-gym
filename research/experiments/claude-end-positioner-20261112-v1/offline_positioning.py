"""Offline: how much does reordering the policy's own boards gain, measured on fresh opponents/seeds?"""
import itertools, json, multiprocessing as mp, pickle, random, sys, copy
from plan_common import install, OUT
install(cuda=False)
LIB = pickle.load(open(OUT / 'library.pkl', 'rb'))
BOARDS = {t: [r['board'] for r in rows] for t, rows in LIB.items()}

def permute(state, order):
    s = copy.deepcopy(state); slots = s['team']
    pets = [slots[i] for i in order]; empties = [x for i, x in enumerate(slots) if i not in order]
    s['team'] = pets + empties
    return s

def equity(args):
    state, turn, opp_idx, seeds = args
    from sapai import Battle, Team
    tot = 0.
    for j, sd in zip(opp_idx, seeds):
        r = Battle(Team.from_state(state), Team.from_state(BOARDS[turn][j]), seed=sd, fast_training=True).battle()
        tot += 1. if r == 0 else (.5 if r == 2 else 0.)
    return tot / len(seeds)

def occupied(state):
    return [i for i, sl in enumerate(state['team']) if sl.get('pet') and sl['pet'].get('name') not in (None, 'pet-none')]

if __name__ == '__main__':
    rng = random.Random(7); pool = mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
    out = []
    for n in range(300):
        turn = rng.randint(3, 12); state = rng.choice(BOARDS[turn]); occ = occupied(state)
        if len(occ) < 3: continue
        perms = list(itertools.permutations(occ))
        screen_idx = [rng.randrange(len(BOARDS[turn])) for _ in range(48)]; screen_seeds = [rng.getrandbits(31) for _ in range(48)]
        cands = [permute(state, p) for p in perms]
        sc = pool.map(equity, [(c, turn, screen_idx, screen_seeds) for c in cands])
        orig = sc[perms.index(tuple(occ))]; best = max(range(len(perms)), key=lambda i: sc[i])
        fresh_idx = [rng.randrange(len(BOARDS[turn])) for _ in range(96)]; fresh_seeds = [rng.getrandbits(31) for _ in range(96)]
        fo, fb = pool.map(equity, [(state, turn, fresh_idx, fresh_seeds), (cands[best], turn, fresh_idx, fresh_seeds)])
        out.append(dict(turn=turn, pets=len(occ), screenOrig=orig, screenBest=sc[best], freshOrig=fo, freshBest=fb))
        if len(out) % 25 == 0:
            m = lambda k: sum(r[k] for r in out) / len(out)
            print(len(out), 'screen orig %.3f best %.3f | fresh orig %.3f best %.3f  gain %.3f' % (m('screenOrig'), m('screenBest'), m('freshOrig'), m('freshBest'), m('freshBest') - m('freshOrig')), flush=True)
    json.dump(out, open(OUT / 'offline_positioning.json', 'w'))
