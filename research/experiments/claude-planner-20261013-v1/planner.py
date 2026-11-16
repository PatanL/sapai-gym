"""Turn planner: choose this shop turn's whole plan by its next-battle equity against the opponent's board
distribution, using the unchanged RL policy to propose plans.

Public information only:
- Plans are simulated on clones whose shop tape is replaced (`enable_paired_tape` with a fresh seed), so no
  future roll, draw or shop-ability outcome of the real game is visible. The clone's opponent is never read.
- Battles are against boards the same checkpoint builds at this turn in separate self-play games
  (build_library.py), never the opponent's real current board.

The policy keeps every roll decision: if its greedy action is ROLL, the planner rolls and re-plans. Plans
themselves contain no rolls. Execution re-plans whenever the real state differs from the simulated plan.
"""
import dataclasses, json, multiprocessing as mp, pickle, random
from pathlib import Path

_LIB = None  # turn -> list of board states (inherited by forked workers)


def pet_names(state):
    """Multiset of pet names on a board state (public composition)."""
    from collections import Counter
    if not state: return Counter()
    return Counter(slot['pet']['name'] for slot in state.get('team', []) if slot.get('pet') and slot['pet'].get('name') not in (None, 'pet-none'))


def strip_rng(value):
    """Board/gold fingerprint without per-pet RNG states (they differ harmlessly between clone and reality)."""
    if isinstance(value, dict): return {k: strip_rng(v) for k, v in value.items() if k != 'seed_state'}
    if isinstance(value, list): return [strip_rng(v) for v in value]
    return value


class BattleTimeout(Exception):
    pass


def safe_battle(team_a, team_b, seed, limit=2):
    """sapai battle result (0 A wins, 1 B wins, 2 draw), or None if it runs past `limit` seconds (rare engine
    loops). Uses SIGALRM, so call only from a single-threaded worker process."""
    import signal
    from sapai import Battle
    def _raise(*_): raise BattleTimeout()
    old = signal.signal(signal.SIGALRM, _raise); signal.alarm(limit)
    try:
        return Battle(team_a, team_b, seed=seed, fast_training=True).battle()
    except BattleTimeout:
        return None
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)


def library_team(state, seed):
    """A library board as a battle team: its pets get fresh RNGs (stored boards carry no RNG state)."""
    import numpy as np
    from sapai import Team
    team = Team.from_state(state)
    for k, slot in enumerate(team):
        if not slot.empty:
            slot.pet.rs = np.random.RandomState((seed * 31 + k) % (2**32))
    return team


def _score_task(args):
    """Mean battle score (win 1, draw .5, loss 0) of one board against fixed opponents and seeds."""
    state, turn, idx, seeds = args
    from sapai import Battle, Team
    boards = _LIB[turn]; total = 0.; n = 0
    for j, s in zip(idx, seeds):
        r = safe_battle(Team.from_state(state), library_team(boards[j], s), s)
        if r is None: continue
        total += 1. if r == 0 else (.5 if r == 2 else 0.); n += 1
    return total / max(1, n)


def _versus_task(args):
    """Diagnostics only: mean battle score of one board against one specific opponent board."""
    state, opp, seeds = args
    from sapai import Battle, Team
    total = 0.; n = 0
    for s in seeds:
        r = safe_battle(Team.from_state(state), Team.from_state(opp), s)
        if r is None: continue
        total += 1. if r == 0 else (.5 if r == 2 else 0.); n += 1
    return total / max(1, n)


def light_copy(core, side):
    """Deep copy of the simulator core that shares the (untouched) other seat's objects, the identity registry and
    the previous battle boards; this seat's stale legal-action cache is dropped."""
    import copy
    other = core.opponent if side == 0 else core.agent
    theirs = 'opponent' if side == 0 else 'agent'; mine = 'agent' if side == 0 else 'opponent'
    memo = {id(other): other, id(other.team): other.team, id(other.shop): other.shop}
    for k in (f'team_{theirs}', f'shop_{theirs}', f'_{theirs}_actions_cache', '_legality_identities',
              'previous_opponent_state', 'previous_agent_state', 'previous_previous_opponent_state', 'previous_previous_agent_state'):
        v = getattr(core, k, None)
        if v is not None: memo[id(v)] = v
    c = copy.deepcopy(core, memo)
    setattr(c, f'_{mine}_actions_cache', None)
    return c


def board_signature(core, side):
    p = core.agent if side == 0 else core.opponent
    team = [(s.pet.name, s.pet.level, s.pet.experience, s.pet.attack, s.pet.health, s.pet.status) for s in p.team if not s.empty]
    shop = [(str(getattr(x.obj, 'name', '')), x.frozen) for x in getattr(p.shop, 'slots', [])]
    return json.dumps([team, p.gold, shop], default=str)


def _enumerate_task(args):
    """DFS (buy pet / buy food / sell / combine; one placement per empty slot) from one first action on a pickled
    public core. Returns {signature: (end-of-turn board, action codes)}."""
    core_bytes, side, first, max_depth, max_nodes = args
    from sapai_gym1.rl.action_schema import FlatActionCodec, ActionType
    from sapai_gym1.rl.end_freeze import expand_end_freeze
    allowed = {ActionType.BUY_PET, ActionType.BUY_FOOD, ActionType.SELL, ActionType.COMBINE}

    def moves(core):
        p = core.agent if side == 0 else core.opponent
        out, seen_empty = [], set()
        for code in sorted(core._avail_actions(side)):
            a = FlatActionCodec.decode(code)
            if a.action_type not in allowed: continue
            if a.action_type == ActionType.BUY_PET and p.team[a.target].empty:
                if a.source in seen_empty: continue
                seen_empty.add(a.source)
            out.append(code)
        return out

    def end_board(core):
        c = light_copy(core, side)
        try:
            for prim in expand_end_freeze(c, side, 0): c.resolve_action(agent_idx=side, action=FlatActionCodec.encode(prim))
        except ValueError:
            return None
        return (c.agent if side == 0 else c.opponent).team.state

    root = pickle.loads(core_bytes)
    results, visited, nodes = {}, set(), 0
    if first is None:
        stack = [(root, [])]
    else:
        c = light_copy(root, side); c.resolve_action(agent_idx=side, action=first); stack = [(c, [first])]
    while stack and nodes < max_nodes:
        core, path = stack.pop(); nodes += 1
        sig = board_signature(core, side)
        if sig in visited: continue
        visited.add(sig)
        board = end_board(core)
        if board is not None: results.setdefault(sig, (board, path))
        if len(path) >= max_depth or first is None: continue
        for code in moves(core):
            c = light_copy(core, side); c.resolve_action(agent_idx=side, action=code); stack.append((c, path + [code]))
    return results


def _expand_task(args):
    """Rebuild one search state by replaying `path` from the pickled root, return its end-of-turn board and the
    fingerprints of its children (one per useful action). States are never shipped between processes."""
    root_bytes, side, path, expand = args
    from sapai_gym1.rl.action_schema import FlatActionCodec, ActionType
    from sapai_gym1.rl.end_freeze import expand_end_freeze
    allowed = {ActionType.BUY_PET, ActionType.BUY_FOOD, ActionType.SELL, ActionType.COMBINE}
    core = pickle.loads(root_bytes)
    cache = '_agent_actions_cache' if side == 0 else '_opponent_actions_cache'
    for code in path:
        setattr(core, cache, None)  # the engine's legal-action cache goes stale after an action
        core.resolve_action(agent_idx=side, action=code)
    setattr(core, cache, None)
    sig = board_signature(core, side)
    board, burn = end_with_burn(core, side)
    children = []
    if expand:
        p = core.agent if side == 0 else core.opponent; seen_empty = set()
        for code in sorted(core._avail_actions(side)):
            a = FlatActionCodec.decode(code)
            if a.action_type not in allowed: continue
            if a.action_type == ActionType.BUY_PET and p.team[a.target].empty:
                if a.source in seen_empty: continue
                seen_empty.add(a.source)
            child = light_copy(core, side); child.resolve_action(agent_idx=side, action=code)
            children.append((board_signature(child, side), code))
    return sig, (None if board is None else (board, burn)), children


def end_with_burn(core, side):
    """End the turn from this state. The rules only allow END at zero gold, so leftover gold is rolled away first
    (the rolls happen after the last purchase, so the board is unchanged). Returns (board, rolls) or (None, 0)."""
    from sapai_gym1.rl.action_schema import FlatActionCodec, ActionType, AutoregressiveAction
    from sapai_gym1.rl.end_freeze import expand_end_freeze
    c = light_copy(core, side); p = c.agent if side == 0 else c.opponent
    roll = FlatActionCodec.encode(AutoregressiveAction(ActionType.ROLL)); burn = 0
    cache = '_agent_actions_cache' if side == 0 else '_opponent_actions_cache'
    try:
        while p.gold > 0 and burn < 12:
            setattr(c, cache, None)
            if roll not in c._avail_actions(side): return None, 0
            c.resolve_action(agent_idx=side, action=roll); burn += 1
        setattr(c, cache, None)
        for prim in expand_end_freeze(c, side, 0): c.resolve_action(agent_idx=side, action=FlatActionCodec.encode(prim))
    except ValueError:
        return None, 0
    return (c.agent if side == 0 else c.opponent).team.state, burn


def enumerate_boards(pool, core, side, max_depth=8, max_states=2000):
    """Breadth-first, globally de-duplicated enumeration of end-of-turn boards (no rolls before the last purchase;
    leftover gold is rolled away). Returns [(board, path, burn_rolls)]."""
    root_bytes = pickle.dumps(core)
    level, seen, out = [()], {board_signature(core, side)}, {}
    for depth in range(max_depth + 1):
        if not level: break
        results = pool.map(_expand_task, [(root_bytes, side, path, depth < max_depth) for path in level], chunksize=1)
        nxt = []
        for path, (sig, ended, children) in zip(level, results):
            if ended is not None: out.setdefault(sig, (ended[0], list(path), ended[1]))
            for child_sig, code in children:
                if child_sig in seen or len(seen) >= max_states: continue
                seen.add(child_sig); nxt.append(path + (code,))
        level = nxt
    return list(out.values())


def _versus_many_task(args):
    """Mean battle score of one board against several specific boards, a few seeds each."""
    state, opps, seeds = args
    from sapai import Team
    total = 0.; n = 0
    for opp in opps:
        for s in seeds:
            r = safe_battle(Team.from_state(state), Team.from_state(opp), s)
            if r is None: continue
            total += 1. if r == 0 else (.5 if r == 2 else 0.); n += 1
    return total / max(1, n)


class BattleScorer:
    def __init__(self, library_path, workers, min_boards=48):
        global _LIB
        raw = pickle.loads(Path(library_path).read_bytes())
        _LIB = {t: [strip_rng(r['board']) for r in rows] for t, rows in raw.items()}
        self.prev_names = {t: [pet_names(r['prev']) for r in rows] for t, rows in raw.items()}
        del raw  # only the RNG-free boards stay resident (shared copy-on-write by the forked workers)
        self.turns = sorted(t for t, rows in _LIB.items() if len(rows) >= min_boards)
        self.pool = mp.get_context('fork').Pool(workers)  # fork BEFORE any CUDA initialisation

    def library_turn(self, turn):
        return min(self.turns, key=lambda t: (abs(t - turn), -t))

    def opponents(self, turn, m, rng, prev=None, pool=96):
        """Library indices for this turn; with the opponent's last revealed board, the `pool` entries whose
        previous board shares the most pets with it (random tie-break), else uniform."""
        t = self.library_turn(turn)
        n = len(_LIB[t])
        if prev is None or not pet_names(prev):
            return t, [rng.randrange(n) for _ in range(m)]
        want = pet_names(prev)
        ranked = sorted(range(n), key=lambda i: (-sum((want & self.prev_names[t][i]).values()), rng.random()))
        near = ranked[:max(pool, m)]
        return t, [rng.choice(near) for _ in range(m)]

    def score(self, states, turn, m, rng, prev=None, pool=96):
        t, idx = self.opponents(turn, m, rng, prev, pool)
        seeds = [rng.getrandbits(31) for _ in range(m)]  # common random numbers across candidates
        return self.pool.map(_score_task, [(s, t, idx, seeds) for s in states])

    def versus_many(self, states, opps, seeds_per, rng):
        seeds = [rng.getrandbits(31) for _ in range(seeds_per)]
        return self.pool.map(_versus_many_task, [(s, opps, seeds) for s in states])

    def versus(self, states, opp, n, rng):
        seeds = [rng.getrandbits(31) for _ in range(n)]
        return self.pool.map(_versus_task, [(s, opp, seeds) for s in states])


class PredictorScorer:
    """Battle-predictor screening on the GPU (bp_train.BattleNet). Uses the same library opponents as BattleScorer;
    only screens - final choices are always re-scored with exact battle simulation."""
    def __init__(self, scorer, path, device='cuda'):
        import torch
        from bp_train import BattleNet
        from bp_common import encode_many
        self.torch, self.scorer, self.dev = torch, scorer, device
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.net = BattleNet().to(device).eval(); self.net.load_state_dict(ck['model'])
        self.encode_many = encode_many
        self.lib = {}
        for t, boards in _LIB.items():
            ids, st, num = encode_many(boards)
            self.lib[t] = tuple(torch.as_tensor(x, device=device) for x in (ids.astype('int64'), st.astype('int64'), num))

    def screen_vs(self, states, opps):
        """Mean predicted equity of each board against an explicit list of opponent boards."""
        torch = self.torch
        a = tuple(torch.as_tensor(x, device=self.dev) for x in (lambda e: (e[0].astype('int64'), e[1].astype('int64'), e[2]))(self.encode_many(states)))
        b = tuple(torch.as_tensor(x, device=self.dev) for x in (lambda e: (e[0].astype('int64'), e[1].astype('int64'), e[2]))(self.encode_many(opps)))
        n, k = len(states), len(opps); out = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            step = max(1, 16384 // k)
            for lo in range(0, n, step):
                hi = min(n, lo + step)
                aa = tuple(x[lo:hi].repeat_interleave(k, 0) for x in a); bb = tuple(x.repeat(hi - lo, *([1] * (x.dim() - 1))) for x in b)
                out.append(self.net.equity(aa, bb).float().view(hi - lo, k).mean(1))
        return torch.cat(out).cpu().tolist()

    def screen(self, states, turn, m, rng, prev=None, pool=96):
        """Mean predicted equity of each board against m sampled library opponents."""
        torch = self.torch
        t, idx = self.scorer.opponents(turn, m, rng, prev, pool)
        ids, st, num = self.encode_many(states)
        a = tuple(torch.as_tensor(x, device=self.dev) for x in (ids.astype('int64'), st.astype('int64'), num))
        oi = torch.as_tensor(idx, device=self.dev); b = tuple(x[oi] for x in self.lib[t])
        n, k = len(states), len(idx); out = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for lo in range(0, n, max(1, 16384 // k)):
                hi = min(n, lo + max(1, 16384 // k))
                aa = tuple(x[lo:hi].repeat_interleave(k, 0) for x in a); bb = tuple(x.repeat(hi - lo, *([1] * (x.dim() - 1))) for x in b)
                out.append(self.net.equity(aa, bb).float().view(hi - lo, k).mean(1))
        return torch.cat(out).cpu().tolist()


@dataclasses.dataclass
class PlannerConfig:
    candidates: int = 16      # policy plans per planning call (first one greedy)
    opponents: int = 64       # library boards per candidate
    margin: float = 0.03      # switch from the greedy plan only if this much better
    max_plan_steps: int = 60
    condition: bool = False   # sample opponents whose previous board resembles the real opponent's last board
    position: bool = False    # before ending the turn, try every reachable pet order (exact END effects)
    position_margin: float = 0.02
    enumerate: bool = False   # candidates = every reachable no-roll board (not only policy samples)
    enum_depth: int = 7
    enum_nodes: int = 400
    enum_screen: int = 16     # battles per board in the screen
    enum_top: int = 8         # boards re-scored on the full opponent set
    predictor: object = None  # PredictorScorer: GPU screening instead of short battle screens
    raw_samples: int = 8      # policy-with-rolls continuations that estimate the policy's own turn value
    enum_margin: float = 0.06 # an enumerated plan must beat the policy's expected turn by this much
    position_only: bool = False  # play exactly like the policy; only reorder the pets right before END
    oracle: bool = False      # DIAGNOSTIC ONLY (privileged): score against the opponent's actual final board
    prev_opponent: bool = False  # public: position against the opponent's last revealed board
    sim_opponent: bool = False   # public: position against simulated opponent turns (its revealed board, fresh
                                 # shops, the known opponent policy playing its turn)
    sim_samples: int = 8
    diag_oracle: bool = False    # logging only (privileged): how much of the oracle's positioning gain is captured
    sim_memory: str = 'zeros'    # opponent policy memory in simulated turns: 'zeros', 'self', or 'true' (DIAGNOSTIC)
    sim_keep_frozen: bool = False  # DIAGNOSTIC (privileged): keep the opponent's actual frozen shop items
    screen_seeds: int = 1     # sim positioning: battles per simulated board when screening every order
    exact_top: int = 6        # sim positioning: orders (besides the current one) played out exactly
    mix_prev: bool = False    # sim positioning: also battle the opponent's last revealed board
    bp_position: bool = False # sim positioning by the battle predictor only (fast labelling), no exact battles
    sim_seeds: int = 4
    lookahead: bool = False   # value = equity now + gamma * equity of the policy's next board (2-turn horizon)
    look_samples: int = 3
    gamma: float = 1.0
    roll_search: bool = False # decide rolls by expectimax: E[best plan after a sampled roll] vs best plan now
    roll_outcomes: int = 4
    roll_candidates: int = 8
    roll_margin: float = 0.02
    pool: int = 96


import time as _time
from contextlib import contextmanager


@contextmanager
def _timed(stats, key):
    t0 = _time.time()
    try: yield
    finally: stats[key] = round(stats.get(key, 0.) + _time.time() - t0, 2)


class TurnPlanner:
    def __init__(self, actor, scorer, config=PlannerConfig(), seed=0):
        from sapai_gym1.rl.action_schema import ActionType
        self.actor, self.scorer, self.cfg = actor, scorer, config
        self.rng = random.Random(seed)
        self.ROLL, self.END = int(ActionType.ROLL), int(ActionType.END_TURN)
        self.plan = None; self.stats = dict(calls=0, overrides=0, rolls=0, replans=0, gain=0.)
        self.last_decision = {}  # turn -> greedy/chosen boards and predicted scores (diagnostics)
        self.positioned_turn = None; self.turn_seen = None; self.stats.update(positions=0, position_gain=0., roll_evals=0, roll_chosen=0, roll_vetoed=0)

    # ---------- helpers
    @staticmethod
    def _player(core, side):
        return core.agent if side == 0 else core.opponent

    def _fingerprint(self, core, side):
        p = self._player(core, side)
        return json.dumps(strip_rng([p.team.state, p.gold]), sort_keys=True, default=str)

    def _public_clone(self, env, side):
        clone = env.clone()
        self._player(clone.core, side).shop.enable_paired_tape(self.rng.getrandbits(62), side)
        return clone

    @staticmethod
    def _apply(core, side, joint):
        from sapai_gym1.rl.action_schema import FlatActionCodec, ActionType
        from sapai_gym1.rl.end_freeze import expand_end_freeze
        if joint.primitive.action_type == ActionType.END_TURN:
            prims = expand_end_freeze(core, side, joint.end_freeze_mask or 0)
        else:
            prims = (joint.primitive,)
        for p in prims:
            code = FlatActionCodec.encode(p)
            if code not in core._avail_actions(side): raise ValueError('illegal planned action')
            core.resolve_action(agent_idx=side, action=code)

    def _roll_joint(self):
        from sapai_gym1.rl.action_schema import AutoregressiveAction, ActionType
        from sapai_gym1.rl.joint_action import JointAction
        return JointAction(AutoregressiveAction(ActionType.ROLL), 0)

    def _request(self, obs, memory, forced=None):
        from teacher_world import ActorRequest, observation_copy, owned_memory
        return ActorRequest(observation_copy(obs), owned_memory(memory), self.actor.owner, forced)

    # ---------- candidate plans
    def _rollouts(self, env, side, memory, k=None, allow_roll=False, all_greedy=False, keep=False):
        k = k or self.cfg.candidates
        clones = [self._public_clone(env, side) for _ in range(k)]
        return self._lockstep(clones, [memory.clone() for _ in range(k)], side, allow_roll, all_greedy, keep)

    def _lockstep(self, clones, mems, side, allow_roll=False, all_greedy=False, keep=False):
        """Play the policy on each clone (this seat only) until it ends the turn. Candidate 0 is greedy, the rest
        sampled unless all_greedy. Returns plans, marks, final boards, done flags (+ clones, memories if keep)."""
        import torch
        from sapai_gym1.rl.action_schema import AutoregressiveAction
        from sapai_gym1.rl.joint_action import JointAction
        model, k = self.actor.model, len(clones)
        plans = [[] for _ in range(k)]; marks = [[] for _ in range(k)]
        active = [i for i in range(k) if not self._player(clones[i].core, side).turn_ended]; steps = 0
        while active and steps < self.cfg.max_plan_steps:
            steps += 1
            nxt = []
            for lo in range(0, len(active), 64):
                part = active[lo:lo + 64]
                reqs = [self._request(clones[i].encoder.encode(clones[i].core, side), mems[i]) for i in part]
                with torch.inference_mode():
                    _, feats = self.actor._encode(reqs)
                    if not allow_roll:
                        obs = dict(feats.observation); tm = obs['type_mask'].clone(); tm[:, self.ROLL] = 0
                        only_roll = tm.sum(dim=1) == 0  # rolling is the only legal move here: keep it
                        tm[only_roll] = obs['type_mask'][only_roll]; obs['type_mask'] = tm
                        feats = dataclasses.replace(feats, observation=obs)
                    kw = dict(allow_end_freeze=model.config.end_freeze, action_only=True)
                    det = model.decode_decision(feats, None, True, **kw)
                    smp = det if all_greedy else model.decode_decision(feats, None, False, **kw)
                for row, i in enumerate(part):
                    out = det if (i == 0 or all_greedy) else smp
                    joint = JointAction(AutoregressiveAction(*out.actions[row].cpu().tolist()), int(out.end_freeze_mask[row]))
                    self._apply(clones[i].core, side, joint)
                    plans[i].append(joint); marks[i].append(self._fingerprint(clones[i].core, side))
                    mems[i] = out.memory[row].detach().float().cpu().clone()
                    if not self._player(clones[i].core, side).turn_ended: nxt.append(i)
            active = nxt
        finals = [self._player(c.core, side).team.state for c in clones]
        done = [self._player(c.core, side).turn_ended for c in clones]
        return (plans, marks, finals, done, clones, mems) if keep else (plans, marks, finals, done)

    def _forced(self, clones, mems, side, action_lists):
        """Execute given joint-action lists on clones through the actor (forced), keeping the policy memory exact."""
        k = len(clones); at = [0] * k
        while True:
            live = [i for i in range(k) if at[i] < len(action_lists[i])]
            if not live: break
            for lo in range(0, len(live), 64):
                part = live[lo:lo + 64]
                reqs = [self._request(clones[i].encoder.encode(clones[i].core, side), mems[i], action_lists[i][at[i]]) for i in part]
                outs = self.actor.act_many(reqs)
                for i, o in zip(part, outs):
                    self._apply(clones[i].core, side, action_lists[i][at[i]]); mems[i] = o.next_memory; at[i] += 1
        return clones, mems

    def _advance_turn(self, clone, side, won):
        """Move only this seat of a clone to its next shop (battle result given), as env.step would."""
        core = clone.core; p = self._player(core, side); mine = 'agent' if side == 0 else 'opponent'
        if won: p.wins += 1
        else: p.lives = max(0, p.lives - 1)
        p.lf_winner = won; core.turn += 1
        p.start_turn(winner=won)
        setattr(core, f'reorder_count_{mine}', 0); setattr(core, f'shop_action_count_{mine}', 0)
        getattr(core, f'_freeze_toggled_{mine}').clear(); core._legality_identities.clear()
        setattr(core, f'_seen_team_orders_{mine}', {core._team_order_key(p)})
        setattr(core, f'_{mine}_actions_cache', None)

    def _next_turn_values(self, post, side, turn, eq_now, samples, rng_seed, prev):
        """post: list of (clone after END, memory). For each, the mean equity of the policy's NEXT board (rolls
        included) over `samples` fresh shops, battle result sampled from eq_now. Same opponents for all."""
        import random as _r
        clones, mems, owner = [], [], []
        for ci, (c, m) in enumerate(post):
            for _ in range(samples):
                c2 = self._public_clone(c, side); self._advance_turn(c2, side, self.rng.random() < eq_now[ci])
                clones.append(c2); mems.append(m.clone()); owner.append(ci)
        _, _, finals, done = self._lockstep(clones, mems, side, allow_roll=True, all_greedy=True)
        ok = [i for i in range(len(clones)) if done[i]]
        vals = self.scorer.score([finals[i] for i in ok], turn + 1, self.cfg.opponents, _r.Random(rng_seed), prev=prev, pool=self.cfg.pool) if ok else []
        out = [[] for _ in post]
        for i, v in zip(ok, vals): out[owner[i]].append(v)
        return [sum(v) / len(v) if v else 0. for v in out]

    def _enum_plan(self, env, side, memory):
        """Best reachable no-roll board vs the policy's own greedy plan (full opponent set, common random numbers)."""
        import random as _r
        from sapai_gym1.rl.action_schema import FlatActionCodec, AutoregressiveAction
        from sapai_gym1.rl.joint_action import JointAction
        self.stats['calls'] += 1
        g_plans, g_marks, g_finals, g_done = self._rollouts(env, side, memory, k=1)
        if not g_done[0]: return None
        clone = self._public_clone(env, side); core_bytes = pickle.dumps(clone.core)
        firsts = [None] + sorted(clone.core._avail_actions(side))
        merged = {}
        for part in self.scorer.pool.map(_enumerate_task, [(core_bytes, side, f, self.cfg.enum_depth, self.cfg.enum_nodes) for f in firsts]):
            for k, v in part.items(): merged.setdefault(k, v)
        items = list(merged.values())
        self.stats['enum_boards'] = self.stats.get('enum_boards', 0) + len(items)
        prev = self._opp_prev(env, side)
        if len(items) > self.cfg.enum_top:
            if self.cfg.predictor is not None:
                screen = self.cfg.predictor.screen([b for b, _ in items], env.core.turn, 128, _r.Random(self.rng.getrandbits(62)), prev=prev, pool=self.cfg.pool)
            else:
                screen = self.scorer.score([b for b, _ in items], env.core.turn, self.cfg.enum_screen, _r.Random(self.rng.getrandbits(62)), prev=prev, pool=self.cfg.pool)
            items = [items[i] for i in sorted(range(len(items)), key=lambda i: -screen[i])[:self.cfg.enum_top]]
        self.opp_seed = self.rng.getrandbits(62)
        full = self.scorer.score([g_finals[0]] + [b for b, _ in items], env.core.turn, self.cfg.opponents, _r.Random(self.opp_seed), prev=prev, pool=self.cfg.pool)
        best = max(range(1, len(full)), key=lambda i: full[i]) if len(full) > 1 else 0
        use_enum = best > 0 and full[best] - full[0] >= self.cfg.margin
        if use_enum:
            self.stats['overrides'] += 1; self.stats['gain'] += full[best] - full[0]
            codes = items[best - 1][1]
            actions = [JointAction(FlatActionCodec.decode(c), 0) for c in codes] + [JointAction(AutoregressiveAction(self.END), 0)]
            plan = dict(actions=actions, marks=[None] * len(actions), at=0, value=full[best], enum=True)
            chosen = items[best - 1][0]
        else:
            plan = dict(actions=g_plans[0], marks=g_marks[0], at=0, value=full[0]); chosen = g_finals[0]
        self.last_decision[env.core.turn] = dict(greedy=g_finals[0], chosen=chosen, predGreedy=full[0],
                                                 predChosen=full[best] if use_enum else full[0], override=use_enum)
        return plan

    def _turn_choice(self, env, side, memory):
        """Once per turn: best enumerated no-roll board vs the policy's own expected turn (with its rolls, sampled on
        fresh-shop clones). Returns an executable plan, or None to follow the policy for the rest of the turn."""
        import random as _r
        from sapai_gym1.rl.action_schema import FlatActionCodec, AutoregressiveAction
        from sapai_gym1.rl.joint_action import JointAction
        self.stats['calls'] += 1
        prev = self._opp_prev(env, side)
        # the policy's own turn, rolls included: expected equity over sampled shops
        with _timed(self.stats, 't_raw'):
            _, _, r_finals, r_done, r_clones, r_mems = self._rollouts(env, side, memory, k=self.cfg.raw_samples, allow_roll=True, all_greedy=True, keep=True)
        raw_idx = [i for i, d in enumerate(r_done) if d]
        raw_states = [r_finals[i] for i in raw_idx]
        # every reachable no-roll board with the current shop
        with _timed(self.stats, 't_enum'):
            clone = self._public_clone(env, side)
            items = enumerate_boards(self.scorer.pool, clone.core, side, self.cfg.enum_depth, self.cfg.enum_nodes)
        self.stats['enum_boards'] = self.stats.get('enum_boards', 0) + len(items)
        if not items or not raw_states: return None
        sim_opps = None
        if self.cfg.sim_opponent:
            with _timed(self.stats, 't_sim'):
                sim_opps = self._simulated_opponent_boards(env, side, self.cfg.sim_samples)
            if not sim_opps: sim_opps = None
        if len(items) > self.cfg.enum_top:
            if self.cfg.predictor is not None and sim_opps is not None:
                screen = self.cfg.predictor.screen_vs([it[0] for it in items], sim_opps)
            elif self.cfg.predictor is not None:
                screen = self.cfg.predictor.screen([it[0] for it in items], env.core.turn, 128, _r.Random(self.rng.getrandbits(62)), prev=prev, pool=self.cfg.pool)
            else:
                screen = self.scorer.score([it[0] for it in items], env.core.turn, self.cfg.enum_screen, _r.Random(self.rng.getrandbits(62)), prev=prev, pool=self.cfg.pool)
            items = [items[i] for i in sorted(range(len(items)), key=lambda i: -screen[i])[:self.cfg.enum_top]]
        seed = self.rng.getrandbits(62)
        with _timed(self.stats, 't_exact'):
            if sim_opps is not None:
                scores = self.scorer.versus_many(raw_states + [it[0] for it in items], sim_opps, self.cfg.sim_seeds, _r.Random(seed))
            else:
                scores = self.scorer.score(raw_states + [it[0] for it in items], env.core.turn, self.cfg.opponents, _r.Random(seed), prev=prev, pool=self.cfg.pool)
        if self.cfg.lookahead:
            # two-turn value: battle now + the policy's next board (its rolls included), for the raw samples and
            # every finalist, with the same next-turn opponents for all of them
            from sapai_gym1.rl.action_schema import FlatActionCodec as _F, AutoregressiveAction as _A
            from sapai_gym1.rl.joint_action import JointAction as _J
            with _timed(self.stats, 't_look'):
                lists = [[_J(_F.decode(c), 0) for c in it[1]] + [self._roll_joint()] * it[2] + [_J(_A(self.END), 0)] for it in items]
                e_clones, e_mems = self._forced([self._public_clone(env, side) for _ in items], [memory.clone() for _ in items], side, lists)
                post = [(r_clones[i], r_mems[i]) for i in raw_idx] + list(zip(e_clones, e_mems))
                nxt = self._next_turn_values(post, side, env.core.turn, scores, self.cfg.look_samples, self.rng.getrandbits(62), prev)
            scores = [a + self.cfg.gamma * b for a, b in zip(scores, nxt)]
        v_raw = sum(scores[:len(raw_states)]) / len(raw_states)
        enum_scores = scores[len(raw_states):]
        best = max(range(len(items)), key=lambda i: enum_scores[i])
        use = enum_scores[best] - v_raw >= self.cfg.enum_margin
        self.last_decision[env.core.turn] = dict(greedy=raw_states[0], chosen=items[best][0] if use else raw_states[0],
                                                 predGreedy=v_raw, predChosen=enum_scores[best] if use else v_raw, override=use)
        if not use: return None
        self.stats['overrides'] += 1; self.stats['gain'] += enum_scores[best] - v_raw
        board, path, burn = items[best]
        actions = ([JointAction(FlatActionCodec.decode(c), 0) for c in path] + [self._roll_joint()] * burn
                   + [JointAction(AutoregressiveAction(self.END), 0)])
        return dict(actions=actions, marks=[None] * len(actions), at=0, value=enum_scores[best], enum=True)

    def _oracle_board(self, env, side):
        """Privileged diagnostic: the opponent's final board this turn, by playing its deterministic greedy policy
        forward on a full clone (its real shop tape). Never used by a legitimate configuration."""
        from sapai_gym1.rl.joint_action import step_decisions
        c = env.clone(); other = 1 - side; mem = self.opp_memory.clone()
        p = self._player(c.core, other)
        steps = 0
        while not p.turn_ended and steps < 80:
            o = self.actor.act_many([self._request(c.encoder.encode(c.core, other), mem)])[0]
            self._apply(c.core, other, o.action); mem = o.next_memory; steps += 1
        return p.team.state

    def _simulated_opponent_boards(self, env, side, n, memory=None):
        """Public opponent model: rebuild the opponent at its last REVEALED board with a fresh shop tape and a new
        turn (start-of-turn effects, gold, tier-correct shop), then let the opponent policy (the same checkpoint,
        unknown memory -> zeros) play the turn. Nothing hidden is read: the opponent's current team, shop, gold
        and tape in the clone are all overwritten before use."""
        import torch
        from sapai import Team
        last = env.core.previous_opponent_state if side == 0 else env.core.previous_agent_state
        if last is None: return []
        other = 1 - side; theirs = 'agent' if other == 0 else 'opponent'
        clones = []
        for _ in range(n):
            c = env.clone(); core = c.core; o = self._player(core, other)
            o.team = Team.from_state(last.state)
            if not self.cfg.sim_keep_frozen:
                for slot in getattr(o.shop, 'slots', []):  # its frozen items are hidden: unfreeze so all re-roll
                    slot.frozen = False
            o.shop.enable_paired_tape(self.rng.getrandbits(62), other)
            o.turn = core.turn - 1; o.turn_ended = True
            o.start_turn(winner=o.lf_winner)
            setattr(core, f'reorder_count_{theirs}', 0); setattr(core, f'shop_action_count_{theirs}', 0)
            getattr(core, f'_freeze_toggled_{theirs}').clear()
            setattr(core, f'_seen_team_orders_{theirs}', {core._team_order_key(o)})
            setattr(core, f'_{theirs}_actions_cache', None)
            clones.append(c)
        if self.cfg.sim_memory == 'true':  # DIAGNOSTIC: the opponent's actual recurrent state (privileged)
            base = self.opp_memory
        else:
            base = memory if (self.cfg.sim_memory == 'self' and memory is not None) else torch.zeros(self.actor.model.config.memory_dim)
        mems = [base.clone() for _ in clones]
        _, _, finals, done = self._lockstep(clones, mems, other, allow_roll=True, all_greedy=True)
        return [f for f, d in zip(finals, done) if d]

    def _decide_position_only(self, env, side, memory, obs, raw):
        turn = env.core.turn
        if self.plan is not None and self.plan['at'] < len(self.plan['actions']):
            return self._execute(env, side, memory, obs, raw, self.plan)  # finishing a reorder (then END)
        self.plan = None
        if raw.action.primitive.action_type == self.END and self.positioned_turn != turn:
            self.positioned_turn = turn
            moves = self._position(env, side, raw.action)
            if moves is not None:
                self.plan = dict(actions=moves, marks=[None] * len(moves), at=0, value=None)
                return self._execute(env, side, memory, obs, raw, self.plan)
        return raw

    def _decide_enum(self, env, side, memory, obs, raw):
        turn = env.core.turn
        if self.turn_seen != turn:  # first decision of this turn: choose between the policy and an enumerated plan
            self.turn_seen = turn
            self.plan = self._turn_choice(env, side, memory)
        plan = self.plan
        if plan is None:  # follow the policy this turn (its rolls included); optimise the order before it ends
            if raw.action.primitive.action_type == self.END and self.cfg.position and self.positioned_turn != turn:
                self.positioned_turn = turn
                moves = self._position(env, side, raw.action)
                if moves is not None:
                    self.plan = dict(actions=moves, marks=[None] * len(moves), at=0, value=None)
                    return self._execute(env, side, memory, obs, raw, self.plan)
            if raw.action.primitive.action_type == self.ROLL: self.stats['rolls'] += 1
            return raw
        if plan['at'] >= len(plan['actions']):
            self.plan = None; return raw
        if raw.action.primitive.action_type == self.ROLL:
            self.stats['rolls_suppressed'] = self.stats.get('rolls_suppressed', 0) + 1  # committed plan beat the policy's rolls
        return self._execute(env, side, memory, obs, raw, plan)

    def _new_plan(self, env, side, memory):
        if self.cfg.enumerate: return self._enum_plan(env, side, memory)
        self.stats['calls'] += 1
        plans, marks, finals, done = self._rollouts(env, side, memory)
        keys = [json.dumps(strip_rng(f), sort_keys=True, default=str) for f in finals]
        unique = {}
        for i, key in enumerate(keys):
            if done[i]: unique.setdefault(key, i)
        order = list(unique.values())
        if 0 not in order: return None  # greedy plan did not finish: keep raw behaviour
        import random as _r
        self.opp_seed = self.rng.getrandbits(62)
        scores = dict(zip(order, self.scorer.score([finals[i] for i in order], env.core.turn, self.cfg.opponents,
                                                    _r.Random(self.opp_seed), prev=self._opp_prev(env, side), pool=self.cfg.pool)))
        best = max(order, key=lambda i: (scores[i], i == 0))
        choice = best if scores[best] - scores[0] >= self.cfg.margin else 0
        if choice != 0:
            self.stats['overrides'] += 1; self.stats['gain'] += scores[choice] - scores[0]
        self.last_decision[env.core.turn] = dict(greedy=finals[0], chosen=finals[choice], predGreedy=scores[0],
                                                 predChosen=scores[choice], override=choice != 0)
        return dict(actions=plans[choice], marks=marks[choice], at=0, value=scores[choice])

    def _opp_prev(self, env, side):
        if not self.cfg.condition: return None
        last = env.core.previous_opponent_state if side == 0 else env.core.previous_agent_state
        return last.state if last is not None else None

    def _roll_value(self, env, side, memory):
        """Expectimax value of rolling now: mean over independently sampled new shops of the best roll-free plan
        on that shop (the planner will re-plan on the real new shop). Same opponents/seeds for every outcome."""
        self.stats['roll_evals'] += 1
        roll = self._roll_joint(); opp_seed = getattr(self, 'opp_seed', None) or self.rng.getrandbits(62)  # same opponents as the stay plan
        values = []
        for _ in range(self.cfg.roll_outcomes):
            c = self._public_clone(env, side)
            mem = self.actor.act_many([self._request(c.encoder.encode(c.core, side), memory, roll)])[0].next_memory
            self._apply(c.core, side, roll)
            plans, _, finals, done = self._rollouts(c, side, mem, k=self.cfg.roll_candidates)
            states = [f for f, d in zip(finals, done) if d]
            if not states: continue
            import random as _r
            scores = self.scorer.score(states, env.core.turn, self.cfg.opponents, _r.Random(opp_seed), prev=self._opp_prev(env, side), pool=self.cfg.pool)
            values.append(max(scores))
        return sum(values) / len(values) if values else -1.

    # ---------- positioning: exact search over reachable pet orders right before END
    @staticmethod
    def _order_paths(core, side):
        """Shortest pop-insert move sequences to every reachable slot order, honouring the per-turn reorder cap and
        the rule that an order already seen this turn cannot be revisited. Keyed by pet sequence (empties ignored)."""
        from collections import deque
        player = core.agent if side == 0 else core.opponent
        count = core.reorder_count_agent if side == 0 else core.reorder_count_opponent
        seen = set(core._seen_team_orders_agent if side == 0 else core._seen_team_orders_opponent)
        start = tuple(core._team_order_key(player)); seen.add(start)
        budget = max(0, core.MAX_REORDERS_PER_TURN - count)
        best = {tuple(x for x in start if x is not None): ()}
        queue = deque([(start, ())]); visited = {start}
        while queue:
            order, path = queue.popleft()
            if len(path) >= min(budget, 4): continue
            for src in range(len(order)):
                if order[src] is None: continue
                for dst in range(len(order)):
                    if dst == src: continue
                    nxt = list(order); item = nxt.pop(src); nxt.insert(dst, item); nxt = tuple(nxt)
                    if nxt in visited or nxt in seen: continue
                    visited.add(nxt); step = path + ((src, dst),)
                    best.setdefault(tuple(x for x in nxt if x is not None), step)
                    queue.append((nxt, step))
        return list(best.values())

    def _position(self, env, side, end_joint):
        """Returns a list of joint actions (moves then END) or None to keep the current order."""
        with _timed(self.stats, 't_position'):
            return self._position_inner(env, side, end_joint)

    def _position_inner(self, env, side, end_joint):
        from sapai_gym1.rl.action_schema import AutoregressiveAction, ActionType
        from sapai_gym1.rl.joint_action import JointAction
        paths = self._order_paths(env.core, side)
        if len(paths) < 2: return None
        if self.cfg.predictor is not None and len(paths) > 7 and not (self.cfg.oracle or self.cfg.prev_opponent or self.cfg.sim_opponent):
            # cheap screen: reorder the current team (no END effects) and let the predictor rank the orders; only the
            # current order and the top 6 are then played out exactly (END effects can depend on position)
            import random as _r
            player = self._player(env.core, side)
            occ = [i for i, sl in enumerate(player.team) if not sl.empty]
            base = player.team.state
            def reordered(path):
                order = list(range(len(base['team'])))
                for a, b in path:
                    item = order.pop(a); order.insert(b, item)
                return dict(base, team=[base['team'][i] for i in order])
            approx = [reordered(p) for p in paths]
            screen = self.cfg.predictor.screen(approx, env.core.turn, 128, _r.Random(self.rng.getrandbits(62)), prev=self._opp_prev(env, side), pool=self.cfg.pool)
            keep = [0] + sorted(range(1, len(paths)), key=lambda i: -screen[i])[:6]
            paths = [paths[i] for i in keep]
        sim_opps = None
        if self.cfg.sim_opponent:
            with _timed(self.stats, 't_sim'):
                sim_opps = self._simulated_opponent_boards(env, side, self.cfg.sim_samples, getattr(self, 'cur_memory', None))
            if not sim_opps: return None
            if self.cfg.bp_position and self.cfg.predictor is not None:
                player = self._player(env.core, side); base = player.team.state
                def reordered_bp(path):
                    order = list(range(len(base['team'])))
                    for a, b in path:
                        item = order.pop(a); order.insert(b, item)
                    return dict(base, team=[base['team'][i] for i in order])
                opps = list(sim_opps)
                if self.cfg.mix_prev:
                    last = env.core.previous_opponent_state if side == 0 else env.core.previous_agent_state
                    if last is not None: opps.append(last.state)
                pred = self.cfg.predictor.screen_vs([reordered_bp(p) for p in paths], opps)
                best = max(range(len(paths)), key=lambda i: (pred[i], i == 0))
                self.stats['positions'] += 1
                if self.cfg.diag_oracle:
                    finals_d = []
                    for path in (paths[0], paths[best]):
                        clone = self._public_clone(env, side)
                        for j in [JointAction(AutoregressiveAction(ActionType.MOVE, a, b), 0) for a, b in path] + [end_joint]: self._apply(clone.core, side, j)
                        finals_d.append(self._player(clone.core, side).team.state)
                    real = self.scorer.versus(finals_d, self._oracle_board(env, side), 32, self.rng)
                    self.diag = getattr(self, 'diag', []) + [dict(turn=env.core.turn, n=len(paths), realCurrent=real[0],
                        realChosen=real[1] if (best != 0 and pred[best] - pred[0] >= self.cfg.position_margin) else real[0],
                        realBest=max(real), simCurrent=pred[0], simChosen=pred[best])]
                if best == 0 or pred[best] - pred[0] < self.cfg.position_margin: return None
                self.stats['position_gain'] += pred[best] - pred[0]
                return [JointAction(AutoregressiveAction(ActionType.MOVE, a, b), 0) for a, b in paths[best]] + [end_joint]
            if len(paths) > 7:  # cheap screen on reordered boards (no END effects), 1 battle per simulated board
                player = self._player(env.core, side); base = player.team.state
                def reordered(path):
                    order = list(range(len(base['team'])))
                    for a, b in path:
                        item = order.pop(a); order.insert(b, item)
                    return dict(base, team=[base['team'][i] for i in order])
                if self.cfg.mix_prev:
                    last = env.core.previous_opponent_state if side == 0 else env.core.previous_agent_state
                    if last is not None: sim_opps = sim_opps + [last.state]
                screen = self.scorer.versus_many([reordered(p) for p in paths], sim_opps, self.cfg.screen_seeds, self.rng)
                keep = [0] + sorted(range(1, len(paths)), key=lambda i: -screen[i])[:self.cfg.exact_top]
                paths = [paths[i] for i in keep]
        finals, plans = [], []
        for path in paths:
            clone = self._public_clone(env, side)
            joints = [JointAction(AutoregressiveAction(ActionType.MOVE, a, b), 0) for a, b in path] + [end_joint]
            try:
                for j in joints: self._apply(clone.core, side, j)
            except ValueError:
                continue
            finals.append(self._player(clone.core, side).team.state); plans.append(joints)
        if len(finals) < 2 or plans[0][:-1]: return None  # first path must be "no moves"
        import random as _r
        if self.cfg.sim_opponent:
            full = self.scorer.versus_many(finals, sim_opps, self.cfg.sim_seeds, self.rng)
            best = max(range(len(full)), key=lambda i: (full[i], i == 0))
            if self.cfg.diag_oracle:
                real = self.scorer.versus(finals, self._oracle_board(env, side), 32, self.rng)
                chosen = best if (best != 0 and full[best] - full[0] >= self.cfg.position_margin) else 0
                self.diag = getattr(self, 'diag', []) + [dict(turn=env.core.turn, n=len(finals), realCurrent=real[0],
                    realChosen=real[chosen], realBest=max(real), simCurrent=full[0], simChosen=full[chosen])]
            self.stats['positions'] += 1
            if best == 0 or full[best] - full[0] < self.cfg.position_margin: return None
            self.stats['position_gain'] += full[best] - full[0]
            return plans[best]
        if self.cfg.oracle or self.cfg.prev_opponent:
            if self.cfg.oracle:
                opp = self._oracle_board(env, side)
            else:
                last = env.core.previous_opponent_state if side == 0 else env.core.previous_agent_state
                if last is None: return None  # turn 1: nothing revealed yet
                opp = last.state
            full = self.scorer.versus(finals, opp, self.cfg.opponents, self.rng)
            best = max(range(len(full)), key=lambda i: (full[i], i == 0))
            self.stats['positions'] += 1
            if best == 0 or full[best] - full[0] < self.cfg.position_margin: return None
            self.stats['position_gain'] += full[best] - full[0]
            return plans[best]
        prev = self._opp_prev(env, side)
        # screen every order cheaply, then re-score the current order and the top 6 on the full opponent set
        if self.cfg.predictor is not None:
            screen = self.cfg.predictor.screen(finals, env.core.turn, 128, _r.Random(self.rng.getrandbits(62)), prev=prev, pool=self.cfg.pool)
        else:
            screen = self.scorer.score(finals, env.core.turn, 24, _r.Random(self.rng.getrandbits(62)), prev=prev, pool=self.cfg.pool)
        top = sorted(range(1, len(finals)), key=lambda i: -screen[i])[:6]
        keep = [0] + top
        full = self.scorer.score([finals[i] for i in keep], env.core.turn, self.cfg.opponents, _r.Random(self.rng.getrandbits(62)), prev=prev, pool=self.cfg.pool)
        scores = {i: v for i, v in zip(keep, full)}
        best = max(keep, key=lambda i: (scores[i], i == 0))
        self.stats['positions'] += 1
        if best == 0 or scores[best] - scores[0] < self.cfg.position_margin: return None
        self.stats['position_gain'] += scores[best] - scores[0]
        d = self.last_decision.get(env.core.turn)
        if d is not None: d.update(chosen=finals[best], predChosen=scores[best], override=True, positioned=True)
        return plans[best]

    def _execute(self, env, side, memory, obs, raw, plan):
        """Next action of `plan` (positioning first if it is END); falls back to raw if illegal."""
        joint = plan['actions'][plan['at']]
        if (self.cfg.position and joint.primitive.action_type == self.END and self.positioned_turn != env.core.turn):
            self.positioned_turn = env.core.turn
            moves = self._position(env, side, joint)
            if moves is not None:
                plan = dict(actions=moves, marks=[None] * len(moves), at=0, value=plan.get('value')); joint = moves[0]
        t, s, u = joint.primitive.as_tuple()
        if not (obs['type_mask'][t] and obs['source_mask'][t, s] and obs['target_mask'][t, s, u]):
            self.plan = None; self.stats['replans'] += 1
            return raw
        plan['at'] += 1; self.plan = plan
        if joint.primitive.action_type == self.END: self.plan = None
        return self.actor.act_many([self._request(obs, memory, joint)])[0]

    def _decide_rolls(self, env, side, memory, obs, raw):
        """Planner owns rolls: at each plan start and before ending with gold left, roll iff its expectimax value
        beats the best plan now by roll_margin."""
        roll_legal = bool(obs['type_mask'][self.ROLL])
        plan = self.plan
        if plan is not None and plan['at'] > 0 and plan['marks'][plan['at'] - 1] is not None and plan['marks'][plan['at'] - 1] != self._fingerprint(env.core, side):
            self.stats['replans'] += 1; plan = None
        fresh = plan is None or plan['at'] >= len(plan['actions'])
        if fresh:
            plan = self._new_plan(env, side, memory)
            if plan is None:
                self.plan = None; return raw
        at_end = plan['actions'][plan['at']].primitive.action_type == self.END
        gold = self._player(env.core, side).gold
        if roll_legal and gold >= 1 and (fresh or at_end) and plan.get('value') is not None:
            v_roll = self._roll_value(env, side, memory)
            if v_roll > plan['value'] + self.cfg.roll_margin:
                self.stats['roll_chosen'] += 1; self.plan = None
                return self.actor.act_many([self._request(obs, memory, self._roll_joint())])[0]
            if raw.action.primitive.action_type == self.ROLL: self.stats['roll_vetoed'] += 1
        return self._execute(env, side, memory, obs, raw, plan)

    # ---------- one real decision for the planner's seat
    def decide(self, env, side, memory):
        """Returns the actor response (action + next memory) for this decision."""
        self.cur_memory = memory
        obs = env.encoder.encode(env.core, side)
        raw = self.actor.act_many([self._request(obs, memory)])[0]
        if self.cfg.roll_search:
            return self._decide_rolls(env, side, memory, obs, raw)
        if self.cfg.position_only:
            return self._decide_position_only(env, side, memory, obs, raw)
        if self.cfg.enumerate:
            return self._decide_enum(env, side, memory, obs, raw)
        if raw.action.primitive.action_type == self.ROLL:
            # The policy keeps all its roll decisions (suppressing them mid-plan cost far more than it saved).
            self.plan = None; self.stats['rolls'] += 1
            return raw  # re-plan on the new shop
        plan = self.plan
        if plan is not None and plan['at'] > 0 and plan['marks'][plan['at'] - 1] is not None and plan['marks'][plan['at'] - 1] != self._fingerprint(env.core, side):
            self.stats['replans'] += 1; plan = None  # reality diverged (e.g. a random shop ability)
        if plan is None or plan['at'] >= len(plan['actions']):
            plan = self._new_plan(env, side, memory)
            if plan is None:
                self.plan = None; return raw
        joint = plan['actions'][plan['at']]
        if (self.cfg.position and joint.primitive.action_type == self.END and self.positioned_turn != env.core.turn):
            self.positioned_turn = env.core.turn
            moves = self._position(env, side, joint)
            if moves is not None:
                plan = dict(actions=moves, marks=[None] * len(moves), at=0); joint = moves[0]
        t, s, u = joint.primitive.as_tuple()
        if not (obs['type_mask'][t] and obs['source_mask'][t, s] and obs['target_mask'][t, s, u]):
            self.plan = None; self.stats['replans'] += 1
            return raw
        plan['at'] += 1; self.plan = plan
        if joint.primitive.action_type == self.END: self.plan = None
        return self.actor.act_many([self._request(obs, memory, joint)])[0]
