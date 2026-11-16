"""Headroom: how much better than the policy's own plan is the best plan reachable with the current shop?

From real greedy self-play states (start of a shop turn, turns 3-12), enumerate every distinct end-of-turn board
reachable without rolling (buy pet / buy food / sell / combine; one placement per empty slot; no reorders),
screen them all against library opponents, then compare the top few and the policy's greedy no-roll plan on a
FRESH opponent/seed set (so selection luck does not count as gain). Public clones only (fresh shop tapes).
"""
import json, pickle, random, sys, time
from plan_common import install, load_actor, OUT, seed_of

STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def signature(core, side):
    from planner import strip_rng
    p = core.agent if side == 0 else core.opponent
    team = [(s.pet.name, s.pet.level, s.pet.experience, s.pet.attack, s.pet.health, s.pet.status) for s in p.team if not s.empty]
    shop = [(str(s.obj.name) if s.obj is not None else '', s.frozen) for s in p.shop.slots] if hasattr(p.shop, 'slots') else []
    return json.dumps([team, p.gold, shop], default=str)


def _enumerate_task(args):
    """DFS from one first action on a pickled public core; returns {signature: (final board state, path)}."""
    core_bytes, side, first, max_depth, max_nodes = args
    from sapai_gym1.rl.action_schema import FlatActionCodec, ActionType, AutoregressiveAction
    from sapai_gym1.rl.end_freeze import expand_end_freeze
    import copy
    allowed = {ActionType.BUY_PET, ActionType.BUY_FOOD, ActionType.SELL, ActionType.COMBINE}

    def moves(core):
        p = core.agent if side == 0 else core.opponent
        out, seen_empty = [], set()
        for code in sorted(core._avail_actions(side)):
            a = FlatActionCodec.decode(code)
            if a.action_type not in allowed: continue
            if a.action_type == ActionType.BUY_PET and p.team[a.target].empty:
                if a.source in seen_empty: continue  # one placement per empty slot is enough (order searched later)
                seen_empty.add(a.source)
            out.append(code)
        return out

    def end_board(core):
        c = copy.deepcopy(core)
        try:
            for prim in expand_end_freeze(c, side, 0): c.resolve_action(agent_idx=side, action=FlatActionCodec.encode(prim))
        except ValueError:
            return None  # END not legal here (e.g. empty team)
        return (c.agent if side == 0 else c.opponent).team.state

    root = pickle.loads(core_bytes)
    results, visited, nodes = {}, set(), 0
    stack = [(root, [])] if first is None else []
    if first is not None:
        c = copy.deepcopy(root); c.resolve_action(agent_idx=side, action=first); stack.append((c, [first]))
    while stack and nodes < max_nodes:
        core, path = stack.pop(); nodes += 1
        sig = signature(core, side)
        if sig in visited: continue
        visited.add(sig)
        board = end_board(core)
        if board is not None: results.setdefault(sig, (board, path))
        if len(path) >= max_depth: continue
        for code in moves(core):
            c = copy.deepcopy(core); c.resolve_action(agent_idx=side, action=code); stack.append((c, path + [code]))
    return results


def main():
    install(cuda=False)
    import planner as P
    scorer = P.BattleScorer(OUT / 'library.pkl', WORKERS)
    install(cuda=True)
    import torch
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    from sapai_gym1.rl.joint_action import step_decisions
    from teacher_world import ActorRequest, observation_copy, owned_memory
    actor = load_actor()
    tp = P.TurnPlanner(actor, scorer, P.PlannerConfig(candidates=1), seed=11)
    rng = random.Random(5); rows = []; began = time.time(); g = 0
    while len(rows) < STATES:
        env = ParallelSelfPlayEnv(); env.reset(seed=seed_of('claude-headroom-v1', g), paired_tape=True); g += 1
        mem = [torch.zeros(actor.model.config.memory_dim) for _ in (0, 1)]
        targets = set(rng.sample(range(3, 13), 3)); new_turn = True
        while True:
            turn = env.core.turn
            if new_turn and turn in targets and len(rows) < STATES:
                t0 = time.time()
                # greedy no-roll plan (the policy's own plan for this shop), on a public clone
                plans, _, finals, done = tp._rollouts(env, 0, mem[0], k=1)
                greedy = finals[0] if done[0] else None
                # enumerate all reachable no-roll boards, split by first action across the pool
                clone = tp._public_clone(env, 0); core_bytes = pickle.dumps(clone.core)
                firsts = [None] + [c for c in sorted(clone.core._avail_actions(0))]
                tasks = [(core_bytes, 0, f, 7, 400) for f in firsts]
                merged = {}
                for part in scorer.pool.map(_enumerate_task, tasks):
                    for k, v in part.items(): merged.setdefault(k, v)
                boards = [v[0] for v in merged.values()]
                if greedy is not None and len(boards) >= 2:
                    screen = scorer.score(boards, turn, 24, random.Random(rng.getrandbits(62)))
                    top = sorted(range(len(boards)), key=lambda i: -screen[i])[:5]
                    fresh_rng = random.Random(rng.getrandbits(62))
                    fresh = scorer.score([greedy] + [boards[i] for i in top], turn, 128, fresh_rng)
                    rows.append(dict(turn=turn, boards=len(boards), screenBest=screen[top[0]], freshGreedy=fresh[0],
                                     freshTop1=fresh[1], freshBestOf5=max(fresh[1:]), seconds=round(time.time() - t0, 1)))
                    print(json.dumps(rows[-1]), flush=True)
            # continue the real game greedily
            obs = env.observations(); resp = {}
            for side, active in enumerate(env.active_players()):
                if active: resp[side] = actor.act_many([ActorRequest(observation_copy(obs[side]), owned_memory(mem[side]), actor.owner)])[0]
            step = step_decisions(env, [resp[s].action if s in resp else None for s in (0, 1)]).result
            for s, r in resp.items(): mem[s] = owned_memory(r.next_memory)
            new_turn = step.battle_resolved
            if step.terminated: break
    m = lambda k: sum(r[k] for r in rows) / len(rows)
    summary = dict(states=len(rows), meanBoards=m('boards'), freshGreedy=m('freshGreedy'), freshTop1=m('freshTop1'),
                   freshBestOf5=m('freshBestOf5'), gainTop1=m('freshTop1') - m('freshGreedy'), seconds=time.time() - began)
    print(json.dumps(summary)); json.dump(dict(summary=summary, rows=rows), open(OUT / 'headroom.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
