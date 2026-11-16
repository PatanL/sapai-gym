"""Planner-wrapped checkpoint vs the raw greedy checkpoint: paired seeds, both seats, terminal match scores.

Mirrors public_match_evaluation.play (paired tape worlds, greedy FP32 actor, winner by _completed_match_result)
but plays one game at a time so the planner can use the battle pool. Results stream to a JSONL file and the
run resumes from it. --mode raw replaces the planner with the plain policy (harness control: exactly 0.5).
"""
import argparse, json, time
from pathlib import Path
from plan_common import install, load_actor, OUT, seed_of, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['planner', 'raw'], default='planner')
    ap.add_argument('--pairs', type=int, default=25)
    ap.add_argument('--first', type=int, default=0, help='index of the first seed pair (split a panel across jobs)')
    ap.add_argument('--namespace', default='claude-eval-v1')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--candidates', type=int, default=16)
    ap.add_argument('--opponents', type=int, default=64)
    ap.add_argument('--margin', type=float, default=0.03)
    ap.add_argument('--condition', action='store_true')
    ap.add_argument('--position', action='store_true')
    ap.add_argument('--roll-search', action='store_true')
    ap.add_argument('--enumerate', action='store_true')
    ap.add_argument('--predictor', default=None, help='battle_net.pt for GPU screening')
    ap.add_argument('--enum-nodes', type=int, default=400)
    ap.add_argument('--enum-top', type=int, default=8)
    ap.add_argument('--enum-margin', type=float, default=0.06)
    ap.add_argument('--raw-samples', type=int, default=8)
    ap.add_argument('--lookahead', action='store_true')
    ap.add_argument('--position-only', action='store_true')
    ap.add_argument('--oracle', action='store_true', help='DIAGNOSTIC: privileged opponent final board')
    ap.add_argument('--prev-opponent', action='store_true', help="position against the opponent's last revealed board")
    ap.add_argument('--sim-opponent', action='store_true', help='position against simulated opponent turns (public)')
    ap.add_argument('--sim-samples', type=int, default=8)
    ap.add_argument('--diag-oracle', action='store_true')
    ap.add_argument('--sim-memory', default='zeros', choices=['zeros', 'self', 'true'])
    ap.add_argument('--sim-keep-frozen', action='store_true', help='DIAGNOSTIC: privileged frozen items')
    ap.add_argument('--screen-seeds', type=int, default=1)
    ap.add_argument('--exact-top', type=int, default=6)
    ap.add_argument('--sim-seeds', type=int, default=4)
    ap.add_argument('--position-margin', type=float, default=0.02)
    ap.add_argument('--mix-prev', action='store_true')
    ap.add_argument('--prev-as-sim', action='store_true', help='sim positioning against the last revealed board only (no simulated turn)')
    ap.add_argument('--label-out', action='store_true', help='record public order labels at every END (see planner._label_decision)')
    ap.add_argument('--label-seeds', type=int, default=4)
    ap.add_argument('--posnet', default=None, help='learned END positioner checkpoint in out/')
    ap.add_argument('--posnet-margin', type=float, default=0.0)
    ap.add_argument('--bp-position', action='store_true', help='positioning by the battle predictor only (fast labels)')
    ap.add_argument('--look-samples', type=int, default=3)
    ap.add_argument('--pure-policy-plan', action='store_true', help='planner only positions; keeps the greedy plan')
    ap.add_argument('--opponent', default=None, help='path:sha of a different checkpoint for the raw side')
    ap.add_argument('--candidate', default=None, help='path:sha of the candidate checkpoint (default: current best)')
    ap.add_argument('--both-seats', action='store_true', help='data generation: the planner plays both seats')
    ap.add_argument('--record', default=None, help='directory: save every planner-seat decision (obs, memory, action)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    install(cuda=False)  # engine on the path for the battle workers (forked before CUDA starts)
    from planner import BattleScorer, PlannerConfig, TurnPlanner
    scorer = BattleScorer(OUT / 'library.pkl', a.workers) if a.mode == 'planner' else None
    install(cuda=True)
    import torch
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    from sapai_gym1.rl.joint_action import step_decisions
    from sapai_gym1.rl.distributed import _completed_match_result
    from teacher_world import ActorRequest, observation_copy, owned_memory
    actor = load_actor() if not a.candidate else load_actor(dict(zip(('path', 'sha'), a.candidate.split(':'))))
    opponent = actor if not a.candidate else load_actor()
    if a.opponent:
        path, sha = a.opponent.split(':'); opponent = load_actor(dict(path=path, sha=sha))
    cfg = PlannerConfig(candidates=a.candidates, opponents=a.opponents, margin=a.margin, condition=a.condition, position=a.position, roll_search=a.roll_search, enumerate=a.enumerate)
    if a.pure_policy_plan: cfg.margin = 9.0  # never switch plans: isolates the positioning effect
    cfg.enum_nodes, cfg.enum_top, cfg.enum_margin, cfg.raw_samples = a.enum_nodes, a.enum_top, a.enum_margin, a.raw_samples
    cfg.lookahead, cfg.look_samples = a.lookahead, a.look_samples
    cfg.position_only = a.position_only; cfg.oracle = a.oracle; cfg.prev_opponent = a.prev_opponent
    cfg.sim_opponent = a.sim_opponent; cfg.sim_samples = a.sim_samples; cfg.diag_oracle = a.diag_oracle; cfg.sim_memory = a.sim_memory; cfg.sim_keep_frozen = a.sim_keep_frozen
    cfg.screen_seeds, cfg.exact_top, cfg.sim_seeds, cfg.position_margin, cfg.mix_prev = a.screen_seeds, a.exact_top, a.sim_seeds, a.position_margin, a.mix_prev
    cfg.bp_position = a.bp_position; cfg.prev_as_sim = a.prev_as_sim
    cfg.label_out, cfg.label_seeds, cfg.posnet_margin = a.label_out, a.label_seeds, a.posnet_margin
    if a.posnet:
        from posnet import PosNetScorer
        cfg.posnet = PosNetScorer(OUT / a.posnet)
    if a.predictor:
        from planner import PredictorScorer
        cfg.predictor = PredictorScorer(scorer, OUT / a.predictor)

    path = OUT / a.out
    done = {(r['seed'], r['plannerSide']) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    specs = [(seed_of(a.namespace, i), side) for i in range(a.first, a.first + a.pairs) for side in (0, 1)]
    began = time.time()
    for n, (seed, pside) in enumerate(specs):
        if (seed, pside) in done: continue
        t0 = time.time()
        env = ParallelSelfPlayEnv(); env.reset(seed=seed, paired_tape=True)
        mem = [torch.zeros((actor if s == pside else opponent).model.config.memory_dim) for s in (0, 1)]
        planner = TurnPlanner(actor, scorer, cfg, seed=seed_of('claude-planner-rng', seed, pside)) if a.mode == 'planner' else None
        planner2 = TurnPlanner(actor, scorer, cfg, seed=seed_of('claude-planner-rng2', seed, pside)) if (a.mode == 'planner' and a.both_seats) else None
        records = []
        steps = 0
        while True:
            obs = env.observations(); resp = {}
            for side, active in enumerate(env.active_players()):
                if not active: continue
                pl = planner if side == pside else planner2
                if pl is not None:
                    pl.opp_memory = mem[1 - side]  # only read by the privileged --oracle diagnostic
                    before = pl.stats.get('t_decide', 0.)
                    _t = time.time(); resp[side] = pl.decide(env, side, mem[side])
                    pl.stats['t_decide'] = round(before + time.time() - _t, 2)
                    if a.record:
                        raw_greedy = actor.act_many([ActorRequest(observation_copy(obs[side]), owned_memory(mem[side]), actor.owner)])[0].action
                        records.append(dict(obs={k: v.copy() for k, v in obs[side].items()}, memory=mem[side].clone(),
                                            action=resp[side].action.to_dict(), raw=raw_greedy.to_dict(),
                                            forced=resp[side].action.to_dict() != raw_greedy.to_dict(), turn=env.core.turn))
                else:
                    pol = actor if side == pside else opponent  # raw mode: candidate seat uses `actor`
                    resp[side] = pol.act_many([ActorRequest(observation_copy(obs[side]), owned_memory(mem[side]), pol.owner)])[0]
            turn = env.core.turn
            step = step_decisions(env, [resp[s].action if s in resp else None for s in (0, 1)]).result
            if step.battle_resolved and planner is not None and turn in planner.last_decision:
                d = planner.last_decision.pop(turn)
                mine = (env.core.previous_agent_state if pside == 0 else env.core.previous_opponent_state).state
                theirs = (env.core.previous_opponent_state if pside == 0 else env.core.previous_agent_state).state
                real = scorer.versus([d['greedy'], d['chosen'], mine], theirs, 64, planner.rng)
                with (OUT / (Path(a.out).stem + '-turns.jsonl')).open('a') as f:
                    f.write(json.dumps(dict(seed=seed, plannerSide=pside, turn=turn, override=d['override'],
                        predGreedy=d['predGreedy'], predChosen=d['predChosen'], realGreedy=real[0], realChosen=real[1],
                        realActual=real[2], battle=step.battle_result)) + '\n')
            for s, r in resp.items(): mem[s] = owned_memory(r.next_memory)
            steps += 1
            if steps > 2500: raise RuntimeError('truncated game')
            if step.terminated: break
        if a.record and records:
            Path(a.record).mkdir(parents=True, exist_ok=True)
            torch.save(records, Path(a.record) / f'game-{seed}-{pside}.pt')
        winner = _completed_match_result(env.core)
        score = .5 if winner == 2 else float(winner == pside)
        p = env.core.agent if pside == 0 else env.core.opponent
        q = env.core.opponent if pside == 0 else env.core.agent
        row = dict(seed=seed, plannerSide=pside, score=score, turn=env.core.turn, steps=steps, seconds=round(time.time() - t0, 1),
                   plannerLives=p.lives, plannerWins=p.wins, rawLives=q.lives, rawWins=q.wins,
                   planner=planner.stats if planner else None, diag=getattr(planner, 'diag', None) if planner else None,
                   labels=(getattr(planner, 'labels', []) + getattr(planner2, 'labels', [])) or None)
        with path.open('a') as f: f.write(json.dumps(row) + '\n')
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        mean = sum(r['score'] for r in rows) / len(rows)
        print(f'{len(rows)}/{len(specs)} games  score {score}  running mean {mean:.3f}  turn {row["turn"]}  {row["seconds"]}s  '
              f'{json.dumps(row["planner"]) if row["planner"] else ""}  elapsed {time.time() - began:.0f}s', flush=True)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    by = {}
    for r in rows: by.setdefault(r['seed'], []).append(r['score'])
    paired = [sum(v) / 2 for v in by.values() if len(v) == 2]
    import numpy as np
    vals = np.asarray(paired); rng = np.random.default_rng(0)
    boot = vals[rng.integers(0, len(vals), size=(20000, len(vals)))].mean(axis=1) if len(vals) else np.zeros(1)
    save(OUT / (Path(a.out).stem + '-SUMMARY.json'), dict(mode=a.mode, pairs=len(paired), games=len(rows),
         meanMatchScore=float(vals.mean()) if len(vals) else None, paired95=np.quantile(boot, [.025, .975]).tolist(),
         config=vars(a), checkpointSHA256=actor.owner))


if __name__ == '__main__':
    main()
