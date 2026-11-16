"""Opponent-board library: boards the current checkpoint actually builds, per turn, in greedy self-play.

The planner's battles are simulated against these instead of the (hidden) real opponent board. Seeds come
from a namespace disjoint from every evaluation seed. Each record keeps the player's previous board too, so
the planner can later condition on the opponent's last revealed board.
"""
import pickle, sys, time
from collections import defaultdict
from plan_common import install, load_actor, OUT, seed_of, save

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 240


def main():
    install(cuda=True)
    import torch
    from sapai import Team
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    from sapai_gym1.rl.joint_action import step_decisions
    from teacher_world import ActorRequest, observation_copy, owned_memory
    actor = load_actor()
    library = defaultdict(list)  # turn -> [dict(board=state, prev=state|None, lives, wins)]
    began = time.time(); batch = 16
    for start in range(0, GAMES, batch):
        games = []
        for i in range(start, min(GAMES, start + batch)):
            env = ParallelSelfPlayEnv(); env.reset(seed=seed_of('claude-library-v1', i), paired_tape=True)
            games.append(dict(env=env, mem=[torch.zeros(actor.model.config.memory_dim) for _ in (0, 1)],
                              prev=[None, None], done=False, steps=0))
        while any(not g['done'] for g in games):
            reqs = []
            for g in games:
                if g['done']: continue
                obs = g['env'].observations()
                for side, active in enumerate(g['env'].active_players()):
                    if active: reqs.append((g, side, ActorRequest(observation_copy(obs[side]), owned_memory(g['mem'][side]), actor.owner)))
            responses = []
            for k in range(0, len(reqs), 64):
                responses += actor.act_many([r[2] for r in reqs[k:k + 64]])
            acts = {}
            for (g, side, _), resp in zip(reqs, responses):
                acts.setdefault(id(g), {})[side] = resp
            for g in games:
                if g['done']: continue
                r = acts[id(g)]
                core = g['env'].core; turn = core.turn
                step = step_decisions(g['env'], [r[s].action if s in r else None for s in (0, 1)]).result
                for s, resp in r.items(): g['mem'][s] = owned_memory(resp.next_memory)
                g['steps'] += 1
                if step.battle_resolved:
                    boards = (core.previous_agent_state.state, core.previous_opponent_state.state)
                    players = (core.agent, core.opponent)
                    for s in (0, 1):
                        library[turn].append(dict(board=boards[s], prev=g['prev'][s], lives=players[s].lives, wins=players[s].wins))
                        g['prev'][s] = boards[s]
                if step.terminated or g['steps'] > 2500: g['done'] = True
        print(f'{min(GAMES, start + batch)}/{GAMES} games, {sum(map(len, library.values()))} boards, {time.time() - began:.0f}s', flush=True)
    with (OUT / 'library.pkl').open('wb') as f:
        pickle.dump(dict(library), f)
    save(OUT / 'LIBRARY.json', dict(games=GAMES, seconds=time.time() - began, perTurn={t: len(v) for t, v in sorted(library.items())},
                                   checkpointSHA256=actor.owner, namespace='claude-library-v1'))


if __name__ == '__main__':
    main()
