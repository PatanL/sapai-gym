"""CPU micro-benchmark: how fast does the frozen engine resolve battles and shop steps?"""
import sys, time, random, json
sys.path[:0] = ['/workspace/runs/search-throughput-5x-20260912-v1/reference-source/sapai',
                '/workspace/runs/search-throughput-5x-20260912-v1/reference-source/sapai-gym']
from sapai import Battle, Team
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec
from sapai_gym1.rl.joint_action import JointAction

rng = random.Random(0)
boards = {}  # turn -> list of team states
t0 = time.time(); steps = 0
for g in range(12):
    env = ParallelSelfPlayEnv(); env.reset(seed=1000 + g)
    done = False
    while not done and env.core.turn <= 12:
        acts = []
        for p, active in enumerate(env.active_players()):
            if not active: acts.append(None); continue
            legal = sorted(env.core._avail_actions(p))
            # mostly buy/shop actions, end turn sometimes
            a = rng.choice(legal)
            acts.append(JointAction(FlatActionCodec.decode(a)))
        from sapai_gym1.rl.joint_action import step_decisions, JointAction
        r = step_decisions(env, acts).result
        steps += 1
        if r.battle_resolved:
            boards.setdefault(env.core.turn - 1, []).append(Team.from_state(env.core.previous_agent_state.state))
            boards.setdefault(env.core.turn - 1, []).append(Team.from_state(env.core.previous_opponent_state.state))
        done = r.terminated
shop_s = time.time() - t0
print(json.dumps(dict(games=12, steps=steps, shop_seconds=round(shop_s, 2), steps_per_s=round(steps / shop_s, 1),
                      boards_per_turn={k: len(v) for k, v in sorted(boards.items())})))
# battles between collected boards of the same turn
for turn in sorted(boards):
    teams = boards[turn]
    if len(teams) < 4: continue
    n = 0; t1 = time.time()
    for i in range(200):
        a, b = rng.sample(teams, 2)
        Battle(Team.from_state(a.state), Team.from_state(b.state), seed=i, fast_training=True).battle()
        n += 1
    dt = time.time() - t1
    print(json.dumps(dict(turn=turn, battles=n, seconds=round(dt, 3), battles_per_s=round(n / dt, 1),
                          pets=[len([p for p in t if not p.empty]) for t in teams[:6]])))
