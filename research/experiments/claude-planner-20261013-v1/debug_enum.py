import time, random, pickle, multiprocessing as mp
from plan_common import install
install(cuda=False)
import planner as P
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec
from sapai_gym1.rl.joint_action import JointAction, step_decisions
env = ParallelSelfPlayEnv(); env.reset(seed=77, paired_tape=True); rng = random.Random(1)
while env.core.turn < 5:
    acts = []
    for p, active in enumerate(env.active_players()):
        if not active: acts.append(None); continue
        legal = sorted(env.core._avail_actions(p)); acts.append(JointAction(FlatActionCodec.decode(rng.choice(legal)), 0))
    step_decisions(env, acts)
core = env.core
print('gold', core.agent.gold, 'team', [s.pet.name for s in core.agent.team if not s.empty])
sig, board, children = P._expand_task((pickle.dumps(core), 0, (), True))
print('root board None?', board is None, 'children', len(children), 'distinct child sigs', len({c for c, _ in children}))
pool = mp.get_context('fork').Pool(3)
t = time.time(); items = P.enumerate_boards(pool, core, 0, 7, 600); print('bfs boards', len(items), 'seconds %.1f' % (time.time() - t))
t = time.time(); merged = {}
cb = pickle.dumps(core)
for part in pool.map(P._enumerate_task, [(cb, 0, f, 7, 400) for f in [None] + sorted(core._avail_actions(0))]):
    for k, v in part.items(): merged.setdefault(k, v)
print('dfs boards', len(merged), 'seconds %.1f' % (time.time() - t))
from sapai_gym1.rl.end_freeze import expand_end_freeze
import traceback
c0 = pickle.loads(pickle.dumps(core))
c = P.light_copy(c0, 0)
try:
    plan = expand_end_freeze(c, 0, 0); print('plan', plan)
    for prim in plan: c.resolve_action(agent_idx=0, action=FlatActionCodec.encode(prim))
    print('END ok')
except Exception as e:
    traceback.print_exc()
c1 = P.light_copy(core, 0)
try:
    plan = expand_end_freeze(c1, 0, 0); print('direct plan', plan)
except Exception as e:
    print('direct error', repr(e))
