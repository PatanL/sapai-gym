import time, random, pickle
from plan_common import install
install(cuda=False)
from planner import light_copy, board_signature
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec, ActionType
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.end_freeze import expand_end_freeze
env = ParallelSelfPlayEnv(); env.reset(seed=77, paired_tape=True); rng = random.Random(1)
while env.core.turn < 7:
    acts = []
    for p, active in enumerate(env.active_players()):
        if not active: acts.append(None); continue
        legal = sorted(env.core._avail_actions(p)); acts.append(JointAction(FlatActionCodec.decode(rng.choice(legal)), 0))
    step_decisions(env, acts)
core = env.core
def bench(name, fn, n=200):
    t = time.time()
    for _ in range(n): fn()
    print(f'{name}: {1000 * (time.time() - t) / n:.2f} ms')
bench('light_copy', lambda: light_copy(core, 0))
def avail():
    c = light_copy(core, 0); c._avail_actions(0)
bench('light_copy+_avail_actions', avail)
bench('board_signature', lambda: board_signature(core, 0))
def endb():
    c = light_copy(core, 0)
    for prim in expand_end_freeze(c, 0, 0): c.resolve_action(agent_idx=0, action=FlatActionCodec.encode(prim))
bench('light_copy+END', endb, 50)
c = light_copy(core, 0); acts = sorted(c._avail_actions(0))
types = {}
for a in acts: types[FlatActionCodec.decode(a).action_type.name] = types.get(FlatActionCodec.decode(a).action_type.name, 0) + 1
print('legal action types', types, 'gold', core.agent.gold)
