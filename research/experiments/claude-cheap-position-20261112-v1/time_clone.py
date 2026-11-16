import copy, time, random, pickle
from plan_common import install
install(cuda=False)
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec
from sapai_gym1.rl.joint_action import JointAction, step_decisions
env = ParallelSelfPlayEnv(); env.reset(seed=123, paired_tape=True)
rng = random.Random(0)
while env.core.turn < 6:  # random play to a mid-game state
    acts = []
    for p, active in enumerate(env.active_players()):
        if not active: acts.append(None); continue
        legal = sorted(env.core._avail_actions(p)); acts.append(JointAction(FlatActionCodec.decode(rng.choice(legal)), 0))
    step_decisions(env, acts)
for name, fn in [('env.clone', lambda: env.clone()), ('deepcopy(core)', lambda: copy.deepcopy(env.core)),
                 ('deepcopy(player)', lambda: copy.deepcopy(env.core.agent)), ('pickle(core)', lambda: pickle.loads(pickle.dumps(env.core)))]:
    t = time.time(); n = 0
    while time.time() - t < 2: fn(); n += 1
    print(f'{name}: {1000 * (time.time() - t) / n:.2f} ms')
acts = env.core._avail_actions(0)
print('legal actions now', len(acts), 'sample', list(acts.items())[:3])
import copy as _c
opp = env.core.opponent
def light():
    memo = {id(opp): opp}
    for k in ('previous_opponent_state', 'previous_agent_state', 'previous_previous_opponent_state', 'previous_previous_agent_state'):
        v = getattr(env.core, k, None)
        if v is not None: memo[id(v)] = v
    return _c.deepcopy(env.core, memo)
t = time.time(); n = 0
while time.time() - t < 2: light(); n += 1
print(f'light deepcopy(core, share opponent): {1000 * (time.time() - t) / n:.2f} ms')
big = sorted(((len(pickle.dumps(v)), k) for k, v in vars(env.core).items() if not callable(v)), reverse=True)[:8]
print('largest core fields', big)
