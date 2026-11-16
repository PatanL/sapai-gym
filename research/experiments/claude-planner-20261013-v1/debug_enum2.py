import pickle, random
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
rb = pickle.dumps(env.core)
level = [()]; seen = set()
for depth in range(4):
    nxt = []; boards = 0
    for path in level[:50]:
        sig, board, children = P._expand_task((rb, 0, path, True))
        boards += board is not None
        for cs, code in children:
            if cs not in seen: seen.add(cs); nxt.append(path + (code,))
    print('depth', depth, 'states', len(level), 'boards', boards, 'next', len(nxt))
    if depth == 1:
        path = level[0]; c = pickle.loads(rb)
        for code in path: c._agent_actions_cache = None; c.resolve_action(agent_idx=0, action=code)
        print('  after', [FlatActionCodec.decode(x) for x in path], 'gold', c.agent.gold, 'team', [s.pet.name for s in c.agent.team if not s.empty])
    level = nxt
