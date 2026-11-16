from plan_common import install
install(cuda=False)
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
env = ParallelSelfPlayEnv(); env.reset(seed=5, paired_tape=True)
o = env.observations()[0]
tot = 0
for k, v in o.items():
    tot += v.nbytes; print(k, v.shape, v.dtype, v.nbytes)
print('total bytes', tot)
