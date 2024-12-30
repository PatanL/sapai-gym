import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.ai import baselines
from sapai import Team, Player
from gymnasium.utils.env_checker import check_env
import gymnasium as gym




def _do_store_phase(env: SuperAutoPetsEnv, ai):
    env.player.start_turn()

    while True:
        actions = env._avail_actions()
        # print("actions: ", actions.keys())
        chosen_action = ai(env.player, actions)
        if isinstance(chosen_action, tuple):
            chosen_action = 0
        # print("chosen_action: ", chosen_action)
        env.resolve_action(chosen_action)

        if SuperAutoPetsEnv._get_action_name(actions[chosen_action]) == "end_turn":
            return

def opp_generator(num_turns, ai):
    opps = list()
    env = SuperAutoPetsEnv(None, valid_actions_only=True, manual_battles=True)
    while env.player.turn <= num_turns:
        _do_store_phase(env, ai)
        opps.append(Team.from_state(env.player.team.state))
    return opps

def biggest_numbers_horizontal_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.biggest_numbers_horizontal_scaling_agent)

def opponent_generator(num_turns):
    # Returns teams to fight against in the gym
    opponents = biggest_numbers_horizontal_opp_generator(25)
    return opponents

print(biggest_numbers_horizontal_opp_generator(25))
print(biggest_numbers_horizontal_opp_generator(25))

env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=False)
print("Observation space:", env.observation_space)
print("Observation space shape:", env.observation_space.shape)
print("Observation space dtype:", env.observation_space.dtype)

obs, _ = env.reset()
print("Reset observation shape:", obs.shape)
print("Reset observation dtype:", obs.dtype)
print("Is in observation space:", env.observation_space.contains(obs))

check_env(env)


for step_num in range(1000):
    if step_num % 100 == 0:
        print(f"Step {step_num}")

    # Random actions
    action = env.action_space.sample() # make this learnable by the agent
    obs, reward, terminated, truncated, info = env.step(action)


    if terminated or truncated:
        obs, info = env.reset()

env.close()
