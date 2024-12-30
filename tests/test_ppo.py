import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.ai import baselines
from sapai import Team, Player
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


# train_maskableppo_superautopets.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

class AddToOpponentPoolCallback(BaseCallback):
    """
    Callback for periodically adding a copy of the agent to the opponent pool.
    Uses MaskablePPO models.
    """
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super(AddToOpponentPoolCallback, self).__init__(verbose)
        self.save_freq = save_freq  # Frequency in timesteps to save and add to pool
        self.save_path = save_path  # Directory to save temporary models

        # Create the directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if it's time to save the model
        if self.num_timesteps % self.save_freq == 0:
            # Define the model path
            model_path = os.path.join(self.save_path, f"temp_model_{self.num_timesteps}.zip")
            
            # Save the current MaskablePPO model
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved temporary MaskablePPO model to {model_path}")

            # Load the saved model as a MaskablePPO opponent
            opponent_model = MaskablePPO.load(model_path, env=self.model.get_env())

            # Add the opponent to the environment's opponent pool
            env = self.training_env.envs[0]  # Assuming single environment
            env.add_opponent(opponent_model)

            if self.verbose > 0:
                print(f"Added MaskablePPO opponent from {model_path} to the opponent pool.")

            # Optionally, remove the temporary model file to save space
            os.remove(model_path)
            if self.verbose > 0:
                print(f"Removed temporary model file {model_path}.")

        return True  # Continue training


def make_env():
    return SuperAutoPetsEnv(valid_actions_only=True, manual_battles=False)

def main():
    # Create the vectorized environment
    # env = DummyVecEnv([make_env])
    env = SuperAutoPetsEnv(valid_actions_only=False, manual_battles=False)
    obs, info = env.reset()
    action_mask = info.get('action_mask', None)
    print("action mask: ", action_mask.shape)
    check_env(env, warn=True)

    # Initialize MaskablePPO
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./maskableppo_superautopets_tensorboard/"
    )

    # Define the callback
    save_frequency = 10000  # Save and add to pool every 10,000 timesteps
    save_directory = "./temp_opponents/"
    add_opponent_callback = AddToOpponentPoolCallback(
        save_freq=save_frequency, 
        save_path=save_directory, 
        verbose=1
    )

    # Train the agent
    model.learn(
        total_timesteps=100000,  # Set to desired number of timesteps
        log_interval=10,
        tb_log_name="MaskablePPO_SuperAutoPets",
        callback=add_opponent_callback
    )

    # Save the model
    model.save("maskableppo_superautopets")

    # Close the training environment
    env.close()

    # Evaluation
    eval_env = make_env()
    obs, info = eval_env.reset()
    num_episodes = 5
    for episode in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action_mask = info.get('action_mask', None)
            if action_mask is None:
                raise ValueError("Action mask not found in info dict.")

            # Predict the action
            action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)

            # Take the action
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward

            # Render the environment
            eval_env.render()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Close the evaluation environment
    eval_env.close()

if __name__ == "__main__":
    main()
# class AddToOpponentPoolCallback(BaseCallback):
#     """
#     Callback for periodically adding a copy of the agent to the opponent pool.
#     Uses MaskablePPO models.
#     """
#     def __init__(self, save_freq: int, save_path: str, verbose=0):
#         super(AddToOpponentPoolCallback, self).__init__(verbose)
#         self.save_freq = save_freq  # Frequency in timesteps to save and add to pool
#         self.save_path = save_path  # Directory to save temporary models

#         # Create the directory if it doesn't exist
#         os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         # Check if it's time to save the model
#         if self.num_timesteps % self.save_freq == 0:
#             # Define the model path
#             model_path = os.path.join(self.save_path, f"temp_model_{self.num_timesteps}.zip")
            
#             # Save the current MaskablePPO model
#             self.model.save(model_path)
#             if self.verbose > 0:
#                 print(f"Saved temporary MaskablePPO model to {model_path}")

#             # Load the saved model as a MaskablePPO opponent
#             opponent_model = MaskablePPO.load(model_path, env=self.model.get_env())

#             # Add the opponent to the environment's opponent pool
#             env = self.training_env.envs[0]  # Access the first environment in the vectorized env
#             env.add_opponent(opponent_model)

#             if self.verbose > 0:
#                 print(f"Added MaskablePPO opponent from {model_path} to the opponent pool.")

#             # Optionally, remove the temporary model file to save space
#             os.remove(model_path)
#             if self.verbose > 0:
#                 print(f"Removed temporary model file {model_path}.")

#         return True  # Continue training

# def make_env():
#     return SuperAutoPetsEnv(valid_actions_only=True, manual_battles=False)

# def main():
#     # Create the vectorized environment
#     env = DummyVecEnv([make_env])  # Using DummyVecEnv ensures self.training_env.envs[0] is defined
    
#     obs, info = env.reset()
#     action_mask = info.get('action_mask', None)
#     print("action mask: ", action_mask.shape)
    
#     # Check environment compliance
#     from stable_baselines3.common.env_checker import check_env
#     check_env(env, warn=True)

#     # Initialize MaskablePPO for the agent
#     model = MaskablePPO(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         tensorboard_log="./maskableppo_superautopets_tensorboard/"
#     )

#     # Define the callback
#     save_frequency = 10000  # Save and add to pool every 10,000 timesteps
#     save_directory = "./temp_opponents/"
#     add_opponent_callback = AddToOpponentPoolCallback(
#         save_freq=save_frequency, 
#         save_path=save_directory, 
#         verbose=1
#     )

#     # Pre-populate the opponent pool with the initial agent model
#     initial_model_path = os.path.join(save_directory, "initial_model.zip")
#     model.save(initial_model_path)
#     opponent_model = MaskablePPO.load(initial_model_path, env=env)
#     env.envs[0].add_opponent(opponent_model)  # Access the first environment in DummyVecEnv
#     os.remove(initial_model_path)
#     print("Pre-populated opponent pool with the initial MaskablePPO agent model.")

#     # Train the agent with the callback
#     model.learn(
#         total_timesteps=100000,  # Adjust as needed
#         log_interval=10,
#         tb_log_name="MaskablePPO_SuperAutoPets",
#         callback=add_opponent_callback
#     )

#     # Save the final agent model
#     model.save("maskableppo_superautopets")

#     # Close the training environment
#     env.close()

# if __name__ == "__main__":
#     main()