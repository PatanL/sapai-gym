import os
import torch
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
from stable_baselines3.common.callbacks import EvalCallback

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
#             env = self.training_env.envs[0]  # Assuming single environment
#             env.add_opponent(opponent_model)

#             if self.verbose > 0:
#                 print(f"Added MaskablePPO opponent from {model_path} to the opponent pool.")

#             # Optionally, remove the temporary model file to save space
#             os.remove(model_path)
#             if self.verbose > 0:
#                 print(f"Removed temporary model file {model_path}.")

#         return True  # Continue training

class AddToOpponentPoolCallback(BaseCallback):
    def __init__(self, save_freq: int, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # Create a new opponent model architecture. It's lightweight.
            # SB3 requires an env to initialize, so we get it from the model.
            opponent_model = MaskablePPO(
                policy=self.model.policy.__class__,
                env=self.model.get_env(),
                **self.model.get_attr("policy_kwargs"),  # mirror keyword args
                verbose=0
            )
            
            # Copy the weights from the current model to the new opponent model
            opponent_model.policy.load_state_dict(self.model.policy.state_dict())
            
            # Get the actual environment instance (assuming DummyVecEnv or a single env)
            # Add to every sub-env in the VecEnv
            # for env in self.training_env.envs:
            #     env.add_opponent(opponent_model)
            self.training_env.env_method("add_opponent", opponent_model)

            if self.verbose > 0:
                print(f"Cloned current policy and added to opponent pool at timestep {self.num_timesteps}.")
        
        return True

class RewardAnnealingCallback(BaseCallback):
    """
    A callback to anneal the scale of shaped rewards over the course of training.
    """
    def __init__(self, total_timesteps: int, annealing_end_fraction: float = 0.8, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        # Calculate the timestep at which annealing should finish
        self.annealing_end_step = int(total_timesteps * annealing_end_fraction)

    def _on_step(self) -> bool:
        # Calculate the current progress of annealing (from 0.0 to 1.0)
        progress = min(1.0, self.num_timesteps / self.annealing_end_step)
        
        # Linear decay from 1.0 down to 0.0
        new_scale = max(0.0, 1.0 - progress)
        
        # Update the shaping_scale in each environment
        self.training_env.env_method("set_shaping_scale", new_scale)

        # Log the scale to TensorBoard to monitor it
        self.logger.record("custom/reward_shaping_scale", new_scale)
        
        return True


def make_env():
    return SuperAutoPetsEnv(valid_actions_only=True, manual_battles=False)

def main():
    # Create the vectorized environment
    env = DummyVecEnv([make_env])
    # env = SuperAutoPetsEnv(valid_actions_only=False, manual_battles=False)
    # obs, info = env.reset()
    # action_mask = info.get('action_mask', None)
    # print("action mask: ", action_mask.shape)
    # check_env(env, warn=True)

    # Initialize MaskablePPO
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./maskableppo_superautopets_tensorboard/",
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    TOTAL_TIMESTEPS = 500_000   
    
     # ——— 1) set up EvalCallback ———
    # We’ll evaluate every 10k steps on 10 episodes, saving the best model   
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=10_000,         # run evaluation every 10k steps
        n_eval_episodes=10,        # average over 10 episodes
        deterministic=True,       # use deterministic actions at eval time
        render=False,
        verbose=1,
    )

    # callback for self-play
    save_frequency = 10000  # Save and add to pool every 10,000 timesteps
    save_directory = "./temp_opponents/"
    add_opponent_callback = AddToOpponentPoolCallback(
        save_freq=save_frequency, 
        verbose=1
    )

    # callback for annealing the shaped rewards
    reward_annealing_callback = RewardAnnealingCallback(total_timesteps=TOTAL_TIMESTEPS, annealing_end_fraction=0.8)

    # Train the agent
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,  # Set to desired number of timesteps
        log_interval=10,
        tb_log_name="MaskablePPO_SuperAutoPets",
        callback=[add_opponent_callback, eval_callback, reward_annealing_callback],
        progress_bar=True
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