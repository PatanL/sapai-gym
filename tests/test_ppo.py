import os
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.ai import baselines
from sapai import Team, Player
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback 
from sb3_contrib.common.maskable.utils import get_action_masks


# train_maskableppo_superautopets.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor   import Monitor
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback

import functools

def make_env(valid_actions_only: bool = True):
    """Utility function to create and wrap a single environment."""
    env = SuperAutoPetsEnv(valid_actions_only=valid_actions_only, manual_battles=False)
    env = Monitor(env)
    return env


class AddToOpponentPoolCallback(BaseCallback):
    def __init__(self, save_freq: int, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # Create a new opponent model architecture. It's lightweight.
            # SB3 requires an env to initialize, so we get it from the model.
            policy_state_dict = self.model.policy.state_dict()

            policy_class = self.model.policy.__class__
            policy_kwargs = self.model.policy_kwargs

            self.training_env.env_method(
                "add_opponent_from_state",
                policy_state_dict,
                policy_class,
                policy_kwargs
            )
            # opponent_model = MaskablePPO(
            #     policy=self.model.policy.__class__,
            #     env=self.model.get_env(),
            #     policy_kwargs=self.model.policy_kwargs,  # mirror keyword args
            #     verbose=0
            # )
            
            # # Copy the weights from the current model to the new opponent model
            # opponent_model.policy.load_state_dict(self.model.policy.state_dict())
            
            # Get the actual environment instance (assuming DummyVecEnv or a single env)
            # Add to every sub-env in the VecEnv
            # for env in self.training_env.envs:
            #     env.add_opponent(opponent_model)
            # self.training_env.env_method("add_opponent", opponent_model)

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

if __name__ == "__main__":
    num_cpu = 1
    TOTAL_TIMESTEPS = 500_000
    seed = 42

    CHECKPOINT_DIR = "/kaggle/working/sap-ppo-checkpoints/"

    CHECKPOINT_PATH = "/kaggle/working/sap-ppo-checkpoints/sap_model_100000_steps.zip" 

    # ——— training env (parallel) ———
    # train_env = make_vec_env(
    #     env_id=SuperAutoPetsEnv,                  # Pass the class itself
    #     n_envs=num_cpu,
    #     seed=seed,
    #     vec_env_cls=SubprocVecEnv,
    #     env_kwargs=dict(valid_actions_only=False, manual_battles=False) 
    # )
    # train_env_fns = [
    #     functools.partial(make_env, valid_actions_only=False)
    #     for _ in range(num_cpu)
    # ]
    # train_env = SubprocVecEnv(train_env_fns)
    train_env = DummyVecEnv([lambda: make_env(valid_actions_only=False)])
    # train_env.seed(seed)

    # ——— eval env (single) ———
    # eval_env = make_vec_env(
    #     env_id=SuperAutoPetsEnv,
    #     n_envs=1, # Typically only need 1 env for evaluation
    #     vec_env_cls=DummyVecEnv,
    #     env_kwargs=dict(valid_actions_only=True, manual_battles=False)
    # )
    eval_env = DummyVecEnv([lambda: make_env(valid_actions_only=True)])

    # ——— eval callback ———
    # eval_freq here is in *vectorized* steps, so dividing by num_cpu
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=10_000 // num_cpu,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # ——— self‑play & annealing callbacks ———
    add_opponent_cb   = AddToOpponentPoolCallback(save_freq=10_000 // num_cpu,
                                                  verbose=1)
    reward_anneal_cb  = RewardAnnealingCallback(total_timesteps=TOTAL_TIMESTEPS,
                                                annealing_end_fraction=0.8)

    # ——— model & train ———
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--- Resuming training from checkpoint: {CHECKPOINT_PATH} ---")
        # The model will load its weights, optimizer, timesteps, etc.
        # You MUST pass the custom_objects if you have custom policies or layers
        # and you MUST pass the environment.
        model = MaskablePPO.load(
            CHECKPOINT_PATH,
            env=train_env,
            # If you want to change the learning rate, you can do so here:
            # learning_rate=1e-5, 
            tensorboard_log="./maskableppo_tensorboard/"
        )
        train_env.env_method("add_opponent_from_path", CHECKPOINT_PATH)
    else:
        print(f"--- Resuming training from checkpoint: {CHECKPOINT_PATH} ---")
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./maskableppo_tensorboard/",
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
        )
    model.learn(
        total_timesteps=500_000,
        callback=[add_opponent_cb, eval_callback, reward_anneal_cb],
        progress_bar=True,
        log_interval=10,
    )
    model.save("maskableppo_superautopets")

    # ——— clean up ———
    train_env.close()
    eval_env.close()

    print("\nTraining finished!")
    print(f"The best model was saved to: ./logs/best_model/best_model.zip")



# def make_env():
#     return SuperAutoPetsEnv(valid_actions_only=True, manual_battles=False)

# def main():
#     # Create the vectorized environment
#     env = DummyVecEnv([make_env])
#     # env = SuperAutoPetsEnv(valid_actions_only=False, manual_battles=False)
#     # obs, info = env.reset()
#     # action_mask = info.get('action_mask', None)
#     # print("action mask: ", action_mask.shape)
#     # check_env(env, warn=True)

#     # Initialize MaskablePPO
#     model = MaskablePPO(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         tensorboard_log="./maskableppo_superautopets_tensorboard/",
#         n_steps=2048,
#         batch_size=64,
#         gamma=0.99
#     )

#     TOTAL_TIMESTEPS = 500_000   
    
#      # ——— 1) set up EvalCallback ———
#     # We’ll evaluate every 10k steps on 10 episodes, saving the best model   
#     eval_env = DummyVecEnv([make_env])
#     eval_callback = MaskableEvalCallback(
#         eval_env,
#         best_model_save_path="./logs/best_model/",
#         log_path="./logs/results/",
#         eval_freq=10_000,         # run evaluation every 10k steps
#         n_eval_episodes=10,        # average over 10 episodes
#         deterministic=True,       # use deterministic actions at eval time
#         render=False,
#         verbose=1,
#     )

#     # callback for self-play
#     save_frequency = 10000  # Save and add to pool every 10,000 timesteps
#     save_directory = "./temp_opponents/"
#     add_opponent_callback = AddToOpponentPoolCallback(
#         save_freq=save_frequency, 
#         verbose=1
#     )

#     # callback for annealing the shaped rewards
#     reward_annealing_callback = RewardAnnealingCallback(total_timesteps=TOTAL_TIMESTEPS, annealing_end_fraction=0.8)

#     # Train the agent
#     model.learn(
#         total_timesteps=TOTAL_TIMESTEPS,  # Set to desired number of timesteps
#         log_interval=10,
#         tb_log_name="MaskablePPO_SuperAutoPets",
#         callback=[add_opponent_callback, eval_callback, reward_annealing_callback],
#         progress_bar=True
#     )

#     # Save the model
#     model.save("maskableppo_superautopets")

#     # Close the training environment
#     env.close()
#     eval_env.close()

# if __name__ == "__main__":
#     main()
