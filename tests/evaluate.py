# /tests/evaluate.py

import os
import sys
import time # Import the time module for delays

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from sapai_gym1 import SuperAutoPetsEnv
from sb3_contrib import MaskablePPO

def main():
    # --- Configuration ---
    # Path to the best model saved by the EvalCallback during training
    BEST_MODEL_PATH = "./maskableppo_superautopets-07-30.zip"
    NUM_EPISODES = 2 # How many games to watch
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model not found at {BEST_MODEL_PATH}")
        print("Please run the training script (test_ppo.py) first.")
        return

    # --- Environment Setup ---
    # Create a single, non-vectorized environment for evaluation and rendering.
    # It MUST have valid_actions_only=True.
    eval_env = SuperAutoPetsEnv(valid_actions_only=True, manual_battles=False)

    # --- Load the Trained Model ---
    print(f"Loading model from: {BEST_MODEL_PATH}")
    # Load onto CPU, as inference is fast and we want to save GPU memory.
    model = MaskablePPO.load(BEST_MODEL_PATH, device='cpu')
    print("Model loaded successfully.")

    # --- Evaluation Loop ---
    for episode in range(NUM_EPISODES):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        turn = 0
        
        print(f"\n H-- Starting Evaluation Episode {episode + 1} ---")
        
        while not done:
            turn += 1
            print(f"\n--- Turn {turn} ---")
            
            # Get the action mask from the single environment
            action_masks = eval_env.action_masks()
            
            # Get the agent's action
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # Take the step in the environment
            obs, reward, done, truncated, info = eval_env.step(action)
            
            # --- RENDER THE ENVIRONMENT ---
            # This will call the render() method in your SuperAutoPetsEnv
            eval_env.render()
            # ----------------------------

            total_reward += reward
            
            # Add a small delay so you can read the text output
            time.sleep(0.5)

        print(f"--- Episode {episode + 1} Finished ---")
        print(f"Total Reward: {total_reward}")

    eval_env.close()

if __name__ == "__main__":
    main()