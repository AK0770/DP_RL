import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from pricing_env import PricingEnvironment

env = PricingEnvironment()
check_env(env)

# Create an evaluation callback to save the best model during training
# This will save the best model in the models/ folder
eval_callback = EvalCallback(env, best_model_save_path='./models/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gamma=0.999,          # Prioritize long-term rewards even more
    gae_lambda=0.95,
    ent_coef=0.01,        # Encourage exploration
    learning_rate=0.0003,
    clip_range=0.2
)

# Increase training time significantly to allow for deep learning
model.learn(total_timesteps=500_000, callback=eval_callback)

# Save the final model (the best model is already saved by the callback)
model.save("models/final_model.zip")
print("✅ Model training complete. The best model is saved as best_model.zip.")