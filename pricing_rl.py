import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from pricing_env import PricingEnvironment
import os
import warnings
warnings.filterwarnings("ignore")

# Fixed price evaluation
def test_fixed_prices():
    print("\nFixed Price Strategy Evaluation:")
    env = PricingEnvironment()
    for price in range(10, 35, 5):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            obs, reward, done, _, _ = env.step(price)
            total_reward += reward
        print(f"Fixed price ${price}: Total Profit = ${total_reward:.2f}")

# Train PPO model
def train_model():
    print("\nTraining PPO model...\n")

    env = DummyVecEnv([lambda: PricingEnvironment()])  # Safer for single env setup

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_logs",
        n_steps=2048,
        batch_size=64
    )

    model.learn(total_timesteps=100_000)

    model.save("ppo_pricing_model")
    print("\nModel training complete and saved as 'ppo_pricing_model.zip'.")

# Predict optimal price using trained model
def predict_price(step, inventory, demand_factor):
    model_path = "ppo_pricing_model.zip"
    if not os.path.exists(model_path):
        print("Trained model not found. Please run training first.")
        return

    model = PPO.load(model_path)
    env = PricingEnvironment()
    obs = np.array([step, inventory, demand_factor], dtype=np.float32).reshape(1, -1)
    action, _ = model.predict(obs, deterministic=True)
    recommended_price = float(np.clip(action[0], 10, 50))
    print(f"Recommended Price: ${recommended_price:.2f}")
    return recommended_price

if __name__ == "__main__":
    test_fixed_prices()
    train_model()
    predict_price(10, 300, 1.1)
