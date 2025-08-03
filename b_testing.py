import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from stable_baselines3 import PPO
from pricing_env import PricingEnvironment

# Constants
N_EPISODES = 20
FIXED_PRICE = 30
DISCOUNT_STEP = 15

# Load PPO model
ppo_model = PPO.load("./models/ppo_pricing")

# Strategies to compare
strategies = ['PPO', 'Fixed', 'Random', 'Rule-Based']
results = []

# Run each strategy
for strategy in strategies:
    for episode in range(N_EPISODES):
        env = PricingEnvironment()
        obs, _ = env.reset()
        done = False
        total_profit = 0
        prices = []
        inventory_start = env.max_inventory
        
        while not done:
            if strategy == 'PPO':
                action, _ = ppo_model.predict(obs, deterministic=True)
            elif strategy == 'Fixed':
                action = np.array([FIXED_PRICE])
            elif strategy == 'Random':
                action = env.action_space.sample()
            elif strategy == 'Rule-Based':
                if env.current_step < DISCOUNT_STEP:
                    action = np.array([40])
                else:
                    action = np.array([25])
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_profit += reward
            prices.append(info['price'])
        
        results.append({
            'Strategy': strategy,
            'Episode': episode + 1,
            'Profit': total_profit,
            'AvgPrice': np.mean(prices),
            'PriceVolatility': np.std(prices),
            'InventoryUsed': inventory_start - env.inventory
        })
        env.close()

# Convert to DataFrame
df = pd.DataFrame(results)

# Summary stats
summary = df.groupby('Strategy').agg({
    'Profit': ['mean', 'std'],
    'InventoryUsed': 'mean',
    'PriceVolatility': 'mean',
    'AvgPrice': 'mean'
}).reset_index()

summary.columns = ['Strategy', 'AvgProfit', 'ProfitStdDev', 'AvgInventoryUsed', 'AvgDiscountDepth', 'AvgPrice']

print("\n=== Evaluation Summary ===")
print(summary)

# Visualization
sns.set(style="whitegrid")

# Profit over episodes
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Episode', y='Profit', hue='Strategy', marker='o')
plt.title("Profit per Episode by Strategy")
plt.tight_layout()
plt.show()

# Box plot of profits
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Strategy', y='Profit', palette='Set3')
plt.title("Profit Distribution")
plt.tight_layout()
plt.show()

# Bar chart of average metrics
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x='Strategy', y='AvgProfit')
plt.title("Average Profit per Strategy")
plt.tight_layout()
plt.show()
