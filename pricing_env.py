import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PricingEnvironment(gym.Env):
    def __init__(self, max_inventory=500, max_steps=30, cost_per_item=8):
        super(PricingEnvironment, self).__init__()
        self.max_inventory = max_inventory
        self.max_steps = max_steps
        self.cost_per_item = cost_per_item

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([10.0]), high=np.array([50.0]), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = self.max_inventory
        self.demand_factor = np.random.uniform(0.7, 1.3)
        self.total_profit = 0  # Reset total profit at the beginning of an episode
        return self._get_obs(), {}

    def step(self, action):
        price = float(np.clip(action[0] if isinstance(action, (np.ndarray, list, tuple)) else action, 10.0, 50.0))
        
        base_demand = max(0, 100 - price)
        demand = int(base_demand * self.demand_factor)
        
        units_sold = min(demand, self.inventory)
        self.inventory -= units_sold
        self.current_step += 1
        
        # Accumulate profit at each step
        self.total_profit += (price - self.cost_per_item) * units_sold
        
        terminated = self.current_step >= self.max_steps or self.inventory <= 0
        
        # FINAL REWARD LOGIC:
        if terminated:
            # The final reward is the total accumulated profit
            reward = self.total_profit
            # Add a penalty for any leftover inventory
            if self.inventory > 0:
                reward -= self.inventory * self.cost_per_item
        else:
            # No reward is given until the episode is over
            reward = 0

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Normalize the observations for better learning
        normalized_inventory = self.inventory / self.max_inventory
        normalized_demand = (self.demand_factor - 0.7) / (1.3 - 0.7)
        return np.array([normalized_inventory, normalized_demand], dtype=np.float32)

    def render(self):
        print(f"Step: {self.current_step}, Inventory: {self.inventory}, Demand Factor: {self.demand_factor}")