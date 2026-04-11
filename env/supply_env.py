import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SupplyEnv(gym.Env):

    def __init__(self, num_warehouses=2, max_steps=50):
        super().__init__()

        self.num_warehouses = num_warehouses
        self.max_steps = max_steps
        self.task_type = "inventory"

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=500,
            shape=(num_warehouses * 2,),
            dtype=np.float32
        )

    def reset(self, task_type="inventory"):
        self.task_type = task_type
        self.step_count = 0

        self.inventory = np.random.randint(50, 100, size=self.num_warehouses)
        self.demand = np.random.randint(30, 60, size=self.num_warehouses)

        return self._get_obs()

    def step(self, action):
        self.step_count += 1
        reward = 0

        # Demand spike (hardness)
        if np.random.rand() < 0.2:
            self.demand += np.random.randint(20, 50, size=self.num_warehouses)

        fulfilled = np.minimum(self.inventory, self.demand)
        unmet = self.demand - fulfilled

        self.inventory -= fulfilled

        # Base reward
        reward += np.sum(fulfilled) * 1.2
        reward -= np.sum(unmet) * 1.5

        # Holding cost
        reward -= np.sum(self.inventory) * 0.03

        # Action effects
        if action == 1:
            self.inventory += 30
            reward -= 8

        elif action == 2 and self.num_warehouses > 1:
            transfer = min(10, self.inventory[0])
            self.inventory[0] -= transfer
            self.inventory[1] += transfer
            reward -= 3

        elif action == 3:
            reward += 2

        # TASK-SPECIFIC SHAPING
        if self.task_type == "inventory":
            reward -= np.sum(unmet) * 2.5

        elif self.task_type == "balance":
            imbalance = max(self.inventory) - min(self.inventory)
            reward -= imbalance * 2.0

        elif self.task_type == "fulfillment":
            reward += np.sum(fulfilled) * 2.5

        # New demand
        self.demand = np.random.randint(20, 80, size=self.num_warehouses)

        # Auto restock
        self.inventory += np.random.randint(5, 15, size=self.num_warehouses)
        self.inventory = np.clip(self.inventory, 0, 200)

        terminated = False
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        return self._get_obs(), float(reward), done, {}

    def _get_obs(self):
        return np.concatenate([self.inventory, self.demand]).astype(np.float32)