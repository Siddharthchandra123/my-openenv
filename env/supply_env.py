import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.models import Observation


class SupplyEnv(gym.Env):

    def __init__(self, num_warehouses=2, max_steps=50):
        super().__init__()

        self.num_warehouses = num_warehouses
        self.max_steps = max_steps

        # Actions:
        # 0 = do nothing
        # 1 = reorder stock
        # 2 = transfer stock
        # 3 = prioritize demand

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=500,
            shape=(num_warehouses * 2,),
            dtype=np.float32
        )
    from env.models import Observation

    def state(self):
        return Observation(
            inventory=self.inventory.tolist(),
            demand=self.demand.tolist()
        )

    def reset(self):
        self.step_count = 0
        self.inventory = np.random.randint(50, 100, size=self.num_warehouses)
        self.demand = np.random.randint(30, 60, size=self.num_warehouses)

        return self.state()   # ✅ IMPORTANT
    
    def step(self, action):
        self.step_count += 1
        reward = 0

        # Fulfill demand
        fulfilled = np.minimum(self.inventory, self.demand)
        self.inventory -= fulfilled

        # Positive reward (scaled)
        reward += np.sum(fulfilled) * 1.5

        # Unmet demand penalty (reduced)
        unmet = self.demand - fulfilled
        reward -= np.sum(unmet) * 1.2   # 🔥 reduced from 2

        # Holding cost (lighter)
        holding_cost = np.sum(self.inventory) * 0.03
        reward -= holding_cost

        # Action costs
        if action == 1:
            self.inventory += 30
            reward -= 5

        elif action == 2:
            if self.num_warehouses > 1:
                transfer_amount = min(10, self.inventory[0])
                self.inventory[0] -= transfer_amount
                self.inventory[1] += transfer_amount
                reward -= 3

        elif action == 3:
            reward += 3

        # 🔥 NEW: Fulfillment ratio reward (VERY IMPORTANT)
        total_demand = np.sum(self.demand) + 1
        fulfillment_ratio = np.sum(fulfilled) / total_demand
        reward += fulfillment_ratio * 20

        # New demand (slightly reduced variance)
        self.demand = np.random.randint(30, 60, size=self.num_warehouses)

        # 🔥 Auto restock (prevents collapse)
        self.inventory += np.random.randint(5, 15, size=self.num_warehouses)
        self.inventory = np.clip(self.inventory, 0, 200)

        # 🔥 Cap extreme negative rewards
        reward = max(reward, -150)

        terminated = False
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated
        return self.state(), float(reward), done, {}
            
    def get_typed_state(self):
        from env.state import SupplyState

        return SupplyState(
            inventory=self.inventory.copy(),
            demand=self.demand.copy()
        )
    
    def _get_obs(self):
        return np.concatenate([self.inventory, self.demand]).astype(np.float32)


def check_env():
    env = SupplyEnv()
    from stable_baselines3.common.env_checker import check_env
    check_env(env)