import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.supply_env import SupplyEnv
import numpy as np

env = SupplyEnv()
obs, _ = env.reset()

total_reward = 0

for step in range(50):
    inventory = obs[:env.num_warehouses]
    demand = obs[env.num_warehouses:]

    # 🔥 Simple decision logic
    if np.sum(inventory) < np.sum(demand):
        action = 1  # reorder
    elif env.num_warehouses > 1 and inventory[0] > inventory[1] + 20:
        action = 2  # transfer
    else:
        action = 0  # do nothing

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print("Smart Baseline Total Reward:", total_reward)