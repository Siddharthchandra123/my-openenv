import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.supply_env import SupplyEnv
import numpy as np

env = SupplyEnv()
obs = env.reset()


total_reward = 0

for step in range(50):
    inventory = np.array(obs.inventory)
    demand = np.array(obs.demand)

    if np.sum(inventory) < np.sum(demand):
        action = 1  
    elif env.num_warehouses > 1 and inventory[0] > inventory[1] + 20:
        action = 2  
    else:
        action = 0  

    obs, reward, terminated, truncated = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print("Smart Baseline Total Reward:", total_reward)