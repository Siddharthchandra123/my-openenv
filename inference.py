import os
import random
import numpy as np

from env.supply_env import SupplyEnv
from graders.grader import grade
# REQUIRED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# REQUIRED: OpenAI client usage
from openai import OpenAI
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)
def smart_policy(obs):
    num_warehouses = len(obs) // 2
    inventory = obs[:num_warehouses]
    demand = obs[num_warehouses:]

    avg_inventory = inventory.mean()
    avg_demand = demand.mean()

    # 🔥 Rule 1: If inventory is too low → reorder
    if avg_inventory < avg_demand:
        return 1  # reorder

    # 🔥 Rule 2: If imbalance between warehouses → transfer
    if num_warehouses > 1:
        if max(inventory) - min(inventory) > 20:
            return 2  # transfer

    # 🔥 Rule 3: If demand is high → prioritize
    if avg_demand > 50:
        return 3  # prioritize demand

    # Default
    return 0
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("[START]")

env = SupplyEnv()
obs, _ = env.reset()

total_reward = 0

for step in range(50):
    obs = env.reset()

total_reward = 0

for step in range(50):
    action = smart_policy(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    if done:
        break

    
    print(f"[STEP] step={step} reward={reward} total_reward={total_reward}")

print(f"[END] grade={grade(total_reward)} total_reward={total_reward}")