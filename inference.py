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
prev_demand = None

def smart_policy(obs):
    global prev_demand

    num_warehouses = len(obs) // 2
    inventory = obs[:num_warehouses]
    demand = obs[num_warehouses:]

    if prev_demand is not None:
        if demand.mean() > prev_demand.mean():
            return 1  # demand increasing → reorder early

    prev_demand = demand.copy()
    return 0
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("[START]")

env = SupplyEnv()
obs = env.reset()

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