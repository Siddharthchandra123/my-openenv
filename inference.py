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

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("[START]")

env = SupplyEnv()
obs, _ = env.reset()

total_reward = 0

for step in range(50):
    action = random.randint(0, env.action_space.n - 1)

    obs, reward, done, info = env.step(action)

    total_reward += reward

    print(f"[STEP] step={step} reward={reward} total_reward={total_reward}")

    if done:
        break

print(f"[END] grade={grade(total_reward)} total_reward={total_reward}")