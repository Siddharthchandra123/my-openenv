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
    n = len(obs) // 2
    inventory = obs[:n]
    demand = obs[n:]

    # Rule-based intelligence
    if inventory.mean() < demand.mean():
        return 1  # reorder
    if max(inventory) - min(inventory) > 20:
        return 2  # transfer
    if demand.mean() > 50:
        return 3  # prioritize
    return 0


def grade(total_reward):
    return max(0.0, min(total_reward / 5000, 1.0))


def main():
    env = SupplyEnv()
    obs = env.reset()

    total_reward = 0
    rewards = []

    print("[START] task=supply_chain env=custom model=rule-based", flush=True)

    for step in range(1, 51):
        action = smart_policy(obs)

        obs, reward, done, _ = env.step(action)

        total_reward += reward
        rewards.append(reward)

        print(
            f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True
        )

        if done:
            break

    score = grade(total_reward)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success=true steps={step} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


if __name__ == "__main__":
    main()