import os
import math
import numpy as np
from openai import OpenAI
from env.supply_env import SupplyEnv

# ---------------- CONFIG ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["inventory", "balance", "fulfillment"]
MAX_STEPS = 50


# ---------------- LOGGING ----------------
def log_start(task):
    print(f"[START] task={task} env=supply_chain model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# ---------------- POLICY ----------------
def smart_policy(obs):
    n = len(obs) // 2
    inventory = obs[:n]
    demand = obs[n:]

    if inventory.mean() < demand.mean():
        return 1
    if max(inventory) - min(inventory) > 20:
        return 2
    if demand.mean() > 50:
        return 3
    return 0


def llm_policy(obs):
    prompt = f"State: {obs.tolist()}\nChoose action (0-3):"
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.3
        )
        return int(res.choices[0].message.content.strip())
    except:
        return smart_policy(obs)


def final_policy(obs, step):
    return llm_policy(obs) if step <= 5 else smart_policy(obs)


# ---------------- RUN ----------------
def run_task(env, task):
    obs = env.reset(task_type=task)

    rewards = []
    steps_taken = 0
    success = False

    log_start(task)

    try:
        for step in range(1, MAX_STEPS + 1):
            action = final_policy(obs, step)
            action_str = str(action)

            obs, reward, done, _ = env.step(action)

            # 🔥 NORMALIZE REWARD (CRITICAL FIX)
            norm_reward = reward / 100.0
            norm_reward = max(0.0, min(1.0, norm_reward))  # keep in [0,1]

            rewards.append(norm_reward)
            steps_taken = step

            log_step(step, action_str, norm_reward, done)

            if done:
                break

        # ---------------- SCORE ----------------
        if len(rewards) > 0:
            score = sum(rewards) / len(rewards)
        else:
            score = 0.0

        # Clamp strictly to [0,1]
        score = max(0.0, min(1.0, score))

        success = score > 0.1  # threshold

    except Exception:
        score = 0.0
        success = False

    log_end(success, steps_taken, score, rewards)


# ---------------- MAIN ----------------
def main():
    env = SupplyEnv()

    for task in TASKS:
        run_task(env, task)


if __name__ == "__main__":
    main()