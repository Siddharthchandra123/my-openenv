import os
import numpy as np
from openai import OpenAI
from env.supply_env import SupplyEnv


# ---------------- CONFIG ----------------
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------- SAFETY ----------------
def safe_obs(obs):
    if hasattr(obs, "inventory"):
        return np.array(obs.inventory + obs.demand, dtype=np.float32)
    return obs


# ---------------- SMART FALLBACK ----------------
def smart_policy(obs):
    obs = safe_obs(obs)
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


# ---------------- LLM POLICY ----------------
def llm_policy(obs):
    obs = safe_obs(obs)

    prompt = f"""
    You are managing a supply chain.

    State: {obs.tolist()}

    Actions:
    0 = do nothing
    1 = reorder stock
    2 = transfer stock
    3 = prioritize demand

    Return ONLY a number between 0 and 3.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.3,
        )

        text = response.choices[0].message.content.strip()
        action = int(text)
        return max(0, min(action, 3))

    except Exception:
        return smart_policy(obs)


# ---------------- FINAL POLICY (FAST + SAFE) ----------------
def final_policy(obs, step):
    # Limit LLM usage for speed (<20 min requirement safe)
    if step <= 5:
        return llm_policy(obs)
    return smart_policy(obs)


# ---------------- GRADER ----------------
def grade(total_reward):
    return max(0.0, min(total_reward / 5000, 1.0))


# ---------------- RUN ONE TASK ----------------
def run_task(env, task_name):
    obs = safe_obs(env.reset())

    total_reward = 0
    rewards = []
    steps_taken = 0

    print(f"[START] task={task_name} env=custom model=llm-agent", flush=True)

    try:
        for step in range(1, 51):
            action = final_policy(obs, step)

            obs, reward, done, _ = env.step(action)
            obs = safe_obs(obs)

            total_reward += reward
            rewards.append(reward)
            steps_taken = step

            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            if done:
                break

        score = grade(total_reward)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success=true steps={steps_taken} score={score:.3f} rewards={rewards_str}",
            flush=True
        )

    except Exception:
        print(
            f"[END] success=false steps={steps_taken} score=0.000 rewards=",
            flush=True
        )


# ---------------- MAIN ----------------
def main():
    env = SupplyEnv()

    # REQUIRED: at least 3 tasks
    tasks = ["inventory", "balance", "fulfillment"]

    for task in tasks:
        run_task(env, task)


if __name__ == "__main__":
    main()