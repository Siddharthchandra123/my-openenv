import numpy as np
from env.supply_env import SupplyEnv


def random_agent(env):
    obs = env.reset()
    total_reward = 0

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


def smart_policy(obs):
    n = len(obs)//2
    inventory = obs[:n]
    demand = obs[n:]

    if inventory.mean() < demand.mean():
        return 1
    if max(inventory) - min(inventory) > 20:
        return 2
    if demand.mean() > 50:
        return 3
    return 0


def smart_agent(env):
    obs = env.reset()
    total_reward = 0

    for _ in range(50):
        action = smart_policy(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


env = SupplyEnv()

random_scores = [random_agent(env) for _ in range(10)]
smart_scores = [smart_agent(env) for _ in range(10)]

print("Random avg:", np.mean(random_scores))
print("Smart avg:", np.mean(smart_scores))