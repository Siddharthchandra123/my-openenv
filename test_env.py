from env.supply_env import SupplyEnv

env = SupplyEnv()
obs = env.reset()

print("Initial obs:", obs)

for step in range(5):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    print(f"Step {step} | Action: {action} | Reward: {reward}")

    if done:
        break