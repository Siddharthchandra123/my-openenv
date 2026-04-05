from env.supply_env import SupplyEnv

env = SupplyEnv()

obs, _ = env.reset()
print("Initial State:", obs)

for _ in range(5):
    obs, reward, done, truncated, _ = env.step(1)
    print("Step → Reward:", reward)