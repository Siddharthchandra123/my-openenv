from env.supply_env import SupplyEnv

env = SupplyEnv()

obs, _ = env.reset()

print("Raw obs:", obs)

state = env.state()
print("Typed state:", state)

# Step once
obs, reward, done, info = env.step(1)

print("After step:", env.state())