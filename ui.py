import gradio as gr
import numpy as np
from env.supply_env import SupplyEnv

env = SupplyEnv()

def run_simulation(steps=30):
    obs, _ = env.reset()
    history = []

    total_reward = 0

    for step in range(steps):
        action = np.random.randint(0, env.action_space.n)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        history.append({
            "step": step,
            "inventory": obs[:env.num_warehouses].tolist(),
            "demand": obs[env.num_warehouses:].tolist(),
            "reward": reward
        })

        if terminated or truncated:
            break

    return history, total_reward


demo = gr.Interface(
    fn=run_simulation,
    inputs=gr.Slider(10, 100, value=30, label="Steps"),
    outputs=["json", "number"],
    title="📦 Supply Chain RL Simulator",
    description="Simulates inventory optimization using reinforcement learning environment."
)

if __name__ == "__main__":
    demo.launch()