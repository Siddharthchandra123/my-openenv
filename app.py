from fastapi import FastAPI
from env.supply_env import SupplyEnv

app = FastAPI()
env = SupplyEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/reset")
def reset():
    obs, _ = env.reset()
    return {"state": obs.tolist()}