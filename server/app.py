import uvicorn
from fastapi import FastAPI
from env.supply_env import SupplyEnv

app = FastAPI()
env = SupplyEnv()

@app.get("/")
def root():
    obs = env.reset()
    return {"state": obs.tolist()}   

@app.post("/reset")
@app.get("/reset")
def reset():
    try:
        obs = env.reset()

        if hasattr(obs, "dict"):
            return obs.dict()
        elif hasattr(obs, "tolist"):
            return obs.tolist()
        else:
            return obs

    except Exception as e:
        return {"error": str(e)}
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# ✅ REQUIRED ENTRY POINT (VERY IMPORTANT)
if __name__ == "__main__":
    main()