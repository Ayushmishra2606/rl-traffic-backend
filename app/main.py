from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
from typing import List
from fastapi.middleware.cors import CORSMiddleware


from .model_loader import load_model


app = FastAPI(title="RL Traffic Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, model_loaded = load_model()


class ObsIn(BaseModel):
    obs: List[float]


class SimRequest(BaseModel):
    policy: str = 'fixed' # 'fixed' | 'random' | 'rl'
    steps: int = 200


@app.post('/infer')
async def infer(payload: ObsIn):
    if not model_loaded or model is None:
        return {"action": 0, "note": "no model loaded"}
    obs = np.array(payload.obs, dtype=np.float32)
    with torch.no_grad():
        t = torch.tensor(obs).unsqueeze(0)
        logits = model(t)
        action = int(logits.argmax(1).item())
    return {"action": action}




@app.post('/simulate')
async def simulate(payload: SimRequest):
    # Lazy import to avoid importing RL env when not needed
    from rl.envs.simple_intersection import SimpleIntersection
    env = SimpleIntersection(max_steps=payload.steps)
    obs = env.reset()
    traj = []
    for _ in range(payload.steps):
        if payload.policy == 'fixed':
            a = 0 if env.elapsed < 10 else 1
        elif payload.policy == 'random':
            a = int(np.random.choice([0,1]))
        elif payload.policy == 'rl' and model_loaded:
            import torch
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = int(model(t).argmax(1).item())
        else:
            a = 0
        obs, r, done, _ = env.step(a)
        traj.append({'obs': obs.tolist(), 'action': int(a), 'reward': float(r)})
        if done:
            break
    return {'traj': traj}


@app.get('/health')
async def health():
    return {"status": "ok", "model_loaded": model_loaded}