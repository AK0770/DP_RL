from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the NEW, retrained model
model = PPO.load("models/best_model.zip")

class PriceRequest(BaseModel):
    step: int
    inventory: int
    demand_factor: float

@app.post("/predict")
def predict_price(data: PriceRequest):
    # Normalize the inputs from the dashboard
    normalized_inventory = data.inventory / 500.0
    normalized_demand = (data.demand_factor - 0.7) / (1.3 - 0.7)
    
    obs = np.array([normalized_inventory, normalized_demand], dtype=np.float32).reshape(1, -1)
    action, _ = model.predict(obs, deterministic=True)
    
    recommended_price = float(action[0])
    
    print(f"[DEBUG] Input: {obs}, Action: {action}, Price: {recommended_price}")
    
    return {"recommended_price": recommended_price}