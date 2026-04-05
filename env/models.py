from pydantic import BaseModel
from typing import List

class Observation(BaseModel):
    inventory: List[float]
    demand: List[float]

class Action(BaseModel):
    action: int

class Reward(BaseModel):
    value: float