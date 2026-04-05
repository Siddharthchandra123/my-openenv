from dataclasses import dataclass
import numpy as np

@dataclass
class SupplyState:
    inventory: np.ndarray
    demand: np.ndarray