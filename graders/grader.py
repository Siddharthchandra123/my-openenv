import numpy as np

EPS = 1e-6


def clamp_score(score: float) -> float:
    """
    Ensures score is strictly within (0, 1)
    """
    return max(EPS, min(1.0 - EPS, float(score)))



def grade_inventory(total_stockouts: float, total_steps: int) -> float:
    """
    Penalizes stockouts.
    Lower stockout rate = higher score.
    """
    stockout_rate = total_stockouts / (total_steps + 1)
    score = 1.0 - stockout_rate
    return clamp_score(score)



def grade_balance(inventory) -> float:
    """
    Penalizes imbalance between warehouses.
    More balanced = higher score.
    """
    inventory = np.array(inventory)
    imbalance = np.max(inventory) - np.min(inventory)

    # Normalize by expected max range
    score = 1.0 - (imbalance / 200.0)
    return clamp_score(score)

def grade_fulfillment(total_fulfilled: float, total_demand: float) -> float:
    """
    Rewards fulfillment ratio.
    Higher fulfilled demand = higher score.
    """
    ratio = total_fulfilled / (total_demand + 1)
    return clamp_score(ratio)