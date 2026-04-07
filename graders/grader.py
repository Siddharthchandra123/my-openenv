def grade_inventory(total_reward):
    return max(0.0, min(total_reward / 4000, 1.0))


def grade_balance(inventory):
    imbalance = max(inventory) - min(inventory)
    score = 1 - (imbalance / 200)
    return max(0.0, min(score, 1.0))


def grade_fulfillment(fulfilled, demand):
    ratio = fulfilled / (demand + 1)
    return max(0.0, min(ratio, 1.0))