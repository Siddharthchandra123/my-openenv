DESCRIPTION = "Balance inventory across warehouses while minimizing stockouts"

def success(history):
    imbalance_penalty = 0

    for step in history:
        inv = step["inventory"]

        if len(inv) > 1:
            imbalance_penalty += abs(inv[0] - inv[1])

    return imbalance_penalty < 1000


def grade(history):
    imbalance_penalty = 0
    stockouts = 0
    total = 0

    for step in history:
        inv = step["inventory"]
        demand = step["demand"]

        if len(inv) > 1:
            imbalance_penalty += abs(inv[0] - inv[1])

        for i in range(len(inv)):
            total += 1
            if inv[i] < demand[i]:
                stockouts += 1

    stockout_score = 1 - (stockouts / total)
    balance_score = max(0, 1 - (imbalance_penalty / 2000))

    final_score = 0.6 * stockout_score + 0.4 * balance_score
    return max(0.0, min(1.0, final_score))