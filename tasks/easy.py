DESCRIPTION = "Maintain minimum inventory and avoid stockouts"

def success(history):
    stockouts = 0

    for step in history:
        inventory = step["inventory"]
        demand = step["demand"]

        for i in range(len(inventory)):
            if inventory[i] < demand[i]:
                stockouts += 1

    # allow small mistakes
    return stockouts < 10


def grade(history):
    stockouts = 0
    total = 0

    for step in history:
        inventory = step["inventory"]
        demand = step["demand"]

        for i in range(len(inventory)):
            total += 1
            if inventory[i] < demand[i]:
                stockouts += 1

    score = 1 - (stockouts / total)
    return max(0.0, min(1.0, score))