def grade(total_reward):
    score = (total_reward + 500) / 3500
    return max(0.0, min(1.0, score))