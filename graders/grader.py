def grade(total_reward):
    if total_reward < 0:
        return 0.0
    elif total_reward > 3000:
        return 1.0
    else:
        return total_reward / 3000