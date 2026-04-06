DESCRIPTION = "Maximize profit while minimizing holding cost and unnecessary actions"

def success(history):
    total_reward = sum(step["reward"] for step in history)
    return total_reward > 2500


def grade(history):
    total_reward = sum(step["reward"] for step in history)

    holding_penalty = 0
    action_penalty = 0

    for step in history:
        inv = step["inventory"]

        # excessive inventory = waste
        holding_penalty += sum(inv)

        # penalize unnecessary actions
        if step.get("action", 0) == 1:
            action_penalty += 1

    reward_score = max(0, min(1, total_reward / 4000))
    holding_score = max(0, 1 - (holding_penalty / 5000))
    action_score = max(0, 1 - (action_penalty / 100))

    final_score = (
        0.5 * reward_score +
        0.3 * holding_score +
        0.2 * action_score
    )

    return max(0.0, min(1.0, final_score))