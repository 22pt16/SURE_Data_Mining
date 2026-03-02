from collections import defaultdict


def train_bigram(train_data):
    transitions = defaultdict(lambda: defaultdict(int))

    for user, seq in train_data.items():
        for i in range(len(seq) - 1):
            transitions[seq[i]][seq[i + 1]] += 1

    return transitions


def recommend_next(transitions, user_sequence, top_k=10):
    if len(user_sequence) == 0:
        return []

    last_item = user_sequence[-1]

    if last_item not in transitions:
        return []

    next_items = transitions[last_item]

    ranked = sorted(next_items.items(), key=lambda x: x[1], reverse=True)

    return [item for item, _ in ranked[:top_k]]