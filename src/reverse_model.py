from collections import defaultdict

def train_reverse_model(long_users):
    reverse_transitions = defaultdict(lambda: defaultdict(int))

    for user, items in long_users.items():
        reversed_seq = items[::-1]

        for i in range(len(reversed_seq) - 1):
            curr_item = reversed_seq[i]
            prev_item = reversed_seq[i + 1]
            reverse_transitions[curr_item][prev_item] += 1

    return reverse_transitions


def predict_prior_items(reverse_model, first_item, top_k=2):

    if first_item not in reverse_model:
        return []

    sorted_items = sorted(
        reverse_model[first_item].items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in sorted_items[:top_k]]