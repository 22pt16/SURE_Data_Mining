from src.reverse_model import predict_prior_items

def extend_short_sequences(short_users, reverse_model, k=2):

    enhanced_sequences = {}

    for user, items in short_users.items():

        if len(items) == 0:
            enhanced_sequences[user] = items
            continue

        first_item = items[0]

        pseudo_items = predict_prior_items(reverse_model, first_item, top_k=k)

        enhanced_sequences[user] = pseudo_items + items

    return enhanced_sequences