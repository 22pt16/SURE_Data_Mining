import numpy as np


def mrr_at_k(ranked_list, ground_truth):
    for idx, item in enumerate(ranked_list):
        if item == ground_truth:
            return 1 / (idx + 1)
    return 0


def ndcg_at_k(ranked_list, ground_truth):
    for idx, item in enumerate(ranked_list):
        if item == ground_truth:
            return 1 / np.log2(idx + 2)
    return 0


def evaluate(transitions, test_data, user_sequences, top_k=10):
    mrr_scores = []
    ndcg_scores = []

    for user, true_item in test_data.items():

        if user not in user_sequences:
            continue

        ranked = recommend_next(transitions, user_sequences[user], top_k)

        mrr_scores.append(mrr_at_k(ranked, true_item))
        ndcg_scores.append(ndcg_at_k(ranked, true_item))

    return {
        "MRR": np.mean(mrr_scores),
        "nDCG": np.mean(ndcg_scores)
    }


# Import inside file to avoid circular dependency
from src.recommender import recommend_next