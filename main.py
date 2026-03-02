from src.preprocessing import load_and_preprocess
from src.arm_filter import (
    extract_uninteresting,
    build_transactions,
    run_fpgrowth,
    run_apriori,
    get_items_to_remove,
    filter_sequences,
)
from src.recommender import train_bigram
from src.evaluation import evaluate

import pandas as pd


def simple_leave_one_out(df):

    train = {}
    test = {}

    grouped = df.groupby("user_id")

    for user, group in grouped:
        items = group.sort_values("timestamp")["item_id"].tolist()

        if len(items) < 2:
            continue

        train[user] = items[:-1]
        test[user] = items[-1]

    return train, test


if __name__ == "__main__":

    raw_path = "data/raw/u.data"
    processed_path = "data/processed/processed.csv"

    df = load_and_preprocess(raw_path, processed_path)

    train, test = simple_leave_one_out(df)

    # Extract uninteresting items
    unf_dict = extract_uninteresting(df)
    transactions = build_transactions(unf_dict)

    print("\n===== APRIORI =====")
    apriori_rules, apriori_time = run_apriori(transactions, min_support=0.01)
    print("Apriori Time:", apriori_time)

    if apriori_rules is not None:
        apriori_remove = get_items_to_remove(apriori_rules)
        print("Apriori Items Removed:", len(apriori_remove))
    else:
        apriori_remove = set()
        print("Apriori found no rules.")
    
    print("\n===== RECOMMENDATION WITH APRIORI FILTERING =====")
    filtered_train = filter_sequences(train, apriori_remove)
    transitions = train_bigram(filtered_train)
    results = evaluate(transitions, test, filtered_train)


    print("\n===== FP-GROWTH =====")
    fp_rules, fp_time = run_fpgrowth(transactions, min_support=0.01)
    print("FP-Growth Time:", fp_time)

    if fp_rules is not None:
        fp_remove = get_items_to_remove(fp_rules)
        print("FP-Growth Items Removed:", len(fp_remove))
    else:
        fp_remove = set()
        print("FP-Growth found no rules.")

    print("\n===== RECOMMENDATION WITH FP-GROWTH FILTERING =====")
    filtered_train = filter_sequences(train, fp_remove)
    transitions = train_bigram(filtered_train)
    results = evaluate(transitions, test, filtered_train)

    print("Evaluation Results:", results)

    print("\n===== TIME COMPARISON =====")
    print("Apriori Time:", apriori_time)
    print("FP-Growth Time:", fp_time)