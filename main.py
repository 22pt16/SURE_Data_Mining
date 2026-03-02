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

import numpy as np


def simple_leave_one_out(df):
    """
    Temporary split for Person 2 branch.
    Last interaction → test
    Remaining → train
    """

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

    print("\n===== LOADING & PREPROCESSING =====")
    df = load_and_preprocess(raw_path, processed_path)

    print("Total users:", df["user_id"].nunique())
    print("Total items:", df["item_id"].nunique())

    # ----------------------------
    # Train/Test Split
    # ----------------------------
    train, test = simple_leave_one_out(df)
    print("Users after split:", len(train))

    # ----------------------------
    # BASELINE (NO FILTERING)
    # ----------------------------
    print("\n===== BASELINE (NO FILTERING) =====")

    baseline_transitions = train_bigram(train)
    baseline_results = evaluate(baseline_transitions, test, train)

    print("Baseline Results:", baseline_results)

    # ----------------------------
    # Build Transactions
    # ----------------------------
    unf_dict = extract_uninteresting(df)
    transactions = build_transactions(unf_dict)

    print("Total uninteresting transactions:", len(transactions))

    # ==========================================================
    # ===================== APRIORI ============================
    # ==========================================================
    print("\n===== APRIORI =====")

    apriori_rules, apriori_time = run_apriori(
        transactions,
        min_support=0.01,
        min_conf=0.3
    )

    if apriori_rules is not None:
        apriori_remove = get_items_to_remove(apriori_rules)
    else:
        apriori_remove = set()

    print("Apriori Time:", round(apriori_time, 4), "seconds")
    print("Apriori Items Removed:", len(apriori_remove))

    filtered_train_ap = filter_sequences(train, apriori_remove)

    transitions_ap = train_bigram(filtered_train_ap)
    results_ap = evaluate(transitions_ap, test, filtered_train_ap)

    print("Apriori Results:", results_ap)

    # ==========================================================
    # ===================== FP-GROWTH ==========================
    # ==========================================================
    print("\n===== FP-GROWTH =====")

    fp_rules, fp_time = run_fpgrowth(
        transactions,
        min_support=0.01,
        min_conf=0.3
    )

    if fp_rules is not None:
        fp_remove = get_items_to_remove(fp_rules)
    else:
        fp_remove = set()

    print("FP-Growth Time:", round(fp_time, 4), "seconds")
    print("FP-Growth Items Removed:", len(fp_remove))

    filtered_train_fp = filter_sequences(train, fp_remove)

    transitions_fp = train_bigram(filtered_train_fp)
    results_fp = evaluate(transitions_fp, test, filtered_train_fp)

    print("FP-Growth Results:", results_fp)

    # ==========================================================
    # ===================== SUMMARY ============================
    # ==========================================================
    print("\n===== FINAL COMPARISON =====")

    print("\nBaseline:", baseline_results)
    print("Apriori :", results_ap)
    print("FP-Growth:", results_fp)

    print("\nTime Comparison:")
    print("Apriori Time:", round(apriori_time, 4), "seconds")
    print("FP-Growth Time:", round(fp_time, 4), "seconds")

    if apriori_time > 0:
        speedup = apriori_time / fp_time if fp_time > 0 else np.inf
        print("FP-Growth Speedup over Apriori:", round(speedup, 2), "x")