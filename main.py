from src.preprocessing import load_and_preprocess
from src.split import leave_one_out_split, split_short_long
from src.reverse_model import train_reverse_model
from src.extension import extend_short_sequences

from src.arm_filter import (
    extract_uninteresting,
    build_transactions,
    run_apriori,
    run_fpgrowth,
    get_items_to_remove,
    filter_sequences,
)

from src.recommender import train_bigram
from src.evaluation import evaluate

import time


if __name__ == "__main__":

    # ==========================================================
    # ================== LOADING DATA ==========================
    # ==========================================================
    print("\n===== LOADING DATA =====")
    df = load_and_preprocess("data/raw/u.data", "data/processed/processed.csv")


    print("\n===== DATASET STATISTICS =====")
    print("Total interactions:", len(df))
    print("Total users:", df["user_id"].nunique())
    print("Total items:", df["item_id"].nunique())

    print("\nAvg interactions per user:",  round(len(df)/df["user_id"].nunique(), 2))

    unf_count = len(df[df["liked"] == 0])
    print("Total uninteresting interactions:", unf_count)
    print("Uninteresting ratio:", round(unf_count/len(df)*100, 2), "%")

    # ==========================================================
    # ================= LEAVE-ONE-OUT SPLIT ====================
    # ==========================================================
    train, val, test = leave_one_out_split(df)
    print("\nUsers after split:", len(train))

    # ==========================================================
    # ================= BASELINE (NO SSE, NO ARM) ==============
    # ==========================================================
    print("\n===== M1: BASELINE =====")

    start = time.time()
    transitions_base = train_bigram(train)
    results_base = evaluate(transitions_base, test, train)
    end = time.time()

    baseline_time = end - start

    print("Baseline:", results_base, "| Time:", round(baseline_time, 4), "sec")

    # ==========================================================
    # ================= ARM ONLY (NO SSE) ======================
    # ==========================================================
    # ==========================================================
    # ================= M2: APRIORI ONLY =======================
    # ==========================================================
    print("\n===== M2: APRIORI ONLY =====")

    unf_dict = extract_uninteresting(df)
    transactions = build_transactions(unf_dict)

    start = time.time()

    apriori_rules, apriori_time = run_apriori(transactions, min_support=0.01)
    apriori_remove = get_items_to_remove(apriori_rules) if apriori_rules is not None else set()

    if apriori_rules is not None:
        print("Apriori Rules:", len(apriori_rules))
        print("Items removed (Apriori):", len(apriori_remove))

    filtered_train_ap = filter_sequences(train, apriori_remove)
    transitions_ap = train_bigram(filtered_train_ap)
    results_ap = evaluate(transitions_ap, test, filtered_train_ap)

    end = time.time()
    ap_total_time = end - start

    print("Apriori:", results_ap, "| Time:", round(ap_total_time, 4), "sec")


    # ==========================================================
    # ================= M3: FP-GROWTH ONLY =====================
    # ==========================================================
    print("\n===== M3: FP-GROWTH ONLY =====")

    start = time.time()

    fp_rules, fp_time = run_fpgrowth(transactions, min_support=0.01)
    fp_remove = get_items_to_remove(fp_rules) if fp_rules is not None else set()

    if fp_rules is not None:
        print("FP Rules:", len(fp_rules))
        print("Items removed (FP):", len(fp_remove))

    filtered_train_fp = filter_sequences(train, fp_remove)
    transitions_fp = train_bigram(filtered_train_fp)
    results_fp = evaluate(transitions_fp, test, filtered_train_fp)

    end = time.time()
    fp_total_time = end - start
    print("FP-Growth:", results_fp, "| Time:", round(fp_total_time, 4), "sec")

    # ==========================================================
    # ====================== SSE PART ==========================
    # ==========================================================
    print("\n===== APPLYING SSE =====")

    short_users, long_users = split_short_long(train, 5)
    print("Short users:", len(short_users),  "|", round(len(short_users)/len(train)*100,2), "%")

    start = time.time()
    reverse_model = train_reverse_model(long_users)
    end = time.time()

    enhanced_short = extend_short_sequences(short_users, reverse_model, k=2)

    train_sse = {}
    train_sse.update(long_users)
    train_sse.update(enhanced_short)

    print("Reverse training time:", round(end - start, 4), "seconds")

    # ==========================================================
    # ================= SSE + APRIORI ==========================
    # ==========================================================
    print("\n===== M4: SSE + APRIORI =====")

    start = time.time()

    filtered_train_sse_ap = filter_sequences(train_sse, apriori_remove)
    transitions_sse_ap = train_bigram(filtered_train_sse_ap)
    results_sse_ap = evaluate(transitions_sse_ap, test, filtered_train_sse_ap)

    end = time.time()
    sse_ap_time = end - start

    print("SSE + Apriori:", results_sse_ap, "| Time:", round(sse_ap_time, 4), "sec")

    # ==========================================================
    # ================= SSE + FP-GROWTH ========================
    # ==========================================================
    print("\n===== M5: SSE + FP-GROWTH =====")

    start = time.time()
    filtered_train_sse_fp = filter_sequences(train_sse, fp_remove)

    transitions_sse_fp = train_bigram(filtered_train_sse_fp)
    results_sse_fp = evaluate(transitions_sse_fp, test, filtered_train_sse_fp)

    end = time.time()
    sse_fp_time = end - start
    print("SSE + Apriori:", results_sse_fp, "| Time:", round(sse_fp_time, 4), "sec")

    # ==========================================================
    # ===================== SUMMARY ============================
    # ==========================================================
    print("\n================ FINAL SUMMARY ================")

    print(f"M1 Baseline        | MRR: {results_base['MRR']:.5f} | "
        f"nDCG: {results_base['nDCG']:.5f} | Time: {baseline_time:.4f}s")

    print(f"M2 Apriori Only    | MRR: {results_ap['MRR']:.5f} | "
        f"nDCG: {results_ap['nDCG']:.5f} | Time: {ap_total_time:.4f}s")

    print(f"M3 FP-Growth Only  | MRR: {results_fp['MRR']:.5f} | "
        f"nDCG: {results_fp['nDCG']:.5f} | Time: {fp_total_time:.4f}s")

    print(f"M4 SSE + Apriori   | MRR: {results_sse_ap['MRR']:.5f} | "
        f"nDCG: {results_sse_ap['nDCG']:.5f} | Time: {sse_ap_time:.4f}s")

    print(f"M5 SSE + FP-Growth | MRR: {results_sse_fp['MRR']:.5f} | "
        f"nDCG: {results_sse_fp['nDCG']:.5f}  | Time: {sse_fp_time:.4f}s")

    print("\nMining Time Comparison:")

    print("Apriori Time:", round(apriori_time, 4))
    print("FP-Growth Time:", round(fp_time, 4))