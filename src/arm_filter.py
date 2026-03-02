import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.frequent_patterns import apriori
import time


def extract_uninteresting(df):
    """
    Extract uninteresting items per user
    """
    unf = df[df["liked"] == 0]
    user_unf = unf.groupby("user_id")["item_id"].apply(list)
    return user_unf.to_dict()


def build_transactions(unf_dict):
    """
    Convert dictionary to list of transactions
    """
    return list(unf_dict.values())

def run_apriori(transactions, min_support=0.01, min_conf=0.3):
    """
    Run Apriori and generate association rules.
    Returns rules + execution time.
    """

    if len(transactions) == 0:
        return None, 0

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    start_time = time.time()

    freq_items = apriori(df_encoded, min_support=min_support, use_colnames=True)

    if freq_items.empty:
        return None, time.time() - start_time

    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)

    exec_time = time.time() - start_time

    return rules, exec_time

def run_fpgrowth(transactions, min_support=0.01, min_conf=0.3):
    """
    Run FP-Growth and generate rules.
    Returns rules + execution time.
    """

    if len(transactions) == 0:
        return None, 0

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    start_time = time.time()

    freq_items = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

    if freq_items.empty:
        return None, time.time() - start_time

    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)

    exec_time = time.time() - start_time

    return rules, exec_time


def get_items_to_remove(rules, min_conf=0.6, min_lift=1.0):

    remove_items = set()

    for _, row in rules.iterrows():

        if row["confidence"] >= min_conf and row["lift"] >= min_lift:

            for item in row["consequents"]:
                remove_items.add(item)

    return remove_items


def filter_sequences(user_sequences, items_to_remove):
    """
    Remove uninteresting items from sequences
    """
    filtered = {}

    for user, seq in user_sequences.items():
        filtered[user] = [item for item in seq if item not in items_to_remove]

    return filtered