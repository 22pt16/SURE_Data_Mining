from collections import defaultdict

def extract_sequential_patterns(unf_dict, min_support=5):
    """
    Lightweight Sequential Pattern Mining (Bigram-based)
    """
    seq_patterns = defaultdict(int)

    for user, seq in unf_dict.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            seq_patterns[pair] += 1

    # Filter patterns
    filtered_patterns = {
        pair: count for pair, count in seq_patterns.items()
        if count >= min_support
    }

    return filtered_patterns


def get_spm_items_to_remove(spm_patterns):
    """
    Extract items from sequential patterns
    (removes items appearing in frequent negative transitions)
    """
    remove_items = set()

    for (i, j) in spm_patterns.keys():
        remove_items.add(j)   # remove next item in negative sequence

    return remove_items