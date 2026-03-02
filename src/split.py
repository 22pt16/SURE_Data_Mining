def leave_one_out_split(df):
    train = {}
    val = {}
    test = {}

    grouped = df.groupby("user_id")

    for user, group in grouped:
        items = group["item_id"].tolist()

        if len(items) < 3:
            continue

        train[user] = items[:-2]
        val[user] = items[-2]
        test[user] = items[-1]

    return train, val, test


def split_short_long(train_data, threshold_percent=30):

    # Sort users by sequence length
    sorted_users = sorted(train_data.items(), key=lambda x: len(x[1]))

    total_users = len(sorted_users)
    cutoff = int(total_users * threshold_percent / 100)

    # Simulate short version for experiment
    short_users = dict(sorted_users[:cutoff])
    long_users = dict(sorted_users[cutoff:])

    return short_users, long_users