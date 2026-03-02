from src.preprocessing import load_and_preprocess
from src.split import leave_one_out_split, split_short_long

if __name__ == "__main__":
    raw_path = "data/raw/u.data"
    processed_path = "data/processed/processed.csv"

    #Data Preprocessing
    df = load_and_preprocess(raw_path, processed_path)
    print(df.head())

    #User Sessions Split
    train, val, test = leave_one_out_split(df)

    short_users, long_users = split_short_long(train, 5)

    print("Total users:", len(train))
    print("Short users:", len(short_users))
    print("Long users:", len(long_users))