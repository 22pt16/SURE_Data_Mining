import time
from src.preprocessing import load_and_preprocess
from src.split import leave_one_out_split, split_short_long
from src.reverse_model import train_reverse_model, predict_prior_items
from src.extension import extend_short_sequences

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

    #Extending Short Sequences
    start = time.time()
    reverse_model = train_reverse_model(long_users)
    end = time.time()

    print("Reverse model training time:", end - start, "seconds")
    print("\nReverse model trained for Short Sequence Extension.")
    example_item = list(reverse_model.keys())[0]
    print("\nExample reverse transitions for item:", example_item)
    print("Backward Dependency:",reverse_model[example_item])

    enhanced_short = extend_short_sequences(short_users, reverse_model, k=2)

    #Example Comparison
    example_user = list(short_users.keys())[0]

    print("\nExample\nOriginal short sequence:", short_users[example_user])
    print("Enhanced short sequence:", enhanced_short[example_user])

    #Extension Impact Stats
    total_before = sum(len(seq) for seq in short_users.values())
    total_after = sum(len(seq) for seq in enhanced_short.values())

    print("\nOverall SSE Extension Impact Stats:")
    print("Average short length before:", round(total_before / len(short_users)))
    print("Average short length after:", round(total_after / len(short_users)))

    #Pass to Forward Model for Prediction
    combined_train = {}
    combined_train.update(long_users)
    combined_train.update(enhanced_short)

    print("Combined training users:", len(combined_train))
