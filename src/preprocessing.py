import pandas as pd
import os

def load_and_preprocess(data_path, output_path):
    print("Loading dataset...")

    # MovieLens 100K format
    columns = ["user_id", "item_id", "rating", "timestamp"]

    df = pd.read_csv(
        data_path,
        sep="\t",
        names=columns
    )

    print("Original shape:", df.shape)

    # Sort chronologically
    df = df.sort_values(by=["user_id", "timestamp"])

    # Convert rating to binary preference
    df["liked"] = df["rating"].apply(lambda x: 1 if x >= 3 else 0)

    print("Preprocessed shape:", df.shape)

    # Save processed version
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Saved processed data.")

    return df