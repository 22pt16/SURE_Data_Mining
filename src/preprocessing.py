import pandas as pd
import os

def load_and_preprocess(data_path, output_path, threshold=3):
    print("Loading raw dataset...")

    columns = ["user_id", "item_id", "rating", "timestamp"]

    df = pd.read_csv(
        data_path,
        sep="\t",
        names=columns
    )

    print("Original shape:", df.shape)

    # Sort chronologically
    df = df.sort_values(by=["user_id", "timestamp"])

    # Create liked column (baseline threshold)
    df["liked"] = df["rating"].apply(lambda x: 1 if x >= threshold else 0)

    # Save processed version
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Processed dataset saved.")
    return df