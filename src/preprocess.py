import pandas as pd


INPUT_PATH = "data/dataset.csv"
OUTPUT_PATH = "data/processed.csv"


def preprocess_data():
    df = pd.read_csv(INPUT_PATH)

    # Remove the text version of the target.
    df = df.drop(columns=["target_name"])

    # Save the processed dataset.
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Processed dataset saved to {OUTPUT_PATH}")
    print(f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    preprocess_data()