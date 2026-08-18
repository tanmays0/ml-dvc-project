import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier


INPUT_PATH = "data/processed.csv"
MODEL_PATH = "models/model.pkl"
PARAMS_PATH = "params.yaml"


def train_model():
    # Load parameters
    with open(PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]

    # Load processed data
    df = pd.read_csv(INPUT_PATH)

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=params["train"]["random_state"]
    )

    model.fit(X, y)

    # Save the trained model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("Model trained successfully.")
    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()