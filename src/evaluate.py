import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml


DATA_PATH = "data/processed.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "metrics/metrics.json"
PARAMS_PATH = "params.yaml"


def evaluate_model():
    # Load parameters
    with open(PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    # Load processed data
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Use the same train/test split configuration
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Load trained model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted")
    }

    # Save metrics
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Evaluation completed successfully.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    evaluate_model()