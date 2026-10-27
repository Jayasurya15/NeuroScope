import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = "spiral/hw_dataset"

COLUMNS = [
    "X", "Y", "Z", "Pressure",
    "GripAngle", "Timestamp", "TestID"
]


def extract_features(file_path):
    try:
        data = pd.read_csv(
            file_path,
            sep=";",
            header=None,
            names=COLUMNS
        )

        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna(subset=["X", "Y"])

        if len(data) < 10:
            return None

        features = {}

        for column in ["X", "Y", "Z", "Pressure", "GripAngle"]:
            values = data[column].dropna()

            if len(values) > 0:
                features[column + "_mean"] = values.mean()
                features[column + "_std"] = values.std()
                features[column + "_min"] = values.min()
                features[column + "_max"] = values.max()

        x = data["X"].to_numpy(dtype=float)
        y = data["Y"].to_numpy(dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)

        features["point_count"] = len(data)
        features["path_length"] = distances.sum()
        features["mean_step"] = distances.mean()
        features["std_step"] = distances.std()
        features["x_range"] = np.ptp(x)
        features["y_range"] = np.ptp(y)

        angles = np.arctan2(dy, dx)
        angle_changes = np.diff(np.unwrap(angles))

        features["mean_angle_change"] = (
            np.mean(np.abs(angle_changes))
            if len(angle_changes) else 0
        )

        timestamps = data["Timestamp"].to_numpy(dtype=float)
        time_diffs = np.diff(timestamps)
        valid = time_diffs > 0

        if np.any(valid):
            speeds = distances[valid] / time_diffs[valid]
            features["mean_speed"] = speeds.mean()
            features["std_speed"] = speeds.std()
        else:
            features["mean_speed"] = 0
            features["std_speed"] = 0

        return features

    except Exception as error:
        print("Could not process:", file_path, error)
        return None


def train_model():
    records = []

    for label, folder in [(0, "control"), (1, "parkinson")]:
        folder_path = os.path.join(DATA_DIR, folder)

        for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
            features = extract_features(file_path)

            if features is not None:
                features["label"] = label
                features["file"] = os.path.basename(file_path)
                records.append(features)

    if len(records) < 10:
        raise RuntimeError(
            "Too few recordings found. Check the dataset folders."
        )

    dataset = pd.DataFrame(records)

    print("Total recordings:", len(dataset))
    print("\nClass counts:")
    print(dataset["label"].map({
        0: "Healthy control",
        1: "Parkinson's"
    }).value_counts())

    X = dataset.drop(columns=["label", "file"])
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(
        "\nPreliminary accuracy:",
        round(accuracy_score(y_test, predictions) * 100, 2),
        "%"
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Healthy control", "Parkinson's"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    joblib.dump({
        "model": model,
        "features": X.columns.tolist()
    }, "spiral_model.pkl")

    print("\nModel saved as spiral_model.pkl")


if __name__ == "__main__":
    train_model()
