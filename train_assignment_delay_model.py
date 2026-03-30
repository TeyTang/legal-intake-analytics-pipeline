from __future__ import annotations

import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project1 import OUTPUT_DIR, RAW_DATA_PATH, clean_data, ensure_directories, load_data


MODEL_PATH = OUTPUT_DIR / "assignment_delay_model.pkl"
METRICS_PATH = OUTPUT_DIR / "assignment_delay_model_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "assignment_delay_predictions.csv"
FEATURE_WEIGHTS_PATH = OUTPUT_DIR / "assignment_delay_feature_weights.csv"

NUMERIC_FEATURES = ["estimated_value_usd", "risk_score", "summary_length"]
CATEGORICAL_FEATURES = [
    "practice_area",
    "intake_channel",
    "requester_type",
    "jurisdiction",
    "assigned_team",
    "summary_topic",
    "received_weekday",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cleaned_df = clean_data(load_data(RAW_DATA_PATH))
    features = cleaned_df[FEATURE_COLUMNS].copy()
    target = cleaned_df["assignment_delay_flag"].eq("delayed").astype(int)
    return cleaned_df, features, target


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def save_model_metrics(y_test: pd.Series, predicted: pd.Series) -> None:
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()

    metrics = {
        "test_rows": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 3),
        "precision_delayed": round(float(precision_score(y_test, predicted, zero_division=0)), 3),
        "recall_delayed": round(float(recall_score(y_test, predicted, zero_division=0)), 3),
        "f1_delayed": round(float(f1_score(y_test, predicted, zero_division=0)), 3),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "note": "Metrics are illustrative because the dataset is small and synthetic.",
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_feature_weights(model: Pipeline) -> None:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]

    weights = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": coefficients.astype(float).copy(),
        }
    )
    weights["absolute_coefficient"] = weights["coefficient"].abs()
    weights = weights.sort_values("absolute_coefficient", ascending=False)
    weights.to_csv(FEATURE_WEIGHTS_PATH, index=False)


def save_predictions(
    cleaned_df: pd.DataFrame,
    test_index: pd.Index,
    y_test: pd.Series,
    predicted: pd.Series,
    delayed_probability: pd.Series,
) -> None:
    results = cleaned_df.loc[
        test_index,
        [
            "matter_id",
            "practice_area",
            "jurisdiction",
            "status",
            "days_to_assign",
            "assignment_delay_flag",
            "summary_topic",
        ],
    ].copy()

    results["actual_label"] = y_test.map({0: "on_time", 1: "delayed"})
    results["predicted_label"] = predicted.map({0: "on_time", 1: "delayed"})
    results["predicted_probability_delayed"] = delayed_probability.round(3)

    results.to_csv(PREDICTIONS_PATH, index=False)


def save_model(model: Pipeline) -> None:
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)


def main() -> None:
    ensure_directories()

    cleaned_df, features, target = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.33,
        random_state=42,
        stratify=target,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predicted = pd.Series(model.predict(X_test), index=X_test.index)
    delayed_probability = pd.Series(
        model.predict_proba(X_test)[:, 1],
        index=X_test.index,
    )

    save_model_metrics(y_test, predicted)
    save_feature_weights(model)
    save_predictions(cleaned_df, X_test.index, y_test, predicted, delayed_probability)
    save_model(model)

    print("Assignment delay model training completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
