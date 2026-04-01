from __future__ import annotations

import json
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from project1 import OUTPUT_DIR, RAW_DATA_PATH, clean_data, ensure_directories, load_data


MODEL_PATH = OUTPUT_DIR / "practice_area_text_classifier.pkl"
METRICS_PATH = OUTPUT_DIR / "practice_area_text_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "practice_area_text_predictions.csv"
TOP_TERMS_PATH = OUTPUT_DIR / "practice_area_top_terms.csv"

TEXT_COLUMN = "summary"
TARGET_COLUMN = "practice_area"


def load_text_classification_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cleaned_df = clean_data(load_data(RAW_DATA_PATH))
    text = cleaned_df[TEXT_COLUMN].fillna("").astype(str)
    target = cleaned_df[TARGET_COLUMN].astype(str)
    return cleaned_df, text, target


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
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


def save_metrics(actual: pd.Series, predicted: pd.Series, labels: list[str]) -> None:
    metrics = {
        "rows": int(len(actual)),
        "cross_validation": "StratifiedKFold(n_splits=2, shuffle=True, random_state=42)",
        "accuracy": round(float(accuracy_score(actual, predicted)), 3),
        "macro_precision": round(
            float(precision_score(actual, predicted, average="macro", zero_division=0)),
            3,
        ),
        "macro_recall": round(
            float(recall_score(actual, predicted, average="macro", zero_division=0)),
            3,
        ),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 3),
        "labels": labels,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "note": "Metrics are illustrative because the dataset is small and synthetic.",
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_predictions(
    cleaned_df: pd.DataFrame,
    predicted: pd.Series,
    probabilities: pd.DataFrame,
) -> None:
    results = cleaned_df[
        ["matter_id", "summary", "practice_area", "summary_topic", "assigned_team"]
    ].copy()
    results["predicted_practice_area"] = predicted
    results["prediction_confidence"] = probabilities.max(axis=1).round(3)
    results["predicted_correctly"] = results["practice_area"] == results["predicted_practice_area"]
    results.to_csv(PREDICTIONS_PATH, index=False)


def save_top_terms(model: Pipeline) -> None:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    rows: list[dict[str, str | int | float]] = []
    for class_index, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        top_feature_indexes = coefficients.argsort()[-10:][::-1]
        for rank, feature_index in enumerate(top_feature_indexes, start=1):
            rows.append(
                {
                    "practice_area": label,
                    "rank": rank,
                    "term": feature_names[feature_index],
                    "coefficient": round(float(coefficients[feature_index]), 4),
                }
            )

    pd.DataFrame(rows).to_csv(TOP_TERMS_PATH, index=False)


def save_model(model: Pipeline) -> None:
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)


def main() -> None:
    ensure_directories()

    cleaned_df, text, target = load_text_classification_data()
    labels = sorted(target.unique().tolist())
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    pipeline = build_pipeline()
    predicted = pd.Series(cross_val_predict(pipeline, text, target, cv=cv), index=cleaned_df.index)
    probability_values = cross_val_predict(
        pipeline,
        text,
        target,
        cv=cv,
        method="predict_proba",
    )
    probabilities = pd.DataFrame(probability_values, columns=labels, index=cleaned_df.index)

    save_metrics(target, predicted, labels)
    save_predictions(cleaned_df, predicted, probabilities)

    final_model = build_pipeline()
    final_model.fit(text, target)
    save_top_terms(final_model)
    save_model(final_model)

    print("Practice-area NLP classification completed.")
    print(f"Rows processed: {len(cleaned_df)}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
