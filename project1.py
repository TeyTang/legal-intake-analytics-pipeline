from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "legal_intake_sample.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
DATABASE_PATH = OUTPUT_DIR / "legal_intake.db"


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned.columns = (
        cleaned.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    for column in ["practice_area", "intake_channel", "assigned_team", "status"]:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip()

    cleaned["practice_area"] = cleaned["practice_area"].replace(
        {
            "Corp": "Corporate",
            "corp": "Corporate",
            "Litigation ": "Litigation",
            "employment ": "Employment",
        }
    )

    cleaned["status"] = cleaned["status"].str.lower().replace(
        {
            "open ": "open",
            "in review": "under_review",
            "under review": "under_review",
            "closed ": "closed",
        }
    )

    cleaned["intake_channel"] = cleaned["intake_channel"].str.lower().replace(
        {
            "email ": "email",
            "portal ": "portal",
        }
    )

    cleaned["received_date"] = pd.to_datetime(
        cleaned["received_date"], errors="coerce"
    )
    cleaned["estimated_value_usd"] = pd.to_numeric(
        cleaned["estimated_value_usd"], errors="coerce"
    )
    cleaned["risk_score"] = pd.to_numeric(cleaned["risk_score"], errors="coerce")
    cleaned["days_to_assign"] = pd.to_numeric(
        cleaned["days_to_assign"], errors="coerce"
    )

    cleaned["estimated_value_usd"] = cleaned["estimated_value_usd"].fillna(
        cleaned["estimated_value_usd"].median()
    )
    cleaned["risk_score"] = cleaned["risk_score"].fillna(cleaned["risk_score"].median())
    cleaned["days_to_assign"] = cleaned["days_to_assign"].fillna(
        cleaned["days_to_assign"].median()
    )

    cleaned["summary"] = cleaned["summary"].fillna("").astype(str).str.strip()
    cleaned["requester_type"] = (
        cleaned["requester_type"].fillna("Unknown").astype(str).str.strip()
    )
    cleaned["jurisdiction"] = (
        cleaned["jurisdiction"].fillna("Unknown").astype(str).str.upper().str.strip()
    )
    cleaned["received_weekday"] = cleaned["received_date"].dt.day_name()

    cleaned["is_high_value"] = cleaned["estimated_value_usd"] >= 250000
    cleaned["is_high_risk"] = cleaned["risk_score"] >= 7
    cleaned["summary_length"] = cleaned["summary"].str.split().str.len()

    cleaned["priority_flag"] = "standard"
    cleaned.loc[
        cleaned["is_high_value"] | cleaned["is_high_risk"], "priority_flag"
    ] = "priority"
    cleaned["assignment_delay_flag"] = "on_time"
    cleaned.loc[cleaned["days_to_assign"] > 3, "assignment_delay_flag"] = "delayed"

    cleaned["summary_topic"] = "general"
    keyword_map = {
        "privacy": ["privacy", "breach", "cyber", "data"],
        "employment": ["employee", "termination", "harassment", "wage"],
        "contracts": ["contract", "vendor", "agreement", "renewal"],
        "litigation": ["dispute", "claim", "lawsuit", "complaint"],
    }

    for topic, keywords in keyword_map.items():
        mask = cleaned["summary"].str.lower().apply(
            lambda text: any(keyword in text for keyword in keywords)
        )
        cleaned.loc[mask, "summary_topic"] = topic

    cleaned = cleaned.sort_values("received_date").reset_index(drop=True)

    return cleaned


def build_practice_area_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("practice_area", dropna=False)
        .agg(
            matter_count=("matter_id", "count"),
            avg_estimated_value_usd=("estimated_value_usd", "mean"),
            avg_risk_score=("risk_score", "mean"),
            avg_days_to_assign=("days_to_assign", "mean"),
            priority_matters=("priority_flag", lambda s: (s == "priority").sum()),
        )
        .reset_index()
        .sort_values(["matter_count", "avg_estimated_value_usd"], ascending=[False, False])
    )


def build_status_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["status", "priority_flag"], dropna=False)
        .agg(matter_count=("matter_id", "count"))
        .reset_index()
        .sort_values(["matter_count", "status"], ascending=[False, True])
    )


def build_metrics(df: pd.DataFrame) -> dict[str, float | int | str]:
    top_practice_area = (
        df["practice_area"].mode().iat[0] if not df["practice_area"].mode().empty else "Unknown"
    )

    return {
        "total_matters": int(df["matter_id"].nunique()),
        "priority_matters": int((df["priority_flag"] == "priority").sum()),
        "open_matters": int((df["status"] == "open").sum()),
        "average_risk_score": round(float(df["risk_score"].mean()), 2),
        "average_days_to_assign": round(float(df["days_to_assign"].mean()), 2),
        "top_practice_area": top_practice_area,
        "high_value_matters": int(df["is_high_value"].sum()),
        "delayed_assignment_matters": int((df["assignment_delay_flag"] == "delayed").sum()),
    }


def save_outputs(
    cleaned_df: pd.DataFrame,
    practice_area_summary: pd.DataFrame,
    status_summary: pd.DataFrame,
    metrics: dict[str, float | int | str],
) -> None:
    cleaned_df.to_csv(OUTPUT_DIR / "cleaned_legal_intake.csv", index=False)
    practice_area_summary.to_csv(OUTPUT_DIR / "practice_area_summary.csv", index=False)
    status_summary.to_csv(OUTPUT_DIR / "status_summary.csv", index=False)

    with open(OUTPUT_DIR / "executive_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    summary_lines = [
        "Legal Intake Analytics Executive Summary",
        f"Total matters analyzed: {metrics['total_matters']}",
        f"Priority matters identified: {metrics['priority_matters']}",
        f"Open matters: {metrics['open_matters']}",
        f"Average risk score: {metrics['average_risk_score']}",
        f"Average days to assign: {metrics['average_days_to_assign']}",
        f"Top practice area by volume: {metrics['top_practice_area']}",
        f"High-value matters: {metrics['high_value_matters']}",
        f"Delayed assignments: {metrics['delayed_assignment_matters']}",
    ]
    (OUTPUT_DIR / "executive_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )


def save_to_database(
    cleaned_df: pd.DataFrame,
    practice_area_summary: pd.DataFrame,
    status_summary: pd.DataFrame,
) -> None:
    with sqlite3.connect(DATABASE_PATH) as conn:
        cleaned_df.to_sql("legal_matters_cleaned", conn, if_exists="replace", index=False)
        practice_area_summary.to_sql(
            "practice_area_summary", conn, if_exists="replace", index=False
        )
        status_summary.to_sql("status_summary", conn, if_exists="replace", index=False)


def main() -> None:
    ensure_directories()

    raw_df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    practice_area_summary = build_practice_area_summary(cleaned_df)
    status_summary = build_status_summary(cleaned_df)
    metrics = build_metrics(cleaned_df)

    save_outputs(cleaned_df, practice_area_summary, status_summary, metrics)
    save_to_database(cleaned_df, practice_area_summary, status_summary)

    print("Legal intake pipeline completed.")
    print(f"Rows processed: {len(cleaned_df)}")
    print(f"SQLite database: {DATABASE_PATH}")
    print(f"Top practice area: {metrics['top_practice_area']}")
    print(f"Priority matters: {metrics['priority_matters']}")


if __name__ == "__main__":
    main()
