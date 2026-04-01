from __future__ import annotations

import json
import re

import pandas as pd

from project1 import OUTPUT_DIR, RAW_DATA_PATH, clean_data, ensure_directories, load_data


EXTRACTIONS_PATH = OUTPUT_DIR / "legal_entity_extractions.csv"
RECORD_VIEW_PATH = OUTPUT_DIR / "legal_entity_record_view.csv"
SUMMARY_PATH = OUTPUT_DIR / "legal_entity_summary.csv"
METRICS_PATH = OUTPUT_DIR / "legal_entity_metrics.json"

ENTITY_PATTERNS: list[dict[str, str]] = [
    {"entity_type": "issue_type", "entity_value": "data breach"},
    {"entity_type": "issue_type", "entity_value": "contract dispute"},
    {"entity_type": "issue_type", "entity_value": "insurance coverage"},
    {"entity_type": "issue_type", "entity_value": "records retention"},
    {"entity_type": "issue_type", "entity_value": "billing controls"},
    {"entity_type": "issue_type", "entity_value": "cyber incident"},
    {"entity_type": "issue_type", "entity_value": "privacy notification"},
    {"entity_type": "issue_type", "entity_value": "due diligence"},
    {"entity_type": "issue_type", "entity_value": "harassment"},
    {"entity_type": "issue_type", "entity_value": "acquisition"},
    {"entity_type": "issue_type", "entity_value": "termination"},
    {"entity_type": "issue_type", "entity_value": "severance"},
    {"entity_type": "issue_type", "entity_value": "dispute"},
    {"entity_type": "document_type", "entity_value": "master services agreement"},
    {"entity_type": "document_type", "entity_value": "agreement"},
    {"entity_type": "document_type", "entity_value": "contract"},
    {"entity_type": "document_type", "entity_value": "complaint"},
    {"entity_type": "document_type", "entity_value": "inquiry"},
    {"entity_type": "document_type", "entity_value": "package"},
    {"entity_type": "document_type", "entity_value": "notification"},
    {"entity_type": "document_type", "entity_value": "claim"},
    {"entity_type": "document_type", "entity_value": "audit"},
    {"entity_type": "document_type", "entity_value": "response"},
    {"entity_type": "party_reference", "entity_value": "vendor"},
    {"entity_type": "party_reference", "entity_value": "employee"},
    {"entity_type": "party_reference", "entity_value": "client"},
    {"entity_type": "party_reference", "entity_value": "acquisition target"},
    {"entity_type": "asset_reference", "entity_value": "software licensing"},
    {"entity_type": "asset_reference", "entity_value": "client files"},
    {"entity_type": "workflow_action", "entity_value": "renewal"},
    {"entity_type": "workflow_action", "entity_value": "review"},
    {"entity_type": "workflow_action", "entity_value": "negotiation"},
    {"entity_type": "workflow_action", "entity_value": "triage"},
    {"entity_type": "workflow_action", "entity_value": "analysis"},
    {"entity_type": "workflow_action", "entity_value": "update"},
    {"entity_type": "workflow_action", "entity_value": "response"},
]


def load_entity_data() -> pd.DataFrame:
    return clean_data(load_data(RAW_DATA_PATH))


def overlaps_existing(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end in spans)


def extract_entities_from_text(text: str) -> list[dict[str, str | int]]:
    lowered_text = text.lower()
    used_spans: list[tuple[int, int]] = []
    matches: list[dict[str, str | int]] = []

    for pattern in ENTITY_PATTERNS:
        phrase = pattern["entity_value"]
        regex = re.compile(rf"\b{re.escape(phrase)}\b")
        for match in regex.finditer(lowered_text):
            start, end = match.span()
            if overlaps_existing(start, end, used_spans):
                continue

            matches.append(
                {
                    "entity_type": pattern["entity_type"],
                    "entity_value": phrase,
                    "matched_text": text[start:end],
                    "start_char": start,
                    "end_char": end,
                }
            )
            used_spans.append((start, end))

    return matches


def build_extraction_tables(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    extraction_rows: list[dict[str, str | int]] = []
    record_rows: list[dict[str, str | int | bool]] = []

    entity_types = [
        "issue_type",
        "document_type",
        "party_reference",
        "asset_reference",
        "workflow_action",
    ]

    for row in cleaned_df.itertuples(index=False):
        extractions = extract_entities_from_text(row.summary)

        for extraction in extractions:
            extraction_rows.append(
                {
                    "matter_id": row.matter_id,
                    "practice_area": row.practice_area,
                    "summary": row.summary,
                    "entity_type": extraction["entity_type"],
                    "entity_value": extraction["entity_value"],
                    "matched_text": extraction["matched_text"],
                    "start_char": extraction["start_char"],
                    "end_char": extraction["end_char"],
                }
            )

        grouped_values: dict[str, list[str]] = {entity_type: [] for entity_type in entity_types}
        for extraction in extractions:
            grouped_values[str(extraction["entity_type"])].append(str(extraction["entity_value"]))

        record_rows.append(
            {
                "matter_id": row.matter_id,
                "practice_area": row.practice_area,
                "summary": row.summary,
                "has_entities": bool(extractions),
                "entity_count": len(extractions),
                "issue_entities": "; ".join(grouped_values["issue_type"]),
                "document_entities": "; ".join(grouped_values["document_type"]),
                "party_entities": "; ".join(grouped_values["party_reference"]),
                "asset_entities": "; ".join(grouped_values["asset_reference"]),
                "action_entities": "; ".join(grouped_values["workflow_action"]),
            }
        )

    extractions_df = pd.DataFrame(extraction_rows)
    record_view_df = pd.DataFrame(record_rows)
    return extractions_df, record_view_df


def save_outputs(extractions_df: pd.DataFrame, record_view_df: pd.DataFrame) -> None:
    if extractions_df.empty:
        pd.DataFrame(
            columns=[
                "matter_id",
                "practice_area",
                "summary",
                "entity_type",
                "entity_value",
                "matched_text",
                "start_char",
                "end_char",
            ]
        ).to_csv(EXTRACTIONS_PATH, index=False)
        pd.DataFrame(columns=["entity_type", "entity_value", "mention_count"]).to_csv(
            SUMMARY_PATH,
            index=False,
        )
    else:
        extractions_df.to_csv(EXTRACTIONS_PATH, index=False)
        (
            extractions_df.groupby(["entity_type", "entity_value"])
            .size()
            .reset_index(name="mention_count")
            .sort_values(["mention_count", "entity_type", "entity_value"], ascending=[False, True, True])
            .to_csv(SUMMARY_PATH, index=False)
        )

    record_view_df.to_csv(RECORD_VIEW_PATH, index=False)


def save_metrics(extractions_df: pd.DataFrame, record_view_df: pd.DataFrame) -> None:
    entity_type_counts = (
        extractions_df["entity_type"].value_counts().sort_index().to_dict()
        if not extractions_df.empty
        else {}
    )
    top_entities = (
        extractions_df["entity_value"].value_counts().head(5).to_dict()
        if not extractions_df.empty
        else {}
    )

    metrics = {
        "total_matters": int(len(record_view_df)),
        "matters_with_entities": int(record_view_df["has_entities"].sum()),
        "matters_without_entities": int((~record_view_df["has_entities"]).sum()),
        "total_entity_mentions": int(len(extractions_df)),
        "unique_entity_values": int(extractions_df["entity_value"].nunique()) if not extractions_df.empty else 0,
        "average_entities_per_matter": round(float(record_view_df["entity_count"].mean()), 2),
        "entity_type_counts": entity_type_counts,
        "top_entity_values": top_entities,
        "method": "Rule-based phrase matching on intake summaries",
        "note": "This is an explainable baseline entity extraction workflow for a small synthetic dataset.",
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    ensure_directories()

    cleaned_df = load_entity_data()
    extractions_df, record_view_df = build_extraction_tables(cleaned_df)

    save_outputs(extractions_df, record_view_df)
    save_metrics(extractions_df, record_view_df)

    print("Legal entity extraction completed.")
    print(f"Rows processed: {len(cleaned_df)}")
    print(f"Entity mentions extracted: {len(extractions_df)}")
    print(f"Saved extractions: {EXTRACTIONS_PATH}")


if __name__ == "__main__":
    main()
