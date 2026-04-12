from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from project1 import OUTPUT_DIR, RAW_DATA_PATH, clean_data, ensure_directories, load_data


SEARCH_RESULTS_PATH = OUTPUT_DIR / "summary_search_demo_results.csv"
SEARCH_METRICS_PATH = OUTPUT_DIR / "summary_search_metrics.json"

DEFAULT_DEMO_QUERIES = [
    "vendor contract renewal",
    "employee complaint review",
    "data breach client files",
    "billing audit inquiry",
]


@dataclass
class SearchArtifacts:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: object
    method: str
    latent_components: int


def load_search_data() -> pd.DataFrame:
    cleaned_df = clean_data(load_data(RAW_DATA_PATH))
    search_df = cleaned_df[
        [
            "matter_id",
            "practice_area",
            "summary",
            "summary_topic",
            "assigned_team",
            "priority_flag",
            "status",
            "risk_score",
        ]
    ].copy()
    search_df["search_text"] = (
        search_df["summary"].fillna("")
        + " "
        + search_df["summary_topic"].fillna("")
        + " "
        + search_df["assigned_team"].fillna("")
    ).str.strip()
    return search_df


def build_search_artifacts(df: pd.DataFrame) -> SearchArtifacts:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(df["search_text"])

    max_components = min(tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1, 3)
    if max_components >= 2:
        svd = TruncatedSVD(n_components=max_components, algorithm="arpack", random_state=42)
        normalizer = Normalizer(copy=False)
        latent_matrix = svd.fit_transform(tfidf_matrix)
        normalized_matrix = normalizer.fit_transform(latent_matrix)
        vectorizer.svd = svd  # type: ignore[attr-defined]
        vectorizer.normalizer = normalizer  # type: ignore[attr-defined]
        return SearchArtifacts(
            df=df,
            vectorizer=vectorizer,
            matrix=normalized_matrix,
            method="Latent semantic search using TF-IDF + TruncatedSVD",
            latent_components=max_components,
        )

    normalized_matrix = Normalizer(copy=False).fit_transform(tfidf_matrix)
    return SearchArtifacts(
        df=df,
        vectorizer=vectorizer,
        matrix=normalized_matrix,
        method="Cosine similarity search using TF-IDF",
        latent_components=0,
    )


def transform_query(query: str, artifacts: SearchArtifacts):
    query_vector = artifacts.vectorizer.transform([query])

    if artifacts.latent_components >= 2:
        svd = artifacts.vectorizer.svd  # type: ignore[attr-defined]
        normalizer = artifacts.vectorizer.normalizer  # type: ignore[attr-defined]
        query_vector = normalizer.transform(svd.transform(query_vector))

    return query_vector


def search_summaries(query: str, artifacts: SearchArtifacts, top_k: int = 5) -> pd.DataFrame:
    normalized_query = query.strip()
    if not normalized_query:
        return pd.DataFrame(
            columns=[
                "rank",
                "similarity_score",
                "matter_id",
                "practice_area",
                "summary",
                "summary_topic",
                "assigned_team",
                "priority_flag",
                "status",
                "risk_score",
            ]
        )

    query_vector = transform_query(normalized_query, artifacts)
    similarities = cosine_similarity(query_vector, artifacts.matrix).flatten()
    top_indexes = similarities.argsort()[::-1][:top_k]

    results = artifacts.df.iloc[top_indexes].copy()
    results.insert(0, "rank", range(1, len(results) + 1))
    results.insert(1, "similarity_score", similarities[top_indexes].round(3))

    return results[
        [
            "rank",
            "similarity_score",
            "matter_id",
            "practice_area",
            "summary",
            "summary_topic",
            "assigned_team",
            "priority_flag",
            "status",
            "risk_score",
        ]
    ].reset_index(drop=True)


def save_demo_results(artifacts: SearchArtifacts, queries: list[str], top_k: int) -> pd.DataFrame:
    result_frames: list[pd.DataFrame] = []
    for query in queries:
        search_results = search_summaries(query, artifacts, top_k=top_k)
        search_results.insert(0, "query", query)
        result_frames.append(search_results)

    demo_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    demo_results.to_csv(SEARCH_RESULTS_PATH, index=False)
    return demo_results


def save_metrics(artifacts: SearchArtifacts, queries: list[str], top_k: int) -> None:
    metrics = {
        "rows_indexed": int(len(artifacts.df)),
        "vectorizer_vocabulary_size": int(len(artifacts.vectorizer.vocabulary_)),
        "search_method": artifacts.method,
        "latent_components": int(artifacts.latent_components),
        "demo_queries": queries,
        "top_k_per_query": int(top_k),
        "note": "This is a lightweight semantic-style search baseline for a small synthetic legal-intake corpus.",
    }

    with open(SEARCH_METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic-style search over intake summaries.")
    parser.add_argument("--query", type=str, help="Optional query to search interactively from the command line.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return per query.")
    args = parser.parse_args()

    ensure_directories()

    search_df = load_search_data()
    artifacts = build_search_artifacts(search_df)
    save_demo_results(artifacts, DEFAULT_DEMO_QUERIES, top_k=args.top_k)
    save_metrics(artifacts, DEFAULT_DEMO_QUERIES, top_k=args.top_k)

    print("Summary search indexing completed.")
    print(f"Rows indexed: {len(search_df)}")
    print(f"Search method: {artifacts.method}")
    print(f"Saved demo results: {SEARCH_RESULTS_PATH}")

    if args.query:
        print()
        print(f"Query: {args.query}")
        print(search_summaries(args.query, artifacts, top_k=args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
