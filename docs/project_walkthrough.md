# Project Walkthrough

This file explains the project in the same order you should learn it, run it, and describe it in an interview.

## Step 1: Understand the business problem

The project simulates a law-firm intake workflow. New matters come in from different teams and channels, the data is messy, and leaders need visibility into volume, risk, and response speed.

This is why the project has two layers:

- an analytics pipeline for cleaning and reporting
- a simple machine learning model for predicting delayed assignment
- an NLP text classifier for routing intake summaries
- an entity extraction workflow for pulling structured terms from summaries
- a semantic search workflow for retrieving similar summaries
- a lightweight dashboard for reviewing everything visually

## Step 2: Inspect the raw data

Open `data/raw/legal_intake_sample.csv`.

The dataset includes:

- matter identifiers
- practice area
- intake channel
- requester type
- jurisdiction
- free-text summary
- estimated value
- risk score
- assignment timing

Before writing any ML, you first need to understand what the fields mean and what could be missing or inconsistent.

## Step 3: Run the analytics pipeline

Run:

```bash
python3 project1.py
```

This script does the core data engineering work:

1. Loads the raw CSV
2. Standardizes column names and text values
3. Fills missing numeric and categorical values
4. Creates useful features such as:
   - `priority_flag`
   - `summary_topic`
   - `received_weekday`
   - `assignment_delay_flag`
5. Creates summary tables for reporting
6. Saves artifacts to CSV, JSON, text, and SQLite

## Step 4: Inspect the outputs

After the script runs, look at:

- `outputs/cleaned_legal_intake.csv`
- `outputs/practice_area_summary.csv`
- `outputs/status_summary.csv`
- `outputs/executive_summary.txt`
- `outputs/executive_metrics.json`

This is where you learn how cleaned data becomes business reporting.

## Step 5: Understand the SQLite layer

The pipeline writes tables into `outputs/legal_intake.db`.

That matters because business teams often need analysis in SQL, not only in Python. This repo shows both.

Run queries from:

```bash
sqlite3 outputs/legal_intake.db < sql/report_queries.sql
```

If `sqlite3` is not installed, you can still inspect the tables with Python.

## Step 6: Understand the ML target

The machine learning target in this repo is `assignment_delay_flag`.

- `on_time`: matter assigned in 3 days or fewer
- `delayed`: matter assigned in more than 3 days

This is a reasonable first prediction task because it connects directly to operations and staffing, and it is easier to explain than a vague generic model.

## Step 7: Train the ML baseline

Run:

```bash
python3 train_assignment_delay_model.py
```

This script:

1. Reuses the cleaned dataset from the pipeline
2. Selects structured numeric and categorical features
3. Splits data into train and test sets
4. Builds a preprocessing pipeline with scaling and one-hot encoding
5. Trains a logistic regression classifier
6. Saves metrics, predictions, feature weights, and the model file

## Step 8: Review the ML outputs

After training, inspect:

- `outputs/assignment_delay_model_metrics.json`
- `outputs/assignment_delay_predictions.csv`
- `outputs/assignment_delay_feature_weights.csv`

These files tell you:

- how the model performed on the small test set
- which examples were predicted correctly or incorrectly
- which features influenced the model the most

## Step 9: Train the NLP baseline

Run:

```bash
python3 train_practice_area_text_classifier.py
```

This script:

1. Uses the intake `summary` text as the model input
2. Uses `practice_area` as the target label
3. Converts text into TF-IDF features
4. Trains a logistic regression classifier
5. Uses cross-validation to produce out-of-fold predictions
6. Saves metrics, predictions, top terms, and the model file

## Step 10: Review the NLP outputs

After training, inspect:

- `outputs/practice_area_text_metrics.json`
- `outputs/practice_area_text_predictions.csv`
- `outputs/practice_area_top_terms.csv`

These files show:

- how well summary text predicts the practice area
- which summaries were classified correctly or incorrectly
- which terms were most associated with each class

## Step 11: Run entity extraction

Run:

```bash
python3 extract_legal_entities.py
```

This script:

1. Uses the intake `summary` text
2. Applies rule-based phrase matching
3. Extracts issue, document, party, asset, and workflow terms
4. Saves a row-level extraction table
5. Saves a matter-level record view
6. Saves an entity summary and extraction metrics

## Step 12: Review the entity outputs

After running the extractor, inspect:

- `outputs/legal_entity_extractions.csv`
- `outputs/legal_entity_record_view.csv`
- `outputs/legal_entity_summary.csv`
- `outputs/legal_entity_metrics.json`

These files show:

- which entities were found in each summary
- which categories appear most often
- how structured information can be derived from unstructured text

## Step 13: Run semantic search

Run:

```bash
python3 semantic_summary_search.py --query "data breach client files"
```

This script:

1. Reuses the cleaned summary text
2. Builds a TF-IDF representation
3. Projects it into a small latent semantic space with TruncatedSVD
4. Scores similarity between the query and each matter summary
5. Saves demo search results and search metadata

## Step 14: Review the search outputs

After running the search script, inspect:

- `outputs/summary_search_demo_results.csv`
- `outputs/summary_search_metrics.json`

These files show:

- which summaries are most similar to each example query
- what search method was used
- how a lightweight retrieval workflow can sit on top of the intake corpus

## Step 15: Run the dashboard

Run:

```bash
streamlit run dashboard.py
```

The dashboard brings together:

- executive KPIs
- tabular analytics outputs
- ML and NLP metrics
- entity extraction outputs
- interactive summary search

## Step 16: Learn what to say in an interview

When you describe this project, use this sequence:

1. The business problem
2. The messy data challenge
3. The cleaning and reporting pipeline
4. The operational ML prediction
5. The NLP routing workflow on intake summaries
6. The entity extraction workflow on intake summaries
7. The semantic search workflow over intake summaries
8. The SQL and dashboard reporting outputs
9. The next step: richer legal-text NLP

That order sounds much stronger than just listing tools.

## Step 17: Know the honest limitation

This dataset is synthetic and small, so the metrics are not production-grade. That is fine.

What matters is that you can explain:

- why you chose the prediction target
- how the data was cleaned
- how the model was trained
- what the outputs mean
- what you would improve next

## Best study order from here

1. Run `project1.py` and inspect every output file
2. Read `project1.py` from top to bottom
3. Run the SQL queries and understand each one
4. Run `train_assignment_delay_model.py`
5. Run `train_practice_area_text_classifier.py`
6. Run `extract_legal_entities.py`
7. Run `semantic_summary_search.py`
8. Open the dashboard with `streamlit run dashboard.py`
9. Read the model metrics, feature weights, text predictions, entity outputs, and search results
10. Practice explaining the full workflow out loud
