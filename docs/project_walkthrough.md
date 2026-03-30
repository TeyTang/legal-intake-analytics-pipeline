# Project Walkthrough

This file explains the project in the same order you should learn it, run it, and describe it in an interview.

## Step 1: Understand the business problem

The project simulates a law-firm intake workflow. New matters come in from different teams and channels, the data is messy, and leaders need visibility into volume, risk, and response speed.

This is why the project has two layers:

- an analytics pipeline for cleaning and reporting
- a simple machine learning model for predicting delayed assignment

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

## Step 9: Learn what to say in an interview

When you describe this project, use this sequence:

1. The business problem
2. The messy data challenge
3. The cleaning and reporting pipeline
4. The operational ML prediction
5. The SQL and stakeholder reporting outputs
6. The next step: NLP on intake summaries

That order sounds much stronger than just listing tools.

## Step 10: Know the honest limitation

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
5. Read the model metrics and feature weights
6. Practice explaining the full workflow out loud
