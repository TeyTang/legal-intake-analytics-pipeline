# Legal Intake Analytics Pipeline

Business-oriented AI/ML portfolio project that simulates a law-firm intake workflow using Python, pandas, SQL, SQLite, and an explainable machine learning baseline.

## Project Snapshot

- Built an end-to-end data pipeline for messy legal intake records
- Cleaned and standardized structured and text-adjacent fields
- Generated stakeholder-facing KPI summaries and SQL-ready tables
- Trained a logistic regression model to predict delayed assignment
- Added an NLP text classifier that routes intake summaries to practice areas
- Documented the workflow for both technical review and interview prep

## Why this project exists

This repo is designed to demonstrate the kind of applied work expected in a business-facing AI/ML role:

- data wrangling and preprocessing
- SQL and database usage
- operational reporting
- explainable baseline modeling
- documentation for technical and non-technical audiences

It is intentionally grounded in a law-firm style intake workflow because that aligns with the target job context.

## Business Problem

Legal and operations teams receive intake matters from multiple channels and stakeholders. In practice, those records are often inconsistent, partially missing, and difficult to analyze quickly.

This project shows how to:

1. clean intake data into a consistent analytical table
2. create business-friendly features such as priority and delay flags
3. produce summary outputs for leadership reporting
4. store clean tables in SQLite for SQL analysis
5. train a simple model to forecast delayed assignment
6. classify intake summaries into practice areas using NLP

## Architecture

```text
Raw CSV
  -> pandas cleaning and feature engineering
  -> reporting tables and executive summary
  -> SQLite database for SQL analysis
  -> ML preprocessing pipeline
  -> logistic regression delay prediction
  -> TF-IDF text vectorization + practice-area classification
  -> saved metrics, predictions, and feature weights
```

## Tech Stack

- Python
- pandas
- SQLite / SQL
- scikit-learn
- pathlib
- JSON / CSV artifacts

## Current Results

From the current synthetic sample dataset:

- Total matters analyzed: `12`
- Priority matters identified: `9`
- Open matters: `6`
- Average risk score: `6.83`
- Average days to assign: `3.42`
- Top practice area by volume: `Corporate`
- High-value matters: `7`
- Delayed assignments: `5`

Current ML evaluation snapshot:

- Test rows: `4`
- Accuracy: `0.50`
- Precision on delayed class: `0.50`
- Recall on delayed class: `0.50`
- F1 on delayed class: `0.50`

The model metrics are intentionally presented with caution because the dataset is small and synthetic. The point of this repo is the workflow, structure, and explainability, not inflated model claims.

Current NLP evaluation snapshot:

- Task: classify intake summary text into `practice_area`
- Method: `TF-IDF + LogisticRegression`
- Validation: `2-fold stratified cross-validation`
- Accuracy: `0.667`
- Macro precision: `0.60`
- Macro recall: `0.65`
- Macro F1: `0.62`

The NLP metrics are also illustrative for the same reason: the dataset is small and synthetic.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── Tech_Ref.txt
├── project1.py
├── train_assignment_delay_model.py
├── train_practice_area_text_classifier.py
├── requirements.txt
├── data/
│   └── raw/
│       └── legal_intake_sample.csv
├── docs/
│   ├── 8_week_job_ready_plan.md
│   ├── employer_pitch.md
│   └── project_walkthrough.md
├── outputs/
│   ├── cleaned_legal_intake.csv
│   ├── practice_area_summary.csv
│   ├── status_summary.csv
│   ├── executive_metrics.json
│   ├── executive_summary.txt
│   ├── assignment_delay_model_metrics.json
│   ├── assignment_delay_predictions.csv
│   └── assignment_delay_feature_weights.csv
└── sql/
    └── report_queries.sql
```

## Key Files

- `project1.py`: analytics pipeline for cleaning, feature engineering, summaries, and SQLite export
- `train_assignment_delay_model.py`: ML training script for delayed-assignment prediction
- `train_practice_area_text_classifier.py`: NLP text-classification script for routing summaries to practice areas
- `sql/report_queries.sql`: SQL reporting queries against the generated SQLite database
- `docs/project_walkthrough.md`: guided step-by-step explanation of the project
- `Tech_Ref.txt`: study reference for reviewing the code later
- `docs/employer_pitch.md`: resume bullets and interview framing

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 project1.py
python3 train_assignment_delay_model.py
python3 train_practice_area_text_classifier.py
```

Optional SQL review:

```bash
sqlite3 outputs/legal_intake.db < sql/report_queries.sql
```

## Output Artifacts

Running the scripts generates:

- `outputs/cleaned_legal_intake.csv`
- `outputs/practice_area_summary.csv`
- `outputs/status_summary.csv`
- `outputs/executive_metrics.json`
- `outputs/executive_summary.txt`
- `outputs/legal_intake.db`
- `outputs/assignment_delay_model_metrics.json`
- `outputs/assignment_delay_predictions.csv`
- `outputs/assignment_delay_feature_weights.csv`
- `outputs/assignment_delay_model.pkl`
- `outputs/practice_area_text_metrics.json`
- `outputs/practice_area_text_predictions.csv`
- `outputs/practice_area_top_terms.csv`
- `outputs/practice_area_text_classifier.pkl`

## Skills Demonstrated

- Data cleaning and standardization
- Feature engineering for business workflows
- Structured reporting with pandas and SQL
- SQLite persistence and query support
- Supervised learning with preprocessing pipelines
- NLP text classification with TF-IDF features
- Basic model evaluation and prediction inspection
- Technical documentation for portfolio presentation

## How to Talk About This Project

In an interview, the strongest framing is:

1. The business team needs better visibility into intake volume, risk, and assignment timing.
2. The raw data is inconsistent, so data quality has to be fixed first.
3. The analytics pipeline creates clean tables and stakeholder-ready outputs.
4. The ML layer adds a simple operational forecast: whether assignment will be delayed.
5. The NLP layer classifies intake summaries to support practice-area routing.
6. The project is structured for future expansion into richer NLP and dashboards.

## Study and Reference Files

- Guided walkthrough: `docs/project_walkthrough.md`
- Technical study reference: `Tech_Ref.txt`
- Job-targeted learning plan: `docs/8_week_job_ready_plan.md`
- Employer-facing notes: `docs/employer_pitch.md`

## Next Steps

- Add entity extraction from intake summaries
- Add a dashboard in Streamlit, Power BI, or Tableau
- Add tests and stronger data validation
- Compare the current baseline model with a tree-based model
- Expand from TF-IDF to embeddings or transformer-based text models
- Expand the dataset and improve evaluation rigor

## Notes

- The dataset is synthetic and intended for learning and portfolio demonstration.
- The current ML results are illustrative, not production-grade.
- The repo is designed to show sound workflow and communication, not to overstate model performance.
