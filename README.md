# Legal Intake Analytics Pipeline

This repository is a portfolio project designed to show practical AI/ML-adjacent skills for a business-facing role. It simulates a legal matter intake workflow by loading raw intake data, cleaning it, deriving priority signals, analyzing trends, and storing outputs in SQLite for downstream reporting.

## Why this project fits the target role

This project maps directly to the responsibilities in the AI and Machine Learning Developer posting:

- Python-based data processing pipeline
- SQL and SQLite data storage
- Data quality checks and preprocessing
- Analytics for stakeholder reporting
- Foundation for later NLP work on unstructured legal text
- Documentation that explains both technical and business value

## What the pipeline does

The script in `project1.py`:

1. Loads a sample legal intake dataset from `data/raw/legal_intake_sample.csv`
2. Cleans inconsistent values and missing fields
3. Creates business-oriented features such as `priority_flag`, `is_high_risk`, and `summary_topic`
4. Builds summary tables for practice-area reporting and status monitoring
5. Saves cleaned data and summaries to a SQLite database
6. Exports CSV and JSON artifacts for reporting

## Repo structure

```text
.
├── README.md
├── project1.py
├── train_assignment_delay_model.py
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       └── legal_intake_sample.csv
├── docs/
│   ├── 8_week_job_ready_plan.md
│   ├── employer_pitch.md
│   └── project_walkthrough.md
├── outputs/
│   └── generated files after running the pipeline
└── sql/
    └── report_queries.sql
```

## Skills demonstrated

- Python scripting
- `pandas` data cleaning and feature engineering
- SQLite persistence with `sqlite3`
- `scikit-learn` preprocessing pipelines and logistic regression
- Model metrics, predictions, and explainability artifacts
- Business KPI generation
- Documentation and project organization
- Stakeholder-facing analytics framing

## How to run

1. Create and activate a virtual environment
2. Install dependencies
3. Run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 project1.py
python3 train_assignment_delay_model.py
```

For a guided explanation of each step, see `docs/project_walkthrough.md`.
For a study-focused technical reference, see `Tech_Ref.txt`.

## Expected outputs

After running the script, the `outputs/` folder will contain:

- `cleaned_legal_intake.csv`
- `practice_area_summary.csv`
- `status_summary.csv`
- `executive_metrics.json`
- `executive_summary.txt`
- `legal_intake.db`
- `assignment_delay_model_metrics.json`
- `assignment_delay_predictions.csv`
- `assignment_delay_feature_weights.csv`
- `assignment_delay_model.pkl`

## Example business questions this project answers

- Which practice areas are generating the most intake volume?
- Which matters should be prioritized based on risk or estimated value?
- How long does assignment take on average?
- Which matters are at risk of delayed assignment?
- Which intake channels are driving the most work?
- Which topics appear most often in unstructured intake summaries?

## Machine learning workflow

The second script, `train_assignment_delay_model.py`, trains an explainable logistic regression model to predict whether a matter will be assigned late. The current target is `assignment_delay_flag`, which marks matters with more than 3 days to assignment as `delayed`.

Because the dataset is small and synthetic, the model metrics are illustrative. The value of this repo is showing the workflow clearly: problem framing, data preparation, feature engineering, modeling, metrics, and documentation.

## Next project expansions

This project is intentionally scoped as a strong foundation. Good follow-on phases include:

- Add NLP classification for intake summaries using text features directly
- Build a dashboard in Power BI, Tableau, or Streamlit
- Add FastAPI endpoints for model or analytics access
- Replace rule-based topic tagging with embeddings or transformers
- Add unit tests and data validation rules

## How to present this to employers

Frame it as a business-oriented AI/ML project, not just a script:

- You built a reusable data pipeline
- You handled messy intake data and created reporting tables
- You added an explainable ML baseline for operational forecasting
- You designed outputs for stakeholders, not just developers
- You structured the repo like a real project with documentation and SQL artifacts
- You planned the next steps toward NLP and dashboard delivery
