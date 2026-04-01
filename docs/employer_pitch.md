# Employer Pitch Notes

## Suggested GitHub repository subtitle

Business-oriented legal intake analytics pipeline built with Python, pandas, SQL, and SQLite, designed as a foundation for NLP and dashboard reporting.

## Resume bullet ideas

- Built a Python and SQLite legal intake analytics pipeline that cleans raw matter data, engineers priority features, and generates stakeholder-ready KPI summaries.
- Developed reproducible reporting outputs using `pandas`, SQL, and SQLite to track intake volume, assignment delays, risk levels, and practice-area trends.
- Trained an explainable logistic regression baseline to predict delayed matter assignment and exported model metrics, predictions, and feature-weight artifacts for review.
- Added an NLP text-classification workflow using TF-IDF and logistic regression to route intake summaries to practice areas and inspect the strongest terms by class.
- Added a rule-based legal entity extraction workflow to turn intake summaries into structured issue, document, party, and workflow terms for downstream analysis.
- Structured a portfolio project with documentation, SQL reporting queries, and executive-style summary artifacts to simulate an applied AI/ML workflow in a regulated business environment.

## How to talk about this project in an interview

Use this structure:

1. Business problem
   A firm receives intake requests from different teams and channels, but the data is messy and hard to analyze consistently.

2. What you built
   You created a small pipeline that standardizes the data, derives priority signals, stores cleaned records in SQLite, produces reporting tables, trains a simple model to predict delayed assignment, adds NLP-based practice-area routing from intake summaries, and extracts structured legal terms from the text.

3. Why it matters
   This reduces manual cleanup time, improves reporting consistency, and creates a clean base for future NLP or predictive modeling work.

4. What you would add next
   Semantic search over the intake summaries, a dashboard layer, model monitoring, and stronger data validation rules.

## Good phrasing for stakeholders

- I designed the outputs so a business stakeholder could review volume, risk, and assignment speed without reading the raw data.
- I treated data quality as part of the solution, not as a separate cleanup step.
- I used simple, explainable logic first so the project has a strong operational foundation before adding heavier ML.
