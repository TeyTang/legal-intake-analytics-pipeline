# 8-Week Job-Ready Plan for the Womble Bond Dickinson AI/ML Developer Role

This plan is designed around the actual posting: Python, SQL, NLP, dashboards, documentation, business communication, and production-minded delivery.

## Week 1: Python, pandas, and project setup

### Day 1
- Create the repo structure
- Set up `venv`, `requirements.txt`, and `.gitignore`
- Review the sample dataset and identify data quality issues

### Day 2
- Practice `pandas` basics: loading CSV, inspecting schema, filtering, sorting
- Document what each field means in the dataset

### Day 3
- Handle missing values, duplicates, inconsistent text labels, and bad types
- Write down every cleaning decision in plain English

### Day 4
- Add feature engineering such as risk flags, priority categories, and text length
- Explain why each feature could matter to a stakeholder

### Day 5
- Refactor the script into reusable functions
- Run the pipeline end to end

### Day 6
- Read your own code critically and simplify anything unclear
- Update the README with project purpose and outputs

### Day 7
- Push the repo to GitHub
- Write a short LinkedIn-style summary of what the project does

## Week 2: SQL, SQLite, and reporting

### Day 8
- Learn `sqlite3` basics and how `to_sql` works in `pandas`
- Save cleaned data into a SQLite database

### Day 9
- Practice SQL `SELECT`, `WHERE`, `ORDER BY`, and `LIMIT`
- Run reporting queries against the generated database

### Day 10
- Practice `GROUP BY`, aggregates, and filtering with `HAVING`
- Build a summary table for practice areas

### Day 11
- Practice joins by creating at least one secondary reference table
- Add one new SQL query for a stakeholder question

### Day 12
- Compare doing analysis in `pandas` versus SQL
- Note when each tool is more useful

### Day 13
- Add a `sql/report_queries.sql` file with polished reporting queries
- Make sure query names map to real business questions

### Day 14
- Write a short markdown note explaining the database schema
- Practice explaining the project without reading from notes

## Week 3: Machine learning foundations

### Day 15
- Learn train/validation/test split
- Read about leakage and class imbalance

### Day 16
- Create a basic supervised learning problem from a tabular dataset
- Use `scikit-learn` preprocessing and pipelines

### Day 17
- Train a logistic regression model
- Evaluate with confusion matrix, precision, recall, and F1

### Day 18
- Train a tree-based model and compare results
- Explain tradeoffs in model complexity and interpretability

### Day 19
- Add feature importance or coefficient interpretation
- Write model results in plain English

### Day 20
- Review overfitting, underfitting, and validation strategy
- Rebuild the experiment cleanly from scratch

### Day 21
- Summarize what metrics matter most for legal or risk-sensitive workflows
- Add notes to GitHub

## Week 4: Business-ready ML workflow

### Day 22
- Learn about data drift, monitoring, and model retraining triggers
- Connect these ideas to business operations

### Day 23
- Add a second notebook or script for a small predictive task
- Keep the code simple and reproducible

### Day 24
- Write down model assumptions and limitations
- Add a section on ethical or operational risks

### Day 25
- Practice describing a model to a non-technical stakeholder
- Record a 2-minute explanation for yourself

### Day 26
- Add metrics reporting to JSON or CSV output
- Make sure an executive could read the results quickly

### Day 27
- Review SQL and `pandas` again
- Tighten any weak spots

### Day 28
- Publish a polished GitHub update
- Capture screenshots of outputs for the README

## Week 5: NLP for document-heavy work

### Day 29
- Learn tokenization, stop words, lemmatization, and text cleaning
- Practice on the intake `summary` field

### Day 30
- Use `spaCy` for basic NLP processing
- Extract entities, keywords, and sentence counts

### Day 31
- Build a simple text classification baseline
- Start with TF-IDF plus logistic regression

### Day 32
- Evaluate classification errors
- Review what kinds of legal text are hard to categorize

### Day 33
- Add a small NLP output to the project or a companion project
- Keep the result inspectable and easy to explain

### Day 34
- Read about privacy and handling sensitive text
- Note where human review is required

### Day 35
- Write a README section called `NLP roadmap`
- Explain how the pipeline can evolve from rules to models

## Week 6: Modern NLP and LLM application skills

### Day 36
- Learn Hugging Face pipelines and transformer inference basics
- Run summarization or classification on a small example

### Day 37
- Learn embeddings and semantic similarity
- Compare keyword matching to semantic search

### Day 38
- Build a tiny document search or FAQ prototype
- Keep source citations visible

### Day 39
- Learn the basics of retrieval-augmented generation
- Focus on grounded answers, not flashy demos

### Day 40
- Write down hallucination risks and failure modes
- Add guardrails and review guidance

### Day 41
- Compare classical NLP, transformer inference, and RAG
- Practice when you would choose each one

### Day 42
- Update GitHub with a second project idea or branch
- Make sure the scope is realistic

## Week 7: Dashboard and delivery

### Day 43
- Choose one dashboard tool: Power BI, Tableau, or Streamlit
- Use the generated CSV outputs as your source tables

### Day 44
- Build KPI cards and one trend chart
- Show matter counts, risk, and assignment delay

### Day 45
- Add one breakdown by practice area and one by intake channel
- Keep the visuals clean and readable

### Day 46
- Add filters and a short narrative summary
- Make the dashboard useful for a CIO or operations lead

### Day 47
- If using Python, build a simple `FastAPI` or Streamlit app
- Keep deployment simple and local-first

### Day 48
- Add screenshots or a short demo GIF to the README
- Make the repo easy to scan in under 2 minutes

### Day 49
- Review the entire project as if you were the hiring manager
- Remove weak or unfinished pieces

## Week 8: Portfolio polish and interview prep

### Day 50
- Finalize your README and project story
- Add architecture notes and assumptions

### Day 51
- Practice explaining the repo in STAR format
- Focus on business problem, approach, result, and next steps

### Day 52
- Practice common interview topics:
  Python, SQL, metrics, overfitting, NLP, data quality, privacy

### Day 53
- Practice stakeholder communication
- Explain the project to a non-technical friend in 3 minutes

### Day 54
- Review data privacy, model governance, and human-in-the-loop controls
- Prepare one answer on responsible AI

### Day 55
- Clean up commit history and repo presentation
- Double-check instructions, screenshots, and links

### Day 56
- Submit applications with the repo linked
- Tailor your resume bullets to match the project outcomes

## Recommended portfolio sequence

1. This repo: legal intake analytics pipeline
2. Next repo: NLP document classifier or semantic search assistant

That combination will present much better for this role than a generic beginner ML notebook collection.
