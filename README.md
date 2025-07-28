# 🚨 Real-Time Comment Moderation with MLOps

> An end-to-end machine learning pipeline to detect and flag inappropriate online comments - in real time - with built-in monitoring, explainability and live inference.

This project implements a production-grade MLOps workflow using modern Python tools and AWS infrastructure. It demonstrates how NLP-based moderation systems can be developed, deployed and maintained collaboratively with full lifecycle support.

---

## Business Problem

Online platforms face growing pressure to moderate user-generated content for safety, civility and compliance. Our goal is to build an end-to-end MLOps pipeline that flags inappropriate comments in real time - with explainability, monitoring and automation - to support safer user interactions at scale.

---

## Dataset

- **Source**: [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) (Wikipedia talk pages)
- **Volume**: 160K training samples, 64K test samples
- **Target**: `moderation_label = 1` if comment is harmful (based on 6 toxicity types)
- **Note**: ~10% of comments are flagged → requires class imbalance handling

---

## Modeling Options

| Approach                 | Description                                         |
|--------------------------|-----------------------------------------------------|
| TF-IDF + Logistic Regression | Simple, fast, interpretable baseline             |
| GloVe / FastText + LSTM  | Adds semantic richness to text, but non-contextual |
| DistilBERT (Transformer) | Contextual, robust, ideal for nuanced phrasing     |
| AutoML (e.g., FLAML, H2O) | Automatically selects best model + tuning strategy |

---

## “Changed” Test Data

To test model robustness in real-world conditions:
- We modify test comments using techniques like synonym swaps or tone softening
- Re-run predictions and capture metric changes
- Monitor distribution drift and flag outliers using Evidently

---

## Tech Stack

| Layer              | Tool/Service                        |
|--------------------|-------------------------------------|
| Modeling           | DistilBERT + AutoML (FLAML/H2O)     |
| Experiment Tracking| MLflow                              |
| Monitoring         | Evidently                           |
| Model Serving      | FastAPI + Docker                    |
| Deployment         | AWS EC2 (Dockerized)                |
| Frontend           | Streamlit                           |
| Dev Tools          | `uv`, `ruff`, `pre-commit`          |

---

## Local Development

1. **Clone the repo (using SSH)**:
    ```bash
    git clone git@github.com:sohammandal/mlops-comment-moderation.git
    cd mlops-comment-moderation
    ```

2. **Install [uv](https://github.com/astral-sh/uv)**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. **Install dependencies and set up virtual environment**:
    ```bash
    uv sync
    source .venv/bin/activate
    ```

4. **Run linters and hooks**:
    ```bash
    pre-commit install
    pre-commit run --all-files
    ```

---

## Project Structure

```
mlops-comment-moderation/
├── data/                     # Local data only (gitignored)
├── docker/                   # Dockerfile and Compose setup
├── notebooks/                # EDA, experiments
├── src/
│   ├── api/                  # FastAPI backend
│   ├── monitoring/           # Evidently checks and reports
│   ├── train/                # Training, preprocessing, evaluation
│   └── ui/                   # Streamlit frontend
├── .env.example              # Sample environment config (copy to .env)
├── .pre-commit-config.yaml   # Pre-commit hooks for linting/formatting
├── pyproject.toml            # Project and dev config
└── uv.lock                   # Locked dependencies
```

---

## Monitoring & Evaluation

- Model metrics are logged and visualized using **MLflow**
- Prediction and data drift are tracked using **Evidently**
- Changes to test data are measured for impact and flagged when behavior shifts
