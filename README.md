# DemandIQ

**Forecast Demand. Detect Risk. Replenish Smarter.**

DemandIQ is an end-to-end retail demand intelligence and replenishment decision-support system developed from a 12-week machine-learning project.

The platform combines demand forecasting, high-demand classification, behavioural segmentation, anomaly detection, explainable AI, replenishment prioritization, API integration, database support, authentication, testing, and a modern web interface.

---

## Project Overview

DemandIQ transforms historical retail data into practical decision-support signals for retail planners.

The system combines:

- Demand forecasting
- High-demand classification
- Behavioural segmentation
- Demand anomaly detection
- Explainable AI
- Replenishment prioritization
- REST API integration
- Web application interface
- Database-backed user and run management
- Batch inference
- Authentication
- Docker-based deployment

DemandIQ is a **decision-support system**, not an autonomous inventory ordering platform.

---

## Project Evolution

| Week | Work Completed |
|------|----------------|
| Week 1 | Data understanding and exploratory data analysis |
| Week 2 | Data cleaning, validation and feature engineering |
| Week 3 | Linear Regression baseline |
| Week 4 | Ridge Regression, Lasso Regression and Logistic Regression |
| Week 5 | Time-aware cross-validation and model selection |
| Week 6 | Interpretable Decision Trees |
| Week 7 | Random Forest regression and classification |
| Week 8 | XGBoost with early stopping and SHAP explainability |
| Week 9 | Behavioural segmentation using K-Means, DBSCAN and PCA |
| Week 10 | Rigorous temporal train-validation-test evaluation |
| Week 11 | Demand anomaly detection |
| Week 12 | Replenishment decision support |

---

## DemandIQ Decision Pipeline

```text
Historical Retail Data
        |
        v
Feature Engineering
        |
        v
------------------------------------------------
Demand Forecast
High-Demand Probability
Behavioural Segmentation
Demand Anomaly Detection
------------------------------------------------
        |
        v
Demand Volatility
        +
Promotion Signal
        |
        v
Replenishment Priority Score
        |
        v
CRITICAL / HIGH / MEDIUM / LOW
        |
        v
Planner Decision Support
```

---

## Final Model Performance

The final temporally evaluated XGBoost pipeline reports the following historical experiment results.

### Demand Regression

| Metric | Value |
|--------|------:|
| MAE | 66.1139 |
| RMSE | 262.4817 |
| R² | 0.9637 |

### High-Demand Classification

| Metric | Value |
|--------|------:|
| F1 Score | 0.9618 |
| ROC-AUC | 0.9976 |
| PR-AUC | 0.9952 |
| Probability Threshold | 0.66 |

> These figures are historical experiment outputs and should not be interpreted as live retailer performance claims.

---

## Replenishment Decision Rule

DemandIQ combines multiple signals into a transparent replenishment-priority score:

```text
0.35 × Demand Score
+ 0.30 × High-Demand Probability
+ 0.15 × Demand Volatility
+ 0.10 × Anomaly Signal
+ 0.10 × Promotion Signal
```

### Priority Thresholds

| Score | Priority | Action |
|------|----------|--------|
| >= 0.75 | CRITICAL | Review / replenish first |
| >= 0.55 | HIGH | Prioritize replenishment |
| >= 0.35 | MEDIUM | Monitor closely |
| < 0.35 | LOW | Normal monitoring |

---

## Final Replenishment Output

The final Week 12 historical decision-support output contains:

- **1,782** store × product-family records
- **429** Critical-priority records
- Decision-support date: **2017-08-15**

These outputs come from the historical project dataset and are not live retailer operations.

---

## Anomaly Detection Output

The Week 11 anomaly artifact contains:

- **98,702 detected anomalies**

DemandIQ uses anomaly signals to identify unusual demand spikes or drops that may require planner attention.

---

## Application Features

DemandIQ includes:

- Public landing page
- Demand command center
- Single-demand prediction
- Direct 23-feature inference
- Batch CSV inference
- High-demand probability estimation
- Replenishment priority queue
- Demand anomaly intelligence
- Product-family behavioural segmentation
- Store behavioural segmentation
- Model performance view
- XGBoost feature importance
- SHAP explainability
- Architecture view
- User authentication
- Prediction-run persistence
- REST API
- Docker-based deployment

---

## System Architecture

```text
Browser / Next.js
        |
        v
FastAPI /api/v1
        |
        +--> PostgreSQL
        |    |
        |    +--> Users
        |    +--> Prediction Runs
        |    +--> Predictions
        |
        +--> Result Artifacts
        |    |
        |    +--> Metrics
        |    +--> Clusters
        |    +--> Anomalies
        |    +--> Replenishment Priorities
        |
        +--> ML Service
             |
             +--> Preprocessor
             +--> XGBoost Regressor
             +--> XGBoost Classifier
```

---

## Technology Stack

| Layer | Technology |
|------|------------|
| Frontend | Next.js, TypeScript |
| Backend | FastAPI, Python |
| Machine Learning | scikit-learn, XGBoost, SHAP |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Migrations | Alembic |
| Testing | pytest |
| Deployment | Docker, Docker Compose |

---

## Repository Structure

```text
DemandIQ/
├── experiments/                 # Original Week 1–12 research code
├── models/
│   ├── week10_model_bundle.joblib
│   └── model_manifest.json
├── data/
│   ├── raw/
│   ├── processed/
│   └── demo/
├── reports/
│   ├── results/
│   └── figures/
├── backend/
├── frontend/
├── database/
│   └── alembic/
├── tests/
├── docker/
├── docs/
├── docker-compose.yml
└── README.md
```

---

## Model Input Contract

The final batch inference pipeline expects the following 23 features:

```text
store_nbr
family
city
state
type
cluster
onpromotion
transactions
dcoilwtico
year
month
day
day_of_week
week_of_year
is_weekend
is_holiday
lag_1
lag_7
lag_14
lag_28
rolling_mean_7
rolling_mean_14
rolling_mean_28
```

A valid example batch file is available in:

```text
data/demo/sample_batch.csv
```

---

## Model Compatibility

The serialized Week 10 model bundle was created using approximately:

```text
Python        3.13.x
scikit-learn  1.9.0
xgboost       3.4.1
numpy         2.5.2
pandas        3.0.5
joblib        1.6.0
```

These ML versions are pinned in:

```text
backend/requirements.txt
```

Do not ignore model serialization compatibility warnings.

If the model is migrated to another environment, re-save the artifact deliberately and compare predictions against the fixed model regression fixture in the test suite.

---

## Quick Start with Docker

### 1. Create environment file

Copy:

```text
.env.example
```

to:

```text
.env
```

Update the JWT secret and PostgreSQL password before deployment.

### 2. Start the application

```bash
docker compose up --build
```

### 3. Open the services

Frontend:

```text
http://localhost:3000
```

API:

```text
http://localhost:8000
```

Swagger / OpenAPI:

```text
http://localhost:8000/docs
```

---

## Local Backend Setup

Create a Python 3.13 virtual environment.

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

### PowerShell

```powershell
$env:PYTHONPATH="backend"
uvicorn app.main:app --app-dir backend --reload --port 8000
```

If `DATABASE_URL` is not set, the local development environment may use SQLite.

Production deployment should use PostgreSQL.

---

## Local Frontend Setup

Move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Create:

```text
frontend/.env.local
```

Add:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

Start the frontend:

```bash
npm run dev
```

Then open:

```text
http://localhost:3000
```

---

## Database Migrations

Set `DATABASE_URL`, then run:

```bash
PYTHONPATH=backend alembic upgrade head
```

Alembic is the intended production migration mechanism.

---

## API Highlights

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/health` | API and model health |
| GET | `/api/v1/model/info` | Feature contract and model information |
| GET | `/api/v1/model/performance` | Historical evaluation metrics |
| GET | `/api/v1/model/explainability` | Feature importance and SHAP references |
| POST | `/api/v1/predictions/demand` | Single demand prediction |
| POST | `/api/v1/predictions/direct` | Direct 23-feature inference |
| POST | `/api/v1/predictions/batch` | Batch CSV inference |
| GET | `/api/v1/analytics/summary` | Dashboard KPIs |
| GET | `/api/v1/anomalies` | Detected demand anomalies |
| GET | `/api/v1/replenishment` | Replenishment priority queue |
| GET | `/api/v1/segments` | Behavioural segmentation output |
| POST | `/api/v1/auth/register` | Register a user |
| POST | `/api/v1/auth/login` | Login and receive token |
| GET | `/api/v1/auth/me` | Current authenticated user |
| GET | `/api/v1/runs` | List persisted prediction runs |
| GET | `/api/v1/runs/{id}` | Inspect a prediction run |

---

## Batch CSV Contract

Batch inference validates the exact 23-column model input contract.

A valid one-row example is stored in:

```text
data/demo/sample_batch.csv
```

and is also available to the frontend as:

```text
frontend/public/sample_batch.csv
```

---

## Testing

Run the backend and model test suite using:

```bash
PYTHONPATH=backend pytest tests -q
```

The project test suite covers:

- Model loading
- 23-feature contract
- Fixed numerical regression fixture
- API health
- Analytics summary
- Forecast behavior
- Batch inference
- Prediction-run persistence
- Authentication helpers
- Token security
- Replenishment threshold boundaries

Current project verification reports:

```text
21 passing tests
```

---

## Security Notes

- Never commit `.env`
- Replace example JWT secrets before public deployment
- Restrict public batch upload size and row count
- Do not accept arbitrary model or pickle uploads
- Use parameterized SQLAlchemy database access
- Configure allowed CORS origins
- Use TLS in production
- Add provider-level rate limiting when deployed publicly

---

## Important Limitations

DemandIQ is a research-derived and portfolio-oriented retail decision-support system.

The available dataset does not include:

- Real-time inventory-on-hand
- Supplier lead times
- Service-level targets
- Verified stockout labels
- Live retailer transactions

Therefore:

- DemandIQ does **not** calculate exact order quantities
- DemandIQ does **not** claim to predict real stockouts
- The classifier represents a **high-demand or demand-pressure signal**
- Replenishment priorities identify store × product-family combinations that may deserve additional planner attention

Feature importance and SHAP explain model behavior but do not prove causality.

---

## Future Extensions

Possible future improvements include:

- Live sales-data ingestion
- Automated retraining
- Model monitoring
- Drift detection
- Forecast confidence intervals
- Model registry integration
- Asynchronous large-batch inference
- Object storage
- Retailer-specific authentication and tenancy
- Inventory-on-hand integration
- Supplier lead-time integration
- Service-level optimization
- Exact replenishment optimization
- Optional PostgreSQL `pgvector` support for future semantic document search

---

## Final Outcome

DemandIQ transforms historical retail data into:

```text
Demand Forecasts
        +
High-Demand Alerts
        +
Behavioural Insights
        +
Demand Anomaly Signals
        +
Replenishment Priorities
        |
        v
Smarter Planner Decisions
```

---

## Final Project Statement

**DemandIQ is an end-to-end retail demand intelligence platform that combines machine learning, explainability, anomaly detection, behavioural segmentation, and transparent replenishment-priority rules to support better retail planning decisions.**

### Forecast Demand. Detect Risk. Replenish Smarter.
