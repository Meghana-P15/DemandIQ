# DemandIQ

**Forecast Demand. Detect Risk. Replenish Smarter.**

DemandIQ is a production-oriented retail demand intelligence platform built from the original 12-week machine-learning project in this repository. The application preserves the research experiments and serves the authentic Week 10 XGBoost model bundle, Week 9 clustering artifacts, Week 11 anomaly outputs, and Week 12 replenishment rules through a modern web product.

## What is real in the demo

The web application does not hard-code fake model results. It reads the original project artifacts and runs the saved model for inference. The current temporal evaluation artifact reports:

- Regression MAE: **66.1139**
- Regression RMSE: **262.4817**
- Regression R²: **0.9637**
- Classification F1: **0.9618**
- Classification ROC-AUC: **0.9976**
- Classification PR-AUC: **0.9952**
- High-demand probability threshold: **0.66**

The anomaly artifact contains **98,702** detected anomalies across the evaluated history. The final replenishment output contains **1,782** store × family rows for 2017-08-15, including **429 Critical** priorities. These figures are historical experiment outputs, not live business claims.

## Product surface

The UI includes a public landing page, demand command center, single-demand forecasting, batch CSV inference, replenishment decision support, anomaly intelligence, behavioural segmentation, model performance, model explainability, and architecture views. A curated demo slice contains 135,432 genuine processed observations from 2017-06-01 through 2017-08-15 so the product can be demonstrated without loading the full multi-million-row feature file on every request.

## Architecture

```text
Browser / Next.js
        |
        v
FastAPI /api/v1
  |       |        |
  |       |        +--> PostgreSQL (users, runs, predictions)
  |       +-----------> Result artifacts (metrics, clusters, anomalies, priorities)
  +-------------------> ML service -> preprocessor -> XGBoost regressor/classifier
```

The core product uses **PostgreSQL**, not a vector database. DemandIQ currently operates on structured/tabular/time-series data. If a future `Ask DemandIQ` feature searches unstructured reports or operational notes, `pgvector` can be added inside PostgreSQL without introducing a separate vector system prematurely.

## Repository layout

```text
DemandIQ/
├── experiments/                 # original Week 1–12 research code (preserved)
├── models/
│   ├── week10_model_bundle.joblib
│   └── model_manifest.json
├── data/
│   ├── raw/                     # original data; ignored by Git
│   ├── processed/               # full feature file; ignored by Git
│   └── demo/                    # small production demo slice + sample batch CSV
├── reports/
│   ├── results/                 # genuine result CSV artifacts
│   └── figures/                 # original experiment figures
├── backend/                     # FastAPI + SQLAlchemy application
├── frontend/                    # Next.js + TypeScript product UI
├── database/alembic/            # migrations
├── tests/                       # backend, model and business-logic tests
├── docker/
├── docs/
└── docker-compose.yml
```

## Model compatibility

The serialized `week10_model_bundle.joblib` was created in the original project environment. The version folders in that environment indicate:

```text
scikit-learn 1.9.0
xgboost       3.4.1
numpy         2.5.2
pandas        3.0.5
joblib        1.6.0
Python        3.13.x
```

Those ML versions are pinned in `backend/requirements.txt`. Do not simply suppress serialization compatibility warnings. If the artifact is migrated later, export/re-save it deliberately and compare predictions against the fixed regression fixture in `tests/test_ml.py`.

## Quick start with Docker

1. Copy `.env.example` to `.env`.
2. Replace `JWT_SECRET` and set a PostgreSQL password for any public deployment.
3. Run:

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- Swagger/OpenAPI: `http://localhost:8000/docs`

## Local backend

Create a Python 3.13 virtual environment and install:

```bash
pip install -r backend/requirements.txt
```

For a zero-setup local run, if `DATABASE_URL` is not set, the app uses a local SQLite development database. Production should use PostgreSQL.

macOS/Linux:

```bash
PYTHONPATH=backend uvicorn app.main:app --app-dir backend --reload --port 8000
```

PowerShell:

```powershell
$env:PYTHONPATH="backend"
uvicorn app.main:app --app-dir backend --reload --port 8000
```

## Local frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

Then:

```bash
npm run dev
```

## Database migrations

Set `DATABASE_URL`, then run from the repository root:

```bash
PYTHONPATH=backend alembic upgrade head
```

FastAPI also creates missing tables on application startup for local convenience, while Alembic is the intended production migration path.

## API highlights

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | API/model health |
| GET | `/api/v1/model/info` | exact feature contract and runtime compatibility |
| GET | `/api/v1/model/performance` | genuine evaluation artifacts |
| GET | `/api/v1/model/explainability` | real feature importance + SHAP artifact references |
| POST | `/api/v1/predictions/demand` | demo-history-assisted single inference |
| POST | `/api/v1/predictions/direct` | direct 23-feature inference |
| POST | `/api/v1/predictions/batch` | validated CSV inference |
| GET | `/api/v1/analytics/summary` | dashboard KPIs |
| GET | `/api/v1/anomalies` | filterable anomaly output |
| GET | `/api/v1/replenishment` | filterable decision-support queue |
| GET | `/api/v1/segments` | family/store K-Means + PCA outputs |
| POST | `/api/v1/auth/register` | create account |
| POST | `/api/v1/auth/login` | issue signed HS256 bearer token |
| GET | `/api/v1/auth/me` | inspect authenticated user |
| GET | `/api/v1/runs` | list persisted prediction runs |
| GET | `/api/v1/runs/{id}` | inspect a persisted run |

## Batch CSV contract

Batch inference validates the exact 23-column model input contract:

`store_nbr, family, city, state, type, cluster, onpromotion, transactions, dcoilwtico, year, month, day, day_of_week, week_of_year, is_weekend, is_holiday, lag_1, lag_7, lag_14, lag_28, rolling_mean_7, rolling_mean_14, rolling_mean_28`

A valid one-row example is in `data/demo/sample_batch.csv` and is also copied to `frontend/public/sample_batch.csv`.

## Tests

```bash
PYTHONPATH=backend pytest tests -q
```

The suite currently contains **21 passing tests** covering model loading, the 23-feature contract, a fixed numerical model regression fixture, API health/summary/forecast/batch behavior, run persistence, password/token security helpers, and exact replenishment threshold boundaries (`0.75`, `0.55`, `0.35` plus just-below values). See `docs/VERIFICATION.md` for the executed verification report.

Frontend CI is configured to run TypeScript checking and a production Next.js build after installing dependencies.

## Replenishment decision rule

The Week 12 score is preserved exactly:

```text
0.35 × demand score
+ 0.30 × high-demand risk
+ 0.15 × volatility
+ 0.10 × anomaly signal
+ 0.10 × promotion score
```

Thresholds:

- `>= 0.75` → **CRITICAL** — Review / replenish first
- `>= 0.55` → **HIGH** — Prioritize replenishment
- `>= 0.35` → **MEDIUM** — Monitor closely
- `< 0.35` → **LOW** — Normal monitoring

## Security notes

- Never commit `.env`.
- Replace the example JWT secret before any public deployment.
- Public batch inference is deliberately limited by file size and row count.
- The API never accepts arbitrary `.joblib`/pickle model uploads.
- SQLAlchemy parameterizes database access.
- CORS origins are configurable.
- Production deployments should terminate TLS at the hosting platform/reverse proxy and add provider-level rate limiting.

## Limitations

DemandIQ is a portfolio/research-derived system, not an autonomous inventory ordering system. Forecasts and replenishment priorities are decision-support signals. The demo uses historical Favorita-style retail data from the original project and should not be presented as a live retailer. Feature importance and SHAP visualizations explain model behavior but do not establish causality.

## Future extensions

Potential next steps include asynchronous large-batch jobs, object storage for uploads, model registry/monitoring, drift detection, calibrated forecast intervals, retailer-specific authentication/tenancy, and optional PostgreSQL `pgvector` only if semantic document retrieval becomes a real product requirement.
