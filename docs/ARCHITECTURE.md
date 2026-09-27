# DemandIQ Architecture

## Design goals

DemandIQ separates research code from production serving. `experiments/` is immutable historical research; production inference lives behind `backend/app/services/`. API routes validate and orchestrate rather than embedding model logic.

## Runtime components

1. **Next.js frontend** — responsive public landing page and analytics product surface.
2. **FastAPI backend** — versioned REST API, validation, upload limits, auth, OpenAPI.
3. **ML service** — loads `week10_model_bundle.joblib` once and enforces the 23-feature contract.
4. **Artifact data service** — exposes genuine Week 8–12 CSV results without recomputing or inventing metrics.
5. **PostgreSQL** — application state such as users, datasets, runs and predictions.
6. **Demo history** — curated genuine processed rows used to derive lag/rolling context for portfolio demonstrations.

## Why no vector database?

The current workload is structured retail/time-series analysis. Relational storage and model-serving APIs are a better fit than semantic vector search. If unstructured notes/reports become searchable later, add `pgvector` within PostgreSQL first.

## Trust boundaries

User CSVs are treated as untrusted input. The backend limits size/rows, checks file extension, parses server-side, validates required columns, and never deserializes user-provided model files.
