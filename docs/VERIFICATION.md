# Verification report

This repository was exercised after the production refactor.

## Completed checks

- Backend/model/business-logic suite: **21 tests passed**.
- FastAPI application import and startup: passed.
- `/api/v1/health`: returned HTTP 200 with the model loaded.
- Real prediction endpoint: scored genuine DemandIQ feature rows using `week10_model_bundle.joblib`.
- Valid one-row batch CSV: scored successfully and persisted a run.
- Invalid batch CSV: rejected with missing-column diagnostics.
- Replenishment boundaries: tested at 0.75 / 0.749, 0.55 / 0.549, 0.35 / 0.349.
- Auth hashing/token round trip and register/login/me route: passed.
- SQLAlchemy run persistence: exercised through API tests.
- Alembic `0001_initial` migration: applied successfully to a clean database and produced the expected application tables.
- Backend Python compile check: passed.
- Frontend TypeScript/TSX source: **26 files transpile-parse with zero syntax diagnostics** using the installed TypeScript compiler.

## ML compatibility note

The execution environment used for these checks has scikit-learn 1.8.0 and XGBoost 3.1.3, while the original project archive records scikit-learn 1.9.0 and XGBoost 3.4.1. The serialized artifact still loaded and the fixed numerical regression fixture passed here, but the production requirements intentionally pin the **original artifact environment** rather than hiding compatibility warnings.

## Frontend dependency/build note

A full `npm install`/Next.js production build could not be completed in this execution environment because outbound npm dependency retrieval timed out. The code was syntax-checked locally, and GitHub Actions is configured to run `npm install`, `npm run test`, `npm run typecheck`, and `npm run build` in a networked CI environment.

This limitation is environmental; it is not represented as a successful frontend build.
