# Production refactor summary

The original DemandIQ research files under `experiments/`, `models/`, `data/`, and `reports/` were preserved. The production refactor added:

- `backend/` — FastAPI, Pydantic validation, SQLAlchemy persistence, auth, batch inference, rate limiting, model/data services.
- `frontend/` — responsive Next.js + TypeScript product interface with landing page, dashboard, forecast, batch, anomalies, replenishment, segmentation, performance, explainability, architecture, login and registration.
- `database/alembic/` — initial production migration.
- `tests/` — 21 backend/model/security/API/business-rule tests.
- `docker/` and `docker-compose.yml` — PostgreSQL + backend + frontend container topology.
- `.github/workflows/ci.yml` — automated backend tests and frontend test/typecheck/build.
- `data/demo/` — 135,432 genuine processed rows from 2017-06-01 through 2017-08-15 for fast public demo use.
- `docs/` and Windows PowerShell helper scripts.

No separate vector database was added. PostgreSQL remains the primary datastore; pgvector is reserved for a future unstructured semantic-search feature if one is actually needed.
