# Deployment

## Recommended topology

- Frontend: Vercel, Cloud Run, or container platform
- Backend: container service with at least enough memory to hold the XGBoost models and preprocessors
- Database: managed PostgreSQL 16+
- Secrets: platform secret manager/environment variables

## Required production configuration

Set `DATABASE_URL`, a strong `JWT_SECRET`, and exact `CORS_ORIGINS`. Build the backend using `docker/backend.Dockerfile` so the original ML dependency versions are installed.

## Model artifact

The `.joblib` file is ~18 MB. Git hosts may accept it directly, but Git LFS or release/object storage is preferable if your repository policy requires it. The service must receive the artifact at `models/week10_model_bundle.joblib` unless `MODEL_PATH` is overridden.

## Database

Run `alembic upgrade head` before the first production deployment. The automatic `create_all` call is retained for local convenience, not as a replacement for controlled migrations.

## Health checking

Use `GET /api/v1/health`. A healthy response confirms the model artifact loaded. Compatibility warnings should be treated as deployment issues, not hidden.
